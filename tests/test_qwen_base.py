import jax
import flax.linen as nn
from jax.sharding import Mesh
from jax.sharding import NamedSharding, PartitionSpec as P
import jax.numpy as jnp
import gc
from safetensors import safe_open
import os
import json

devices = jax.devices()
devices_mesh = devices.reshape(1,2)
# =====================================================================
# ШАГ 1: НАСТРОЙКА ГЕОМЕТРИИ ЖЕЛЕЗА (SHARDING)
# =====================================================================

# 1. Объединяем 8 чипов TPU в логическую сеть
# Ось 'tensor' отвечает за распиливание матриц Qwen/Mamba на 8 частей
mesh = Mesh(devices_mesh, ('data', 'tensor'))

# 1. Маска репликации (Полное дублирование на каждый чип)
# P(None, None) означает: не резать ни строки, ни столбцы.
# Применяется для векторов нормализации (RMSNorm) и маленьких параметров.
sharding_repl = NamedSharding(mesh, P(None, None))    # Дублировать везде

# 2. Вертикальная маска (Нарезка по столбцам)
# P(None, 'tensor') означает: строки не трогать, а столбцы распилить вдоль оси 'tensor' на 8 частей.
# Применяется для входных слоев: gate_proj и up_proj.
sharding_col = NamedSharding(mesh, P(None, 'tensor'))

# 3. Горизонтальная маска (Нарезка по строкам)
# P('tensor', None) означает: распилить строки по оси 'tensor' на 8 частей, а столбцы не трогать.
# Применяется для выходных слоев: down_proj.
sharding_row = NamedSharding(mesh, P('tensor', None))

# Инициализируем экстрактор, чтобы он прочитал model.safetensors.index.json
model_dir = "/kaggle/input/qwen/qwen3-14b/transformers/default/1"

# =====================================================================
# ШАГ 2: СТРУКТУРА МОДЕЛИ НА FLAX (ЧИСТАЯ МАТЕМАТИКА БЕЗ ВЕСОВ)
# =====================================================================

class FlaxRMSNorm(nn.Module):
    dim: int          # Скрытая размерность модели (в Qwen3-14B это 5120)
    eps: float = 1e-6
    # Обычный вектор коэффициентов, деление на корень из среднего квадрата
    @nn.compact
    def __call__(self, x):
        # 1. Запрашиваем/создаем вектор весов (гамма-коэффициенты нормализации)
        # Он инициализируется единицами и имеет форму (5120,)
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        # 2. Считаем средний квадрат элементов (mean square) вдоль последней оси (-1)
        # keepdims=True сохраняет размерность для корректного деления [batch, seq, 1]
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        # 3. Считаем инверсный квадратный корень (rsqrt) от (variance + eps)
        # jax.lax.rsqrt — это низкоуровневая операция, которая на TPU v5e выполняется аппаратно за 1 такт
        inv_rms = jax.lax.rsqrt(variance + self.eps)
        # 4. Умножаем входной вектор на инверсный корень и масштабируем весами
        return weight * (x * inv_rms)

class FlaxQwenMLP(nn.Module):
    hidden_size: int         # Исходный размер вектора (5120)
    intermediate_size: int   # Промежуточный размер SwiGLU (13824)
    # Логика SwiGLU: (SiLU(Gate) * Up) -> Down
    @nn.compact
    def __call__(self, x):
        # 1. Расширяем входной вектор в две параллельные ветки
        gate = nn.Dense(features=self.intermediate_size, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(features=self.intermediate_size, use_bias=False, name="up_proj")(x)
        
        # 2. Применяем плавную активацию SiLU к гейту и мягко фильтруем ветку 'up'
        activated = jax.nn.silu(gate) * up
        
        # 3. Сужаем отфильтрованный результат обратно в скрытую размерность модели
        output = nn.Dense(features=self.hidden_size, use_bias=False, name="down_proj")(activated)
        
        return output

class FlaxModelLayer(nn.Module):
    # Объединяем Нормализацию и MLP блок в один слой конвейера
    def __call__(self, x):
        residual = x
        x = FlaxRMSNorm(dim=5120, name="input_layernorm")(x)
        x = FlaxQwenMLP(hidden_size=5120, intermediate_size=13824, name="mlp")(x)
        return residual + x

# =====================================================================
# ШАГ 3: ЛЕНИВАЯ КРАЖА ВЕСОВ С ДИСКА KAGGLE
# =====================================================================

def load_and_shard_weights(model_dir, weight_map, sharding_repl, sharding_col, sharding_row):
    """
    Послойно загружает оригинальные веса Qwen с диска,
    конвертирует их под Flax и распределяет по чипам TPU.
    """
    flax_params = {}
    
    for i in range(40): 
        layer_key = f"layers_{i}"
        flax_params[layer_key] = {"input_layernorm": {}, "mlp": {}}

        # 1. Загрузка RMSNorm вектора (Дублируется целиком на все чипы)
        norm_key = f"model.layers.{i}.input_layernorm.weight"
        norm_file = os.path.join(model_dir, weight_map[norm_key])
        with safe_open(norm_file, framework="np", device="cpu") as f:
            raw_norm = f.get_tensor(norm_key)
        flax_params[layer_key]["input_layernorm"]["weight"] = jax.device_put(jnp.array(raw_norm), sharding_repl)
        
        # 2. Загрузка Gate и Up проекций (Транспонируются и режутся по столбцам)
        gate_key = f"model.layers.{i}.mlp.gate_proj.weight"
        gate_file = os.path.join(model_dir, weight_map[gate_key])
        with safe_open(gate_file, framework="np", device="cpu") as f:
            raw_gate = f.get_tensor(gate_key)
        flax_params[layer_key]["mlp"]["gate_proj"]["kernel"] = jax.device_put(jnp.array(raw_gate.T), sharding_col)
        
        up_key = f"model.layers.{i}.mlp.up_proj.weight"
        up_file = os.path.join(model_dir, weight_map[up_key])
        with safe_open(up_file, framework="np", device="cpu") as f:
            raw_up = f.get_tensor(up_key)
        flax_params[layer_key]["mlp"]["up_proj"]["kernel"] = jax.device_put(jnp.array(raw_up.T), sharding_col)
        
        # 3. Загрузка Down проекции (Транспонируется и режется по строкам)
        down_key = f"model.layers.{i}.mlp.down_proj.weight"
        down_file = os.path.join(model_dir, weight_map[down_key])
        with safe_open(down_file, framework="np", device="cpu") as f:
            raw_down = f.get_tensor(down_key)
        flax_params[layer_key]["mlp"]["down_proj"]["kernel"] = jax.device_put(jnp.array(raw_down.T), sharding_row)
        
        # 4. Принудительная очистка RAM хоста после каждого этажа
        del raw_norm, raw_gate, raw_up, raw_down
        gc.collect()
        
    return {"params": flax_params}


# =====================================================================
# ШАГ 4: СКОМПИЛИРОВАННЫЙ ШАГ ИНФЕРЕНСА (ИНКАПСУЛЯЦИЯ ДЛЯ JIT)
# =====================================================================

model = FlaxModelLayer()

# Компилятор XLA берет математику Flax слоев, сопоставляет её с масками шардинга весов 
# и автоматически дописывает сетевые команды обмена данными (All-Reduce) между 8 чипами TPU.
@jax.jit
def inference_step(weights, x):
    # weights — это наш готовый словарь разрезанных параметров из Шага 3
    # sharded_tokens — дублированный по чипам входной текст
    return model.apply(weights, x)

with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
    weight_map = json.load(f)["weight_map"]
# =====================================================================
# ШАГ 5: ОРКЕСТРАЦИЯ ЗАПУСКА
# =====================================================================

# А. Собираем разрезанную модель на TPU (вызов функции из прошлого шага)
print("[Шаг А] Потоковая нарезка и загрузка весов Qwen3-14B в 8 чипов TPU...")
tpu_params = load_and_shard_weights(
    model_dir=model_dir,
    weight_map=weight_map,
    sharding_repl=sharding_repl,
    sharding_col=sharding_col,
    sharding_row=sharding_row
)

# Б. Готовим входные токены
print("[Шаг Б] Подготовка входного скрытого состояния (Batch=1, Seq=4, Dim=5120)...")
# В реальном коде тут будет выход слоя Embeddings, пока имитируем его единицами
dummy_input = jnp.ones((1, 4, 5120), dtype=jnp.float32)

# Накладываем маску репликации: дублируем входной текст на каждый чип TPU
tpu_tokens = jax.device_put(dummy_input, sharding_repl)

# В. Запуск вычислений в железе
print("[Шаг В] Запуск JIT-компиляции графа и инференса на ядрах TPU...")
# Первый запуск займет ~10-15 секунд на компиляцию, последующие — микросекунды
output_logits = inference_step(tpu_params, tpu_tokens)

# Г. Извлечение финального ответа
print("[Шаг Г] Вычисление вероятностей следующего токена...")
# jnp.argmax находит индекс самого вероятного токена по последней оси словаря
next_token_id = jnp.argmax(output_logits[:, -1, :], axis=-1)

print("\n" + "="*40)
print(" РЕЗУЛЬТАТ ИНФЕРЕНСА НА TPU v5e-8:")
print(f"Форма выходного тензора: {output_logits.shape}")
print(f"Шардинг выхода: {output_logits.sharding}")
print(f"ID предсказанного токена: {next_token_id}")
print("="*40)