import jax
import flax.linen as nn
from jax.sharding import Mesh
from jax.sharding import NamedSharding, PartitionSpec as P
import jax.numpy as jnp
import numpy as np
import gc
from safetensors import safe_open
import os
import json
from huggingface_hub import snapshot_download

# =====================================================================
# ШАГ 0: АВТОМАТИЧЕСКАЯ ЗАГРУЗКА QWEN3-14B С HUGGING FACE
# =====================================================================
print("[Шаг 0] Скачивание Qwen3-14B с Hugging Face...")
# snapshot_download скачает только веса (.safetensors) и конфиги (.json)
# На Kaggle это займет буквально пару минут благодаря высокой скорости
model_dir = snapshot_download(
    repo_id="Qwen/Qwen3-14B",  # Можно заменить на "Qwen/Qwen3-14B-Instruct" при желании
    allow_patterns=["*.json", "*.safetensors"]
)
print(f"Модель успешно скачана в локальный кэш: {model_dir}\n")

# =====================================================================
# ШАГ 1: НАСТРОЙКА ГЕОМЕТРИИ ЖЕЛЕЗА (SHARDING)
# =====================================================================
devices = np.array(jax.devices())
num_devices = len(devices)

# Решепим устройства в сетку (1, N), где N - количество чипов TPU (например, 8)
devices_mesh = devices.reshape(1, num_devices)
mesh = Mesh(devices_mesh, ('data', 'tensor'))

# Маски распределения весов на чипы
sharding_repl = NamedSharding(mesh, P(None, None))    # Дублировать везде
sharding_col = NamedSharding(mesh, P(None, 'tensor'))  # Разрезать столбцы
sharding_row = NamedSharding(mesh, P('tensor', None))  # Разрезать строки

# =====================================================================
# ШАГ 2: СТРУКТУРА МОДЕЛИ НА FLAX
# =====================================================================

class FlaxRMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    @nn.compact
    def __call__(self, x):
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        inv_rms = jax.lax.rsqrt(variance + self.eps)
        return weight * (x * inv_rms)

class FlaxQwenMLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(features=self.intermediate_size, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(features=self.intermediate_size, use_bias=False, name="up_proj")(x)
        activated = jax.nn.silu(gate) * up
        output = nn.Dense(features=self.hidden_size, use_bias=False, name="down_proj")(activated)
        return output

class FlaxModelLayer(nn.Module):
    def __call__(self, x):
        residual = x
        x = FlaxRMSNorm(dim=5120, name="input_layernorm")(x)
        x = FlaxQwenMLP(hidden_size=5120, intermediate_size=13824, name="mlp")(x)
        return residual + x

# Обертка над всеми 40 слоями модели
class FlaxQwenDecoder(nn.Module):
    num_layers: int = 40
    def __call__(self, x):
        for i in range(self.num_layers):
            x = FlaxModelLayer(name=f"layers_{i}")(x)
        return x

# =====================================================================
# ШАГ 3: ЗАГРУЗКА И НАРЕЗКА ВЕСОВ С ЛОКАЛЬНОГО КЭША HF
# =====================================================================

def load_and_shard_weights(model_dir, weight_map, sharding_repl, sharding_col, sharding_row):
    """
    Послойно загружает оригинальные веса Qwen3-14B, сконвертированные
    в LLaMA-формат, и распределяет их по чипам TPU.
    """
    flax_params = {}
    
    for i in range(40): 
        layer_key = f"layers_{i}"
        flax_params[layer_key] = {"input_layernorm": {}, "mlp": {}}

        # 1. Загрузка RMSNorm
        norm_key = f"model.layers.{i}.input_layernorm.weight"
        norm_file = os.path.join(model_dir, weight_map[norm_key])
        with safe_open(norm_file, framework="np", device="cpu") as f:
            raw_norm = f.get_tensor(norm_key)
        flax_params[layer_key]["input_layernorm"]["weight"] = jax.device_put(raw_norm, sharding_repl)
        
        # 2. Загрузка Gate & Up проекций (столбцы)
        gate_key = f"model.layers.{i}.mlp.gate_proj.weight"
        gate_file = os.path.join(model_dir, weight_map[gate_key])
        with safe_open(gate_file, framework="np", device="cpu") as f:
            raw_gate = f.get_tensor(gate_key)
        flax_params[layer_key]["mlp"]["gate_proj"]["kernel"] = jax.device_put(raw_gate.T, sharding_col)
        
        up_key = f"model.layers.{i}.mlp.up_proj.weight"
        up_file = os.path.join(model_dir, weight_map[up_key])
        with safe_open(up_file, framework="np", device="cpu") as f:
            raw_up = f.get_tensor(up_key)
        flax_params[layer_key]["mlp"]["up_proj"]["kernel"] = jax.device_put(raw_up.T, sharding_col)
        
        # 3. Загрузка Down проекции (строки)
        down_key = f"model.layers.{i}.mlp.down_proj.weight"
        down_file = os.path.join(model_dir, weight_map[down_key])
        with safe_open(down_file, framework="np", device="cpu") as f:
            raw_down = f.get_tensor(down_key)
        flax_params[layer_key]["mlp"]["down_proj"]["kernel"] = jax.device_put(raw_down.T, sharding_row)
        
        # Принудительная очистка RAM хоста
        del raw_norm, raw_gate, raw_up, raw_down
        gc.collect()
        
    return {"params": flax_params}

# =====================================================================
# ШАГ 4: СКОМПИЛИРОВАННЫЙ ШАГ ИНФЕРЕНСА
# =====================================================================

model = FlaxQwenDecoder(num_layers=40)

@jax.jit
def inference_step(weights, x):
    return model.apply(weights, x)

with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
    weight_map = json.load(f)["weight_map"]

# =====================================================================
# ШАГ 5: ОРКЕСТРАЦИЯ ЗАПУСКА
# =====================================================================

# А. Сборка модели на TPU
print(f"[Шаг А] Нарезка и загрузка весов Qwen3-14B в {num_devices} чипов TPU...")
tpu_params = load_and_shard_weights(
    model_dir=model_dir,
    weight_map=weight_map,
    sharding_repl=sharding_repl,
    sharding_col=sharding_col,
    sharding_row=sharding_row
)

# Б. Подготовка входных токенов
print("[Шаг Б] Подготовка входного скрытого состояния (Batch=1, Seq=4, Dim=5120)...")
dummy_input = jnp.ones((1, 4, 5120), dtype=jnp.float32)
tpu_tokens = jax.device_put(dummy_input, sharding_repl)

# В. Запуск вычислений в железе
print("[Шаг В] Запуск JIT-компиляции графа и инференса на ядрах TPU...")
output_hidden_states = inference_step(tpu_params, tpu_tokens)

print("\n" + "="*40)
print(f" РЕЗУЛЬТАТ ИНФЕРЕНСА НА {num_devices} ЧИПАХ TPU:")
print(f"Форма выходного тензора скрытых состояний: {output_hidden_states.shape}")
print(f"Шардинг выхода: {output_hidden_states.sharding}")
print("========================================")