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
config_path = os.path.join(model_dir, "config.json")

with open(config_path, "r", encoding="utf-8") as f:
    hf_config = json.load(f)

HIDDEN_SIZE = hf_config["hidden_size"]
INTERMEDIATE_SIZE = hf_config["intermediate_size"]
NUM_LAYERS = hf_config["num_hidden_layers"]
RMS_NORM_EPS = hf_config.get("rms_norm_eps", 1e-6)

print("Конфигурация модели:")
print(f"  hidden_size       = {HIDDEN_SIZE}")
print(f"  intermediate_size = {INTERMEDIATE_SIZE}")
print(f"  num_hidden_layers = {NUM_LAYERS}")
print(f"  rms_norm_eps      = {RMS_NORM_EPS}")

# =====================================================================
# ШАГ 1: НАСТРОЙКА ГЕОМЕТРИИ ЖЕЛЕЗА (SHARDING)
# =====================================================================
devices = np.array(jax.devices())
num_devices = len(devices)

# Решепим устройства в сетку (1, N), где N - количество чипов TPU (например, 8)
devices_mesh = devices.reshape(1, num_devices)
mesh = Mesh(devices_mesh, ('data', 'tensor'))

# Маски распределения весов на чипы
# 1. Маска репликации для 1D-тензоров (RMSNorm веса)
sharding_repl_1d = NamedSharding(mesh, P(None))

# 2. Маска репликации для 3D-тензоров (входные токены)
sharding_repl_3d = NamedSharding(mesh, P(None, None, None))

# 3. Вертикальная маска для 2D-тензоров (матрицы Gate, Up)
sharding_col = NamedSharding(mesh, P(None, 'tensor'))

# 4. Горизонтальная маска для 2D-тензоров (матрицы Down)
sharding_row = NamedSharding(mesh, P('tensor', None))

# =====================================================================
# ШАГ 2: СТРУКТУРА МОДЕЛИ НА FLAX
# =====================================================================

class FlaxRMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    @nn.compact
    def __call__(self, x):
        weight = self.param(
            "weight",
            nn.initializers.ones,
            (self.dim,),
        )

        input_dtype = x.dtype
        x_float = x.astype(jnp.float32)

        variance = jnp.mean(
            jnp.square(x_float),
            axis=-1,
            keepdims=True,
        )

        normalized = x_float * jax.lax.rsqrt(variance + self.eps)
        normalized = normalized.astype(input_dtype)

        return normalized * weight

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
    hidden_size: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        residual = x

        x = FlaxRMSNorm(
            dim=self.hidden_size,
            eps=self.rms_norm_eps,
            name="input_layernorm",
        )(x)

        x = FlaxQwenMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            name="mlp",
        )(x)

        return residual + x


class FlaxQwenDecoder(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_layers: int
    rms_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = FlaxModelLayer(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                rms_norm_eps=self.rms_norm_eps,
                name=f"layers_{i}",
            )(x)

        return x


# =====================================================================
# ШАГ 3: ЗАГРУЗКА И НАРЕЗКА ВЕСОВ С ЛОКАЛЬНОГО КЭША HF
# =====================================================================

def load_and_shard_weights(
    model_dir,
    weight_map,
    num_layers,
    sharding_repl_1d,
    sharding_col,
    sharding_row,
):
    flax_params = {}

    for i in range(num_layers):
        layer_key = f"layers_{i}"
        flax_params[layer_key] = {
            "input_layernorm": {},
            "mlp": {},
        }
        # ИСПОЛЬЗУЕМ 1D-МАСКУ ДЛЯ RMSNORM ВЕКТОРА
        norm_key = f"model.layers.{i}.input_layernorm.weight"
        norm_file = os.path.join(model_dir, weight_map[norm_key])
        with safe_open(norm_file, framework="np", device="cpu") as f:
            raw_norm = f.get_tensor(norm_key)
        flax_params[layer_key]["input_layernorm"]["weight"] = jax.device_put(raw_norm, sharding_repl_1d)
        
# 2. Загрузка Gate & Up проекций (2D)
        gate_key = f"model.layers.{i}.mlp.gate_proj.weight"
        gate_file = os.path.join(model_dir, weight_map[gate_key])
        with safe_open(gate_file, framework="np", device="cpu") as f:
            raw_gate = f.get_tensor(gate_key)
        # ИСПРАВЛЕНИЕ: присваиваем словарь целиком
        flax_params[layer_key]["mlp"]["gate_proj"] = {"kernel": jax.device_put(raw_gate.T, sharding_col)}
        
        up_key = f"model.layers.{i}.mlp.up_proj.weight"
        up_file = os.path.join(model_dir, weight_map[up_key])
        with safe_open(up_file, framework="np", device="cpu") as f:
            raw_up = f.get_tensor(up_key)
        # ИСПРАВЛЕНИЕ: присваиваем словарь целиком
        flax_params[layer_key]["mlp"]["up_proj"] = {"kernel": jax.device_put(raw_up.T, sharding_col)}
        
        # 3. Загрузка Down проекции (2D)
        down_key = f"model.layers.{i}.mlp.down_proj.weight"
        down_file = os.path.join(model_dir, weight_map[down_key])
        with safe_open(down_file, framework="np", device="cpu") as f:
            raw_down = f.get_tensor(down_key)
        # ИСПРАВЛЕНИЕ: присваиваем словарь целиком
        flax_params[layer_key]["mlp"]["down_proj"] = {"kernel": jax.device_put(raw_down.T, sharding_row)}
        
        del raw_norm, raw_gate, raw_up, raw_down
        gc.collect()      
    return {"params": flax_params}

# =====================================================================
# ШАГ 4: СКОМПИЛИРОВАННЫЙ ШАГ ИНФЕРЕНСА
# =====================================================================

model = FlaxQwenDecoder(
    num_layers=NUM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    rms_norm_eps=RMS_NORM_EPS,
)

@jax.jit
def inference_step(weights, x):
    return model.apply(weights, x)

with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
    weight_map = json.load(f)["weight_map"]

# =====================================================================
# ШАГ 5: ОРКЕСТРАЦИЯ ЗАПУСКА
# =====================================================================

# А. Сборка модели на TPU (передаем sharding_repl_1d)
print(f"[Шаг А] Нарезка и загрузка весов Qwen3-14B в {num_devices} чипов TPU...")
tpu_params = load_and_shard_weights(
    model_dir=model_dir,
    weight_map=weight_map,
    num_layers=NUM_LAYERS,
    sharding_repl_1d=sharding_repl_1d,
    sharding_col=sharding_col,
    sharding_row=sharding_row,
)

# Б. Подготовка входных токенов (передаем sharding_repl_3d для 3D-массива)
print(
    "[Шаг Б] Подготовка входного скрытого состояния "
    f"(Batch=1, Seq=4, Dim={HIDDEN_SIZE})..."
)

dummy_input = jnp.ones(
    (1, 4, HIDDEN_SIZE),
    dtype=jnp.bfloat16,
)

tpu_tokens = jax.device_put(
    dummy_input,
    sharding_repl_3d,
)

print("[Шаг В] JIT-компиляция и инференс...")

output_hidden_states = inference_step(
    tpu_params,
    tpu_tokens,
)

output_hidden_states.block_until_ready()

output_mean = jnp.mean(
    output_hidden_states.astype(jnp.float32)
)
output_mean.block_until_ready()

print("\n" + "=" * 60)
print(f"Результат на {num_devices} устройствах")
print(f"Форма: {output_hidden_states.shape}")
print(f"Dtype: {output_hidden_states.dtype}")
print(f"Шардинг: {output_hidden_states.sharding}")
print(f"Среднее: {float(output_mean)}")
print("=" * 60)
