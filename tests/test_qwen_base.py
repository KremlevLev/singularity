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

# =====================================================================
# ШАГ 1: НАСТРОЙКА ГЕОМЕТРИИ ЖЕЛЕЗА (SHARDING)
# =====================================================================

devices = np.array(jax.devices())
num_devices = len(devices)

# Автоматически решепим под доступное количество TPU-ядер (например, 1х8)
devices_mesh = devices.reshape(1, num_devices)

# Объединяем чипы в логическую сеть
mesh = Mesh(devices_mesh, ('data', 'tensor'))

# 1. Маска репликации
sharding_repl = NamedSharding(mesh, P(None, None))

# 2. Вертикальная маска (по столбцам)
sharding_col = NamedSharding(mesh, P(None, 'tensor'))

# 3. Горизонтальная маска (по строкам)
sharding_row = NamedSharding(mesh, P('tensor', None))

model_dir = "/kaggle/input/qwen/qwen3-14b/transformers/default/1"

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

# Добавляем верхнеуровневый декодер для связи всех 40 слоев
class FlaxQwenDecoder(nn.Module):
    num_layers: int = 40
    
    def __call__(self, x):
        for i in range(self.num_layers):
            x = FlaxModelLayer(name=f"layers_{i}")(x)
        return x

# =====================================================================
# ШАГ 3: ЗАГРУЗКА ВЕСОВ С ДИСКА
# =====================================================================

def load_and_shard_weights(model_dir, weight_map, sharding_repl, sharding_col, sharding_row):
    flax_params = {}
    
    for i in range(40): 
        layer_key = f"layers_{i}"
        flax_params[layer_key] = {"input_layernorm": {}, "mlp": {}}

        # 1. RMSNorm
        norm_key = f"model.layers.{i}.input_layernorm.weight"
        norm_file = os.path.join(model_dir, weight_map[norm_key])
        with safe_open(norm_file, framework="np", device="cpu") as f:
            raw_norm = f.get_tensor(norm_key)
        flax_params[layer_key]["input_layernorm"]["weight"] = jax.device_put(raw_norm, sharding_repl)
        
        # 2. Gate & Up (транспонируем из PyTorch (out, in) -> (in, out))
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
        
        # 3. Down (транспонируем)
        down_key = f"model.layers.{i}.mlp.down_proj.weight"
        down_file = os.path.join(model_dir, weight_map[down_key])
        with safe_open(down_file, framework="np", device="cpu") as f:
            raw_down = f.get_tensor(down_key)
        flax_params[layer_key]["mlp"]["down_proj"]["kernel"] = jax.device_put(raw_down.T, sharding_row)
        
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

print(f"[Шаг А] Потоковая нарезка и загрузка весов Qwen в {num_devices} чипов TPU...")
tpu_params = load_and_shard_weights(
    model_dir=model_dir,
    weight_map=weight_map,
    sharding_repl=sharding_repl,
    sharding_col=sharding_col,
    sharding_row=sharding_row
)

print("[Шаг Б] Подготовка входного скрытого состояния (Batch=1, Seq=4, Dim=5120)...")
dummy_input = jnp.ones((1, 4, 5120), dtype=jnp.float32)
tpu_tokens = jax.device_put(dummy_input, sharding_repl)

print("[Шаг В] Запуск JIT-компиляции графа и инференса на ядрах TPU...")
output_hidden_states = inference_step(tpu_params, tpu_tokens)

print("\n" + "="*40)
print(" РЕЗУЛЬТАТ ИНФЕРЕНСА НА TPU:")
print(f"Форма выходного тензора: {output_hidden_states.shape}")
print(f"Шардинг выхода: {output_hidden_states.sharding}")
print("Примечание: Получены скрытые состояния. Для предсказания токенов требуется lm_head.")
print("="*40)