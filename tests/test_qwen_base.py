import jax
import flax.linen as nn
from jax.sharding import Mesh
from jax.sharding import NamedSharding, PartitionSpec as P
import jax.numpy as jnp
import numpy as np
import gc
from safetensors import safe_open
import os
import torch
import torch.nn.functional as F
import json
from huggingface_hub import snapshot_download
from scripts.tg_notifier import send_telegram_notification

notify=1
if notify:
    send_telegram_notification("success")

TEST_NUM_LAYERS = 1

# =====================================================================
# ШАГ 0: АВТОМАТИЧЕСКАЯ ЗАГРУЗКА QWEN3-14B С HUGGING FACE
# =====================================================================
print("[Шаг 0] Скачивание Qwen3-14B с Hugging Face...")
model_dir = snapshot_download(
    repo_id="Qwen/Qwen3-14B",
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

NUM_ATTENTION_HEADS = hf_config["num_attention_heads"]
NUM_KEY_VALUE_HEADS = hf_config["num_key_value_heads"]
HEAD_DIM = hf_config.get("head_dim", HIDDEN_SIZE // NUM_ATTENTION_HEADS)
ROPE_THETA = hf_config.get("rope_theta", 1_000_000.0)

print("Конфигурация модели:")
print(f"  hidden_size       = {HIDDEN_SIZE}")
print(f"  intermediate_size = {INTERMEDIATE_SIZE}")
print(f"  num_hidden_layers = {NUM_LAYERS} (для теста используем {TEST_NUM_LAYERS})")
print(f"  rms_norm_eps      = {RMS_NORM_EPS}")
print("Attention config:")
print(f"  query heads = {NUM_ATTENTION_HEADS}")
print(f"  KV heads    = {NUM_KEY_VALUE_HEADS}")
print(f"  head dim    = {HEAD_DIM}")
print(f"  RoPE theta  = {ROPE_THETA}")

# Загружаем weight_map заранее, чтобы узнать про QK-norm
with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
    weight_map = json.load(f)["weight_map"]

q_norm_key_test = "model.layers.0.self_attn.q_norm.weight"
k_norm_key_test = "model.layers.0.self_attn.k_norm.weight"
HAS_QK_NORM = q_norm_key_test in weight_map and k_norm_key_test in weight_map

print(f"q_norm / k_norm present: {HAS_QK_NORM}")

# =====================================================================
# ШАГ 1: НАСТРОЙКА ГЕОМЕТРИИ ЖЕЛЕЗА (SHARDING)
# =====================================================================
devices = np.array(jax.devices())
num_devices = len(devices)

devices_mesh = devices.reshape(1, num_devices)
mesh = Mesh(devices_mesh, ('data', 'tensor'))

sharding_repl_1d = NamedSharding(mesh, P(None))
sharding_repl_3d = NamedSharding(mesh, P(None, None, None))
sharding_col = NamedSharding(mesh, P(None, 'tensor'))
sharding_row = NamedSharding(mesh, P('tensor', None))

# =====================================================================
# ШАГ 2: СТРУКТУРА МОДЕЛИ НА FLAX (В ПРАВИЛЬНОМ ПОРЯДКЕ)
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
        variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
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

# --- УТИЛИТЫ ДЛЯ ROPE ---
def precompute_freqs_cis(dim, max_position, theta):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(max_position)
    freqs = jnp.outer(t, freqs)
    return jnp.cos(freqs), jnp.sin(freqs)

def apply_rotary_emb(q, k, position_ids, theta, head_dim):
    cos, sin = precompute_freqs_cis(head_dim, 131072, theta)
    cos = cos[position_ids][:, :, None, :]
    sin = sin[position_ids][:, :, None, :]
    q1, q2 = q[..., :head_dim//2], q[..., head_dim//2:]
    k1, k2 = k[..., :head_dim//2], k[..., head_dim//2:]
    q_rotated = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rotated = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rotated, k_rotated

class FlaxQwenAttention(nn.Module):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    use_qk_norm: bool
    rms_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x, position_ids, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = nn.Dense(self.num_attention_heads * self.head_dim, use_bias=False, name="q_proj")(x)
        k = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, name="k_proj")(x)
        v = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False, name="v_proj")(x)
        
        q = q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # QK-нормализация (только если веса присутствуют в конфиге)
        if self.use_qk_norm:
            q = FlaxRMSNorm(
                dim=self.head_dim,
                eps=self.rms_norm_eps,
                name="q_norm",
            )(q)

            k = FlaxRMSNorm(
                dim=self.head_dim,
                eps=self.rms_norm_eps,
                name="k_norm",
            )(k)

        
        q, k = apply_rotary_emb(q, k, position_ids, self.rope_theta, self.head_dim)
        
        num_groups = self.num_attention_heads // self.num_key_value_heads
        k = jnp.repeat(k, num_groups, axis=2)
        v = jnp.repeat(v, num_groups, axis=2)
        
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = causal_mask.reshape(1, 1, seq_len, seq_len)
        attn_weights = jnp.where(causal_mask == 0, -1e9, attn_weights)
        
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32)).astype(q.dtype)
        attn_output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        output = nn.Dense(self.hidden_size, use_bias=False, name="o_proj")(attn_output)
        return output

class FlaxQwenDecoderLayer(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    use_qk_norm: bool
    rms_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x, position_ids, attention_mask=None):
        residual = x
        x = FlaxRMSNorm(dim=self.hidden_size, eps=self.rms_norm_eps, name="input_layernorm")(x)
        x = FlaxQwenAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            use_qk_norm=self.use_qk_norm,
            rms_norm_eps=self.rms_norm_eps,
            name="self_attn",
        )(x, position_ids, attention_mask)
        x = residual + x
        
        residual = x
        x = FlaxRMSNorm(dim=self.hidden_size, eps=self.rms_norm_eps, name="post_attention_layernorm")(x)
        x = FlaxQwenMLP(hidden_size=self.hidden_size, intermediate_size=self.intermediate_size, name="mlp")(x)
        x = residual + x
        
        return x

class FlaxQwenDecoder(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    use_qk_norm: bool
    rms_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x, position_ids, attention_mask=None):
        for i in range(self.num_layers):
            x = FlaxQwenDecoderLayer(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                rope_theta=self.rope_theta,
                use_qk_norm=self.use_qk_norm,
                rms_norm_eps=self.rms_norm_eps,
                name=f"layers_{i}",
            )(x, position_ids, attention_mask)
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
    has_qk_norm,
):
    flax_params = {}

    for i in range(num_layers):
        layer_key = f"layers_{i}"
        
        # ЕДИНСТВЕННАЯ инициализация структуры словаря для предотвращения перезаписи
        flax_params[layer_key] = {
            "input_layernorm": {},
            "post_attention_layernorm": {},
            "self_attn": {
                "q_proj": {},
                "k_proj": {},
                "v_proj": {},
                "o_proj": {},
            },
            "mlp": {},
        }
        
        if has_qk_norm:
            flax_params[layer_key]["self_attn"]["q_norm"] = {}
            flax_params[layer_key]["self_attn"]["k_norm"] = {}

        # 1. Нормализации
        norm_key = f"model.layers.{i}.post_attention_layernorm.weight"
        norm_file = os.path.join(model_dir, weight_map[norm_key])
        with safe_open(norm_file, framework="np", device="cpu") as f:
            raw_norm = f.get_tensor(norm_key)
        flax_params[layer_key]["post_attention_layernorm"]["weight"] = jax.device_put(raw_norm, sharding_repl_1d)

        norm_key = f"model.layers.{i}.input_layernorm.weight"
        norm_file = os.path.join(model_dir, weight_map[norm_key])
        with safe_open(norm_file, framework="np", device="cpu") as f:
            raw_norm = f.get_tensor(norm_key)
        flax_params[layer_key]["input_layernorm"]["weight"] = jax.device_put(raw_norm, sharding_repl_1d)

        # 2. Attention: Q, K, V, O
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            hf_key = f"model.layers.{i}.self_attn.{proj_name}.weight"
            hf_file = os.path.join(model_dir, weight_map[hf_key])
            with safe_open(hf_file, framework="np", device="cpu") as f:
                raw_weight = f.get_tensor(hf_key)

            if proj_name == "o_proj":
                flax_params[layer_key]["self_attn"][proj_name] = {"kernel": jax.device_put(raw_weight.T, sharding_row)}
            else:
                flax_params[layer_key]["self_attn"][proj_name] = {"kernel": jax.device_put(raw_weight.T, sharding_col)}

        # 3. Attention: QK Нормализация (опционально)
        if has_qk_norm:
            q_norm_key = f"model.layers.{i}.self_attn.q_norm.weight"
            q_norm_file = os.path.join(model_dir, weight_map[q_norm_key])
            with safe_open(q_norm_file, framework="np", device="cpu") as f:
                raw_q_norm = f.get_tensor(q_norm_key)
            flax_params[layer_key]["self_attn"]["q_norm"]["weight"] = jax.device_put(raw_q_norm, sharding_repl_1d)

            k_norm_key = f"model.layers.{i}.self_attn.k_norm.weight"
            k_norm_file = os.path.join(model_dir, weight_map[k_norm_key])
            with safe_open(k_norm_file, framework="np", device="cpu") as f:
                raw_k_norm = f.get_tensor(k_norm_key)
            flax_params[layer_key]["self_attn"]["k_norm"]["weight"] = jax.device_put(raw_k_norm, sharding_repl_1d)

        # 4. MLP проекции
        gate_key = f"model.layers.{i}.mlp.gate_proj.weight"
        gate_file = os.path.join(model_dir, weight_map[gate_key])
        with safe_open(gate_file, framework="np", device="cpu") as f:
            raw_gate = f.get_tensor(gate_key)
        flax_params[layer_key]["mlp"]["gate_proj"] = {"kernel": jax.device_put(raw_gate.T, sharding_col)}
        
        up_key = f"model.layers.{i}.mlp.up_proj.weight"
        up_file = os.path.join(model_dir, weight_map[up_key])
        with safe_open(up_file, framework="np", device="cpu") as f:
            raw_up = f.get_tensor(up_key)
        flax_params[layer_key]["mlp"]["up_proj"] = {"kernel": jax.device_put(raw_up.T, sharding_col)}
        
        down_key = f"model.layers.{i}.mlp.down_proj.weight"
        down_file = os.path.join(model_dir, weight_map[down_key])
        with safe_open(down_file, framework="np", device="cpu") as f:
            raw_down = f.get_tensor(down_key)
        flax_params[layer_key]["mlp"]["down_proj"] = {"kernel": jax.device_put(raw_down.T, sharding_row)}
        
        gc.collect()      

    return {"params": flax_params}

# =====================================================================
# ШАГ 4: СКОМПИЛИРОВАННЫЙ ШАГ ИНФЕРЕНСА
# =====================================================================

model = FlaxQwenDecoder(
    num_layers=TEST_NUM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    rms_norm_eps=RMS_NORM_EPS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    num_key_value_heads=NUM_KEY_VALUE_HEADS,
    head_dim=HEAD_DIM,
    rope_theta=ROPE_THETA,
    use_qk_norm=HAS_QK_NORM
)

@jax.jit
def inference_step(weights, x, position_ids, attention_mask=None):
    return model.apply(weights, x, position_ids, attention_mask)

# =====================================================================
# ШАГ 5: ОРКЕСТРАЦИЯ ЗАПУСКА
# =====================================================================

print(f"[Шаг А] Нарезка и загрузка весов Qwen3-14B в {num_devices} чипов TPU...")
tpu_params = load_and_shard_weights(
    model_dir=model_dir,
    weight_map=weight_map,
    num_layers=TEST_NUM_LAYERS,
    sharding_repl_1d=sharding_repl_1d,
    sharding_col=sharding_col,
    sharding_row=sharding_row,
    has_qk_norm=HAS_QK_NORM,
)

print(f"[Шаг Б] Подготовка входного скрытого состояния (Batch=1, Seq=4, Dim={HIDDEN_SIZE})...")
dummy_input = jnp.ones(
    (1, 4, HIDDEN_SIZE),
    dtype=jnp.bfloat16,
)
tpu_tokens = jax.device_put(dummy_input, sharding_repl_3d)

# Создание position_ids ДО вызова inference_step
seq_len = 4
position_ids = jnp.arange(seq_len)[None, :]  # (1, seq_len)

print("[Шаг В] JIT-компиляция и инференс...")
output_hidden_states = inference_step(
    tpu_params,
    tpu_tokens,
    position_ids,
)

output_hidden_states.block_until_ready()

output_mean = jnp.mean(output_hidden_states.astype(jnp.float32))
output_mean.block_until_ready()

print("\n" + "=" * 60)
print(f"Результат на {num_devices} устройствах")
print(f"Форма: {output_hidden_states.shape}")
print(f"Dtype: {output_hidden_states.dtype}")
print(f"Шардинг: {output_hidden_states.sharding}")
print(f"Среднее: {float(output_mean)}")
print("=" * 60)


# =====================================================================
# ТЕСТОВЫЕ ЧАСТИ (Сравнение с PyTorch)
# =====================================================================

def torch_rms_norm(x, weight, eps):
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    normalized = x_fp32 * torch.rsqrt(variance + eps)
    return normalized.to(x.dtype) * weight.to(x.dtype)

def torch_mlp_reference(
    x,
    norm_weight,
    gate_weight,
    up_weight,
    down_weight,
    eps,
):
    residual = x
    x = torch_rms_norm(x, norm_weight, eps)
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    x = F.silu(gate) * up
    x = F.linear(x, down_weight)
    return residual + x

# Примечание: PyTorch тест ниже проверяет ТОЛЬКО MLP блок (без attention),
# в то время как JAX отрабатывает полный слой (Attention + MLP).
# Для получения нулевой разницы в тестах нужно реализовать PyTorch Attention с RoPE.
x_torch = torch.ones((1, 4, HIDDEN_SIZE), dtype=torch.bfloat16)

print(f"[Тест] Запуск референсного расчета в PyTorch для {TEST_NUM_LAYERS} слоев (только MLP часть)...")
for i in range(TEST_NUM_LAYERS):
    layer_key = f"layers_{i}"
    
    norm_weight_jax = tpu_params["params"][layer_key]["post_attention_layernorm"]["weight"]
    gate_kernel_jax = tpu_params["params"][layer_key]["mlp"]["gate_proj"]["kernel"]
    up_kernel_jax = tpu_params["params"][layer_key]["mlp"]["up_proj"]["kernel"]
    down_kernel_jax = tpu_params["params"][layer_key]["mlp"]["down_proj"]["kernel"]
    
    norm_weight = torch.from_numpy(np.array(norm_weight_jax.astype(jnp.float32))).to(torch.bfloat16)
    gate_weight = torch.from_numpy(np.array(gate_kernel_jax.T.astype(jnp.float32))).to(torch.bfloat16)
    up_weight = torch.from_numpy(np.array(up_kernel_jax.T.astype(jnp.float32))).to(torch.bfloat16)
    down_weight = torch.from_numpy(np.array(down_kernel_jax.T.astype(jnp.float32))).to(torch.bfloat16)
    
    x_torch = torch_mlp_reference(
        x_torch,
        norm_weight,
        gate_weight,
        up_weight,
        down_weight,
        eps=RMS_NORM_EPS,
    )

reference_output = x_torch

jax_result = np.asarray(jax.device_get(output_hidden_states))
torch_result = reference_output.float().cpu().numpy()
abs_diff = np.abs(jax_result - torch_result)

print("\n" + "=" * 60)
print("СРАВНЕНИЕ ТОЧНОСТИ (Flax Attention+MLP vs PyTorch MLP):")
print("max abs diff:", abs_diff.max())
print("mean abs diff:", abs_diff.mean())
print("=" * 60)