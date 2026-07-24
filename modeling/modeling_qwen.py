#сюда все классы
"""
финальный класс-обертка FlaxQwenForCausalLM. 
Он объединит эмбеддинги, 40 слоев декодера, финальную норму и голову проекции на словарь
"""
# modeling.py (добавление к вашим классам)
import flax.linen as nn
import jax.numpy as jnp
import jax
from modeling.rope import apply_rotary_emb
from config.model_config import QwenConfig

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

class FlaxQwenDecoder(nn.Module):
    config: QwenConfig

    @nn.compact
    def __call__(self, x, position_ids, attention_mask=None):
        # Шаг 6: Проход по слоям (от 0 до 39)
        for i in range(self.config.num_hidden_layers):
            x = FlaxQwenDecoderLayer(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                rope_theta=self.config.rope_theta,
                use_qk_norm=self.config.use_qk_norm, # HAS_QK_NORM
                rms_norm_eps=self.config.rms_norm_eps,
                name=f"layers_{i}",
            )(x, position_ids, attention_mask)
        return x

class FlaxQwenForCausalLM(nn.Module):
    config: QwenConfig

    def setup(self):
        # Шаг 7.1: Входные эмбеддинги (токен ID -> вектор hidden_size)
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Шаг 6: Те самые 40 слоев
        self.model = FlaxQwenDecoder(config=self.config)
        
        # Шаг 7.2: Финальный RMSNorm перед lm_head
        self.norm = FlaxRMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps
        )
        
        # Шаг 7.3: Языковая голова (LM Head)
        # Обратите внимание: bias в Qwen отсутствует
        self.lm_head = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False
        )

    def __call__(self, input_ids, position_ids, attention_mask=None):
        # Принимаем на вход индексы токенов (например, shape: [B, S])
        x = self.embed_tokens(input_ids)
        
        # Прогоняем через блоки декодера
        x = self.model(x, position_ids, attention_mask)
        
        # Применяем финальную норму
        x = self.norm(x)
        
        # Получаем логиты распределения по словарю (shape: [B, S, Vocab_Size])
        logits = self.lm_head(x)
        return logits

