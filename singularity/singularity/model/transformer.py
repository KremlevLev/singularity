from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import linen as nn

from .attention import MultiHeadLatentAttention
from .config import MLAConfig, SingularityConfig
from .embeddings import LlamaEmbeddingAdapter
from .ffn import SwiGLU
from .moe import SharedRoutedMoE
from .rope import DecoupledRoPE


class SingularityTransformer(nn.Module):
    @nn.compact
    def __call__(self, input_ids:jnp.ndarray) -> jnp.ndarray:
        hidden_states = LlamaEmbeddingAdapter( #1. Переводим индексы токенов в скрытые векторы
            self.config.vocab_size,           
            self.config.hidden_size
            ) (input_ids)
        #создаем индексы позиций для каждого токена в батче
        # Форма будет (1, seq_len), чтобы JAX правильно применил её ко всему батчу
        position_ids = jnp.arange(hidden_states.shape[1], dtype = jnp.int32) [None, :]
        
        #применяем RoPE к hidden
        hidden_states= DecoupledRoPE(
            theta = self.config.rope_theta,
            max_seq_len = self.config.max_seq_len
        ) (hidden_states, position_ids)