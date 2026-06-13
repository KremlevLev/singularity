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
    config: SingularityConfig

    def setup(self) -> None:
        self.embedder = LlamaEmbeddingAdapter(self.config.vocab_size, self.config.hidden_size)
        self.rope = DecoupledRoPE(theta=self.config.rope_theta, max_seq_len=self.config.max_seq_len)
        self.mla = MultiHeadLatentAttention(
            config=MLAConfig(),
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
        )
        self.moe = SharedRoutedMoE(
            config=self.config.moe,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
        )
        self.output_norm = nn.RMSNorm(dtype=jnp.bfloat16)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        del attention_mask
        hidden_states = self.embedder(input_ids)
        position_ids = jnp.arange(hidden_states.shape[1], dtype=jnp.int32)[None, :]
        hidden_states = self.rope(hidden_states, position_ids)
        for _ in range(self.config.num_layers):
            residual = hidden_states
            attn_out = self.mla(hidden_states)
            hidden_states = residual + attn_out
            residual = hidden_states
            if self.config.moe is not None:
                hidden_states = self.moe(hidden_states)
            else:
                hidden_states = SwiGLU(self.config.hidden_size, self.config.intermediate_size)(hidden_states)
            hidden_states = residual + hidden_states
        hidden_states = self.output_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
