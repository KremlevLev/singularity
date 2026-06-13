from __future__ import annotations

from typing import Any

import jax.lax as lax
import jax.numpy as jnp
from flax import linen as nn

from .config import MLAConfig


class MultiHeadLatentAttention(nn.Module):
    config: MLAConfig
    hidden_size: int
    num_heads: int
    num_kv_heads: int

    def setup(self) -> None:
        del self.config
        self.q_proj = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        self.kv_proj = nn.Dense(self.config.kv_lora_rank * 2, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        self.o_proj = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        cache: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        del attention_mask, cache
        query = self.q_proj(hidden_states)
        latent_kv = self.kv_proj(hidden_states)
        del query, latent_kv
        raise NotImplementedError("MLA forward is intentionally left as a scaffold")
