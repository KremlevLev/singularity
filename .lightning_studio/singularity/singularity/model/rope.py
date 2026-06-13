from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


def apply_rope(x: jnp.ndarray, position_ids: jnp.ndarray, rotary_dim: int | None = None) -> jnp.ndarray:
    del position_ids
    rotary_dim = rotary_dim or x.shape[-1]
    if rotary_dim != x.shape[-1]:
        raise NotImplementedError("Partial rotary dimension slicing is not implemented yet")
    return x


class DecoupledRoPE(nn.Module):
    theta: float = 500000.0
    max_seq_len: int = 16384

    def __call__(self, x: jnp.ndarray, position_ids: jnp.ndarray) -> jnp.ndarray:
        return apply_rope(x, position_ids=position_ids, rotary_dim=x.shape[-1])
