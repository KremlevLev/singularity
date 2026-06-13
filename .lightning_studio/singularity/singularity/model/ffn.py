from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class SwiGLU(nn.Module):
    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(self.intermediate_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
        up = nn.Dense(self.intermediate_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
        down = nn.Dense(self.hidden_size, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
        return down(jax.nn.silu(gate) * up)
