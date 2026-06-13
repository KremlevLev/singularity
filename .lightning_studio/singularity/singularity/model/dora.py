from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class DoRALinear(nn.Module):
    features: int
    rank: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        base = nn.Dense(self.features, use_bias=self.use_bias, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        magnitude = self.param("magnitude", nn.initializers.ones, (self.features,), jnp.bfloat16)
        low_rank_a = self.param("lora_a", nn.initializers.normal(stddev=0.02), (x.shape[-1], self.rank), jnp.bfloat16)
        low_rank_b = self.param("lora_b", nn.initializers.zeros, (self.rank, self.features), jnp.bfloat16)
        adapted = base(x) + magnitude * (jnp.einsum("...i,ir,rf->...f", x, low_rank_a, low_rank_b))
        return adapted
