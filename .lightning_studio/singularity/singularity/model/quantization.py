from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

try:  # Optional research dependency.
    import aqt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    aqt = None  # type: ignore[assignment]

try:  # Optional research dependency.
    import qwix  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    qwix = None  # type: ignore[assignment]


def fake_int4_weight(weight: jnp.ndarray) -> jnp.ndarray:
    scale = jnp.max(jnp.abs(weight), axis=-1, keepdims=True)
    quantized = jnp.round(jnp.clip(weight / jnp.maximum(scale, 1e-6), -7.0, 7.0))
    return (quantized / 7.0) * scale


class FakeQuantDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param("kernel", nn.initializers.lecun_normal(), (x.shape[-1], self.features), jnp.bfloat16)
        quantized_weight = fake_int4_weight(weight)
        bias = self.param("bias", nn.initializers.zeros, (self.features,), jnp.bfloat16)
        return jnp.dot(x, quantized_weight) + bias
