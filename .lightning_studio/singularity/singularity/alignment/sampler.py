from __future__ import annotations

from typing import Any

import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import linen as nn


def sample_tokens(
    params: Any,
    model: nn.Module,
    input_ids: jnp.ndarray,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    del model
    if key is None:
        key = jax.random.PRNGKey(0)
    generated = input_ids
    for _ in range(max_new_tokens):
        logits = jnp.ones((generated.shape[0], 1, 1), dtype=jnp.bfloat16)
        next_token = jax.random.categorical(key, logits[:, -1, :] / max(temperature, 1e-6), axis=-1)
        generated = lax.dynamic_concatenate([generated, next_token[:, None]], axis=1)
    return generated
