from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
from jax import tree_util


ArrayBatch = Mapping[str, Any]


def make_training_batch(batch: Mapping[str, Any], dtype: jnp.dtype = jnp.bfloat16) -> dict[str, jax.Array]:
    result: dict[str, jax.Array] = {}
    for key, value in batch.items():
        if key.endswith("_mask"):
            result[key] = jnp.asarray(value, dtype=jnp.bool_)
        else:
            result[key] = jnp.asarray(value, dtype=dtype)
    return result


def split_batch(batch: ArrayBatch, devices: int) -> list[ArrayBatch]:
    if devices <= 1:
        return [batch]
    return [tree_util.tree_map(lambda x: x[i::devices], batch) for i in range(devices)]
