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

# функция для имитации батча, debug
def make_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, dtype: jnp.dtype = jnp.int32) -> ArrayBatch:
    key = jax.random.PRNGKey(42)

    key_input, key_labels = jax.random.split(key, 2)
    input_ids = jax.random.randint(key_input, shape=(batch_size, seq_len), minval=0, maxval=vocab_size, dtype=dtype)
    labels = jax.random.randint(key_labels, shape=(batch_size, seq_len), minval=0, maxval=vocab_size, dtype=dtype)

    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

    assert input_ids.dtype in [jnp.int32, jnp.int64], f"неверный тип input_ids {input_ids.dtype}"
    assert labels.dtype in [jnp.int32, jnp.int64], f"неверный тип labels {labels.dtype}"
    assert attention_mask.dtype == jnp.bool_, f"неверный attention mask dtype {attention_mask.dtype}"

    assert input_ids.shape == (batch_size, seq_len), f"input ids mismatch {input_ids.shape}"
    assert labels.shape == (batch_size, seq_len), f"labels mismatch {labels.shape}"

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

