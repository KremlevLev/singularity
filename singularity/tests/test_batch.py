import jax.numpy as jnp

from singularity.data.batch import make_synthetic_batch, make_training_batch


def test_make_synthetic_batch():
    batch_size = 4
    seq_len = 128
    vocab_size = 32000

    batch = make_synthetic_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        dtype=jnp.int32,
    )

    assert set(batch) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].shape == (batch_size, seq_len)
    assert batch["labels"].shape == (batch_size, seq_len)
    assert batch["attention_mask"].shape == (batch_size, seq_len)

    assert batch["input_ids"].dtype == jnp.int32
    assert batch["labels"].dtype == jnp.int32
    assert batch["attention_mask"].dtype == jnp.bool_

    assert jnp.all(batch["input_ids"] >= 0)
    assert jnp.all(batch["input_ids"] < vocab_size)
    assert jnp.all(batch["labels"] >= 0)
    assert jnp.all(batch["labels"] < vocab_size)
    assert jnp.all(batch["attention_mask"])


def test_make_training_batch_converts_masks_to_bool():
    raw_batch = make_synthetic_batch(batch_size=2, seq_len=8, vocab_size=100, dtype=jnp.int32)
    training_batch = make_training_batch(raw_batch, dtype=jnp.bfloat16)

    assert training_batch["input_ids"].dtype == jnp.bfloat16
    assert training_batch["labels"].dtype == jnp.bfloat16
    assert training_batch["attention_mask"].dtype == jnp.bool_