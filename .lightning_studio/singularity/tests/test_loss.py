from __future__ import annotations

import jax.numpy as jnp

from singularity.training.losses import language_modeling_loss, router_z_loss


def test_language_modeling_loss_shape() -> None:
    logits = jnp.zeros((2, 4, 8), dtype=jnp.float32)
    labels = jnp.ones((2, 4), dtype=jnp.int32)
    loss = language_modeling_loss(logits, labels)
    assert loss.shape == ()


def test_router_z_loss_shape() -> None:
    logits = jnp.zeros((2, 4, 8), dtype=jnp.float32)
    loss = router_z_loss(logits)
    assert loss.shape == ()
