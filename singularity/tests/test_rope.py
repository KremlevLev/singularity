from __future__ import annotations

import jax.numpy as jnp

from singularity.model.rope import apply_rope


def test_apply_rope_identity_scaffold() -> None:
    x = jnp.ones((2, 4, 8), dtype=jnp.bfloat16)
    y = apply_rope(x, position_ids=jnp.arange(4)[None, :])
    assert y.shape == x.shape
