from __future__ import annotations

import jax
import jax.numpy as jnp

from singularity.model.config import MoEConfig
from singularity.model.moe import ExpertChoiceRouter


def test_router_logits_shape() -> None:
    router = ExpertChoiceRouter(MoEConfig(num_experts=4), hidden_size=8)
    params = router.init(jax.random.PRNGKey(0), jnp.ones((2, 3, 8)))
    logits, probs = router.apply(params, jnp.ones((2, 3, 8)))
    assert logits.shape == (2, 3, 4)
    assert probs.shape == logits.shape
