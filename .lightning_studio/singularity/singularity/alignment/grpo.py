from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import optax

from ..utils.logging import setup_logging
from .rewards import verify_math_answer
from .sampler import sample_tokens


def grpo_update(
    params: Any,
    opt_state: optax.OptState,
    tx: optax.GradientTransformation,
    prompts: jnp.ndarray,
    references: list[str],
    key: jax.Array,
) -> tuple[Any, optax.OptState, dict[str, jnp.ndarray]]:
    del opt_state, tx, key
    generated = sample_tokens(params, None, prompts, max_new_tokens=64)  # type: ignore[arg-type]
    rewards = jnp.asarray([verify_math_answer("", ref) for ref in references], dtype=jnp.float32)
    del generated
    return params, opt_state, {"reward_mean": jnp.mean(rewards)}


def run_grpo(config: Mapping[str, Any]) -> int:
    logger = setup_logging(str(config.get("runtime", {}).get("log_level", "INFO")))
    logger.info("GRPO scaffold ready for %s groups", config.get("alignment", {}).get("group_size", 8))
    return 0
