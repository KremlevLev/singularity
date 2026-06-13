from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
from orbax import checkpoint as ocp


def merge_dora_linear(base_weight: jnp.ndarray, lora_a: jnp.ndarray, lora_b: jnp.ndarray, magnitude: jnp.ndarray) -> jnp.ndarray:
    adapted = magnitude * jnp.einsum("ir,rf->if", lora_a, lora_b)
    return base_weight + adapted


def load_checkpoint_state(checkpoint_dir: str | Path, step: int | None = None) -> dict[str, Any]:
    manager = ocp.CheckpointManager(str(checkpoint_dir))
    step = step or manager.latest_step()
    return manager.restore(step, args=ocp.args.Composite(state=ocp.args.StandardRestore(None)))  # type: ignore[arg-type]
