from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from ..model import SingularityConfig, SingularityTransformer
from ..utils.logging import setup_logging
from .checkpoint import build_checkpoint_manager, save_checkpoint
from .losses import language_modeling_loss, load_balancing_loss, router_z_loss
from .sharding import build_mesh
from .state import TrainState, build_train_state


def make_optimizer(config: Mapping[str, Any]) -> optax.GradientTransformation:
    optimizer = config.get("optimizer", {})
    return optax.chain(
        optax.clip_by_global_norm(float(optimizer.get("max_grad_norm", 1.0))),
        optax.adamw(
            learning_rate=float(optimizer.get("lr", 1.5e-4)),
            weight_decay=float(optimizer.get("weight_decay", 0.1)),
        ),
    )


def train_step(
    params: Any,
    opt_state: optax.OptState,
    tx: optax.GradientTransformation,
    batch: Mapping[str, Any],
    key: jax.Array,
) -> tuple[Any, optax.OptState, dict[str, jnp.ndarray]]:
    del key

    def loss_fn(current_params: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        logits = SingularityTransformer(SingularityConfig.from_dict({})).apply(current_params, batch["input_ids"])
        ce_loss = language_modeling_loss(logits, batch["labels"], batch.get("attention_mask"))
        aux_loss = jnp.asarray(0.0, dtype=logits.dtype)
        return ce_loss + aux_loss, {"loss": ce_loss, "aux_loss": aux_loss}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    metrics["loss"] = loss
    return new_params, new_opt_state, metrics


def run_sft(config: Mapping[str, Any]) -> int:
    logger = setup_logging(str(config.get("runtime", {}).get("log_level", "INFO")))
    mesh = build_mesh(
        mesh_shape=config.get("sharding", {}).get("mesh_shape"),
        mesh_axes=config.get("sharding", {}).get("mesh_axes"),
    )
    logger.info("Created JAX mesh with %s devices", mesh.devices.size)

    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "artifacts/checkpoints")
    manager = build_checkpoint_manager(checkpoint_dir, keep_last=int(config.get("training", {}).get("checkpoint", {}).get("keep_last", 3)))

    for step in tqdm(range(int(config.get("training", {}).get("max_steps", 0)))):
        del step
        raise NotImplementedError("SFT training loop body is scaffolded; add data loader and JIT train step here")

    save_checkpoint(manager, 0, {"config": config})
    return 0
