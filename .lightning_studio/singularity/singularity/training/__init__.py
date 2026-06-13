from __future__ import annotations

__all__ = [
    "build_checkpoint_manager",
    "language_modeling_loss",
    "load_balancing_loss",
    "router_z_loss",
    "build_mesh",
    "make_named_sharding",
    "TrainState",
    "build_train_state",
]


def __getattr__(name: str):
    if name in {"build_checkpoint_manager"}:
        from .checkpoint import build_checkpoint_manager

        return build_checkpoint_manager
    if name in {"language_modeling_loss", "load_balancing_loss", "router_z_loss"}:
        from .losses import language_modeling_loss, load_balancing_loss, router_z_loss

        return {
            "language_modeling_loss": language_modeling_loss,
            "load_balancing_loss": load_balancing_loss,
            "router_z_loss": router_z_loss,
        }[name]
    if name in {"build_mesh", "make_named_sharding"}:
        from .sharding import build_mesh, make_named_sharding

        return {"build_mesh": build_mesh, "make_named_sharding": make_named_sharding}[name]
    if name in {"TrainState", "build_train_state"}:
        from .state import TrainState, build_train_state

        return {"TrainState": TrainState, "build_train_state": build_train_state}[name]
    raise AttributeError(name)
