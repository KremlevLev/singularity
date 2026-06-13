from __future__ import annotations

from pathlib import Path
from typing import Any

from orbax import checkpoint as ocp


def build_checkpoint_manager(directory: str | Path, keep_last: int = 3) -> ocp.CheckpointManager:
    target_directory = Path(directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    return ocp.CheckpointManager(
        directory=str(target_directory),
        checkpointers={"state": ocp.PyTreeCheckpointer()},
        options=ocp.CheckpointManagerOptions(max_to_keep=keep_last),
    )


def save_checkpoint(manager: ocp.CheckpointManager, step: int, state: Any) -> None:
    manager.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(state)))
