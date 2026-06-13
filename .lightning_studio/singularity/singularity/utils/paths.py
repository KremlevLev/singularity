from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | os.PathLike[str] | None, root: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(root) if root is not None else project_root()) / candidate
