from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return data


def merge_configs(*configs: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in configs:
        _deep_update(merged, config)
    return merged


def load_config(*paths: str | Path) -> dict[str, Any]:
    return merge_configs(*(load_yaml(path) for path in paths))
