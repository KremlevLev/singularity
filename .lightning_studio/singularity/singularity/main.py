from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import jax
import yaml

from .utils.config import load_config
from .utils.logging import setup_logging
from .utils.seed import seed_everything
from .utils.tpu import setup_tpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Singularity 32B JAX/Flax runner")
    parser.add_argument(
        "--phase",
        choices=["sft", "grpo", "serve"],
        default=None,
        help="Training/serving phase to run.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config path. Can be passed multiple times.",
    )
    parser.add_argument("--no-tpu", action="store_true", help="Skip Kaggle TPU bootstrap.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_paths = [Path(path) for path in args.config]
    config = load_config(*config_paths)

    logger = setup_logging(level=config.get("runtime", {}).get("log_level", "INFO"))
    if not args.no_tpu:
        devices = setup_tpu(enable_x64=config.get("runtime", {}).get("enable_x64", False))
    else:
        devices = list(jax.devices())

    logger.info("JAX backend: %s", jax.default_backend())
    logger.info("Visible JAX devices: %s", len(devices))

    seed_everything(int(config.get("project", {}).get("seed", 42)))

    phase = args.phase or config.get("training", {}).get("phase") or config.get("alignment", {}).get("phase")
    if phase == "sft":
        from .training.train_sft import run_sft

        return int(run_sft(config))
    if phase == "grpo":
        from .alignment.grpo import run_grpo

        return int(run_grpo(config))
    if phase == "serve":
        from .serving.api import run_api

        return int(run_api(config))

    raise ValueError(f"Unknown phase: {phase!r}")


__all__ = ["main", "build_parser"]
