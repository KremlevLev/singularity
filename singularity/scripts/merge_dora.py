from __future__ import annotations

import argparse
from pathlib import Path

from singularity.serving.merge import load_checkpoint_state, merge_dora_linear


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge DoRA adapters into base weights")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    state = load_checkpoint_state(Path(args.checkpoint_dir))
    del state, merge_dora_linear
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
