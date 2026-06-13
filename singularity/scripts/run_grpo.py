from __future__ import annotations

import argparse

from singularity.main import main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Singularity GRPO alignment")
    parser.add_argument("--config", action="append", default=[], help="YAML config path; can be passed multiple times")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    cli_args = ["--phase", "grpo"]
    for config_path in args.config:
        cli_args.extend(["--config", config_path])
    raise SystemExit(main(cli_args))
