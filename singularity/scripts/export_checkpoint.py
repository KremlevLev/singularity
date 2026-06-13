from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def export_checkpoint(checkpoint_dir: str | Path, repo_id: str, token: str | None = None) -> None:
    api = HfApi(token=token)
    api.upload_folder(folder_path=str(checkpoint_dir), repo_id=repo_id, repo_type="model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload Singularity checkpoint to Hugging Face Hub")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token", default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    export_checkpoint(args.checkpoint_dir, args.repo_id, args.token)
