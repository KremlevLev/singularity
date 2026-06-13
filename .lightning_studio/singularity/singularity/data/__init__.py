from __future__ import annotations

__all__ = ["make_training_batch", "load_parquet_dataset", "build_dataset_mix", "TokenizerAdapter"]


def __getattr__(name: str):
    if name in {"make_training_batch"}:
        from .batch import make_training_batch

        return make_training_batch
    if name == "load_parquet_dataset":
        from .datasets import load_parquet_dataset

        return load_parquet_dataset
    if name == "build_dataset_mix":
        from .preprocessing import build_dataset_mix

        return build_dataset_mix
    if name == "TokenizerAdapter":
        from .tokenization import TokenizerAdapter

        return TokenizerAdapter
    raise AttributeError(name)
