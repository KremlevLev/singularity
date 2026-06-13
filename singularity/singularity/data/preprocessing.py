from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import polars as pl


def build_dataset_mix(
    parquet_paths: Iterable[str | Path],
    output_path: str | Path,
    fractions: Mapping[str, float] | None = None,
    shuffle: bool = True,
) -> Path:
    scans = [pl.scan_parquet(path) for path in parquet_paths]
    mixed = pl.concat(scans, how="vertical_relaxed") if len(scans) > 1 else scans[0]
    if shuffle:
        mixed = mixed.shuffle()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mixed.sink_parquet(str(output_path))
    return Path(output_path)
