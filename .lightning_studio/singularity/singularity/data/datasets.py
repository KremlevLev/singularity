from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl


def load_parquet_dataset(path: str | Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def iter_numpy_batches(path: str | Path, batch_size: int, columns: tuple[str, ...]) -> Iterator[dict[str, np.ndarray]]:
    frame = pl.read_parquet(path).select(columns)
    for start in range(0, len(frame), batch_size):
        batch = frame.slice(start, batch_size).to_dict(as_series=False)
        yield {column: np.asarray(values) for column, values in batch.items()}
