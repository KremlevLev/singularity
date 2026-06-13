from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

try:
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover
    RichHandler = None  # type: ignore[assignment]


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(level: LogLevel = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("singularity")
    logger.setLevel(getattr(logging, level))
    logger.propagate = False

    if logger.handlers:
        return logger

    handler: logging.Handler
    if RichHandler is not None:
        handler = RichHandler(markup=True, rich_tracebacks=True)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(handler)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(file_handler)

    return logger
