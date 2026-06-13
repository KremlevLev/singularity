from __future__ import annotations

import logging

import jax
from jax import config as jax_config

logger = logging.getLogger(__name__)


def setup_tpu(enable_x64: bool = False) -> list[jax.Device]:
    jax_config.update("jax_enable_x64", enable_x64)
    try:
        import jax.tools.colab_tpu

        jax.tools.colab_tpu.setup_tpu()
    except Exception as exc:  # Kaggle/Colab may not need explicit setup.
        logger.debug("Kaggle/Colab TPU bootstrap skipped: %s", exc)
    return list(jax.devices())
