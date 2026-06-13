from __future__ import annotations

import os
import random

import jax
import numpy as np
from jax import random as jrandom


def seed_everything(seed: int) -> jrandom.KeyArray:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    key = jrandom.PRNGKey(seed)
    return key
