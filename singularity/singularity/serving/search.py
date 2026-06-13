from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import jax.numpy as jnp
import numpy as np


@dataclass
class MCTSNode:
    token: int
    visits: int = 0
    value_sum: float = 0.0
    children: list["MCTSNode"] = field(default_factory=list)

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visits, 1)


def choose_best_branch(candidates: Iterable[MCTSNode]) -> MCTSNode | None:
    nodes = list(candidates)
    if not nodes:
        return None
    return max(nodes, key=lambda node: node.value + 0.1 * np.sqrt(np.log(max(sum(node.visits for node in nodes), 2)) / max(node.visits, 1)))


def prune_low_reward(tokens: jnp.ndarray, rewards: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    return tokens[rewards >= threshold]
