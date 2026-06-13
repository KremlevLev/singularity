from __future__ import annotations

from typing import Any

from flax import struct
import optax


@struct.dataclass
class TrainState:
    step: int
    params: Any
    tx: optax.GradientTransformation
    opt_state: optax.OptState

    def apply_gradients(self, grads: Any) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state)


def build_train_state(params: Any, learning_rate: float, weight_decay: float, max_grad_norm: float | None = None) -> TrainState:
    chain = []
    if max_grad_norm is not None:
        chain.append(optax.clip_by_global_norm(max_grad_norm))
    chain.append(optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay))
    tx = optax.chain(*chain)
    return TrainState(step=0, params=params, tx=tx, opt_state=tx.init(params))
