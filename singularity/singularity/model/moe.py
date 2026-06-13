from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import linen as nn

from .config import MoEConfig
from .ffn import SwiGLU


class ExpertChoiceRouter(nn.Module):
    config: MoEConfig
    hidden_size: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits = nn.Dense(self.config.num_experts, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(hidden_states)
        if self.config.noisy_gating and not deterministic:
            logits = logits + jax.random.normal(self.make_rng("dropout"), logits.shape, dtype=logits.dtype)
        return logits, lax.stop_gradient(jax.nn.softmax(logits, axis=-1))


class SharedRoutedMoE(nn.Module):
    config: MoEConfig
    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        del deterministic
        router = ExpertChoiceRouter(self.config, self.hidden_size, name="router")
        logits, probs = router(x)
        del logits
        experts = [SwiGLU(self.hidden_size, self.intermediate_size, name=f"expert_{idx}") for idx in range(self.config.num_experts)]
        expert_outputs = jnp.stack([expert(x) for expert in experts], axis=0)
        return jnp.sum(probs[..., None] * expert_outputs, axis=0)
