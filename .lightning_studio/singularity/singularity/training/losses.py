from __future__ import annotations

import jax.numpy as jnp


def language_modeling_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    loss = jax_nn_cross_entropy(logits, labels)
    if mask is not None:
        loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        loss = jnp.mean(loss)
    return loss


def jax_nn_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(labels, logits.shape[-1]), axis=-1) * -1.0


def router_z_loss(router_logits: jnp.ndarray, coefficient: float = 1e-4) -> jnp.ndarray:
    log_z = jax.nn.logsumexp(router_logits, axis=-1)
    return coefficient * jnp.mean(log_z**2)


def load_balancing_loss(router_probs: jnp.ndarray, coefficient: float = 1e-3) -> jnp.ndarray:
    per_token = jnp.mean(router_probs, axis=tuple(range(router_probs.ndim - 1)))
    return coefficient * router_probs.shape[-1] * jnp.sum(per_token**2)
