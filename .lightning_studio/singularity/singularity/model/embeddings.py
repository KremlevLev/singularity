from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class LlamaEmbeddingAdapter(nn.Module):
    vocab_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        embedder = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )
        return embedder(input_ids)
