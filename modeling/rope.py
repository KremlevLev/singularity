import jax.numpy as jnp


def apply_rotary_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    position_ids: jnp.ndarray,
    theta: float,
    head_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    q: (batch, seq, num_q_heads, head_dim)
    k: (batch, seq, num_kv_heads, head_dim)
    position_ids: (batch, seq)
    """

    assert head_dim % 2 == 0, "RoPE требует чётный head_dim"

    inv_freq = 1.0 / (
        theta ** (
            jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
        )
    )

    # (batch, seq, head_dim / 2)
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq

    # (batch, seq, 1, head_dim / 2)
    cos = jnp.cos(freqs)[:, :, None, :]
    sin = jnp.sin(freqs)[:, :, None, :]

    q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
    k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]

    q_rot = jnp.concatenate(
        [q1 * cos - q2 * sin, q1 * sin + q2 * cos],
        axis=-1,
    )
    k_rot = jnp.concatenate(
        [k1 * cos - k2 * sin, k1 * sin + k2 * cos],
        axis=-1,
    )

    return q_rot.astype(q.dtype), k_rot.astype(k.dtype)
