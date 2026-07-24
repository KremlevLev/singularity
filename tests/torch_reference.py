import math
import torch
import torch.nn.functional as F


def torch_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(variance + eps)

    return x_norm.to(dtype=x.dtype) * weight.to(dtype=x.dtype)


def torch_apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q: (batch, seq, q_heads, head_dim)
    k: (batch, seq, kv_heads, head_dim)
    position_ids: (batch, seq)
    """

    assert head_dim % 2 == 0

    inv_freq = 1.0 / (
        theta ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=q.device,
                dtype=torch.float32,
            ) / head_dim
        )
    )

    freqs = position_ids.to(torch.float32)[..., None] * inv_freq
    cos = torch.cos(freqs)[:, :, None, :]
    sin = torch.sin(freqs)[:, :, None, :]

    q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
    k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]

    q_rot = torch.cat(
        [q1 * cos - q2 * sin, q1 * sin + q2 * cos],
        dim=-1,
    )
    k_rot = torch.cat(
        [k1 * cos - k2 * sin, k1 * sin + k2 * cos],
        dim=-1,
    )

    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def torch_qwen_attention(
    x: torch.Tensor,
    input_norm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor | None,
    k_norm_weight: torch.Tensor | None,
    position_ids: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    rope_theta: float,
    eps: float,
) -> torch.Tensor:
    """
    Возвращает результат attention residual-блока:
    residual + self_attn(RMSNorm(x))
    """

    residual = x
    x = torch_rms_norm(x, input_norm_weight, eps)

    q = F.linear(x, q_proj_weight)
    k = F.linear(x, k_proj_weight)
    v = F.linear(x, v_proj_weight)

    batch_size, seq_len, _ = q.shape

    q = q.reshape(batch_size, seq_len, num_attention_heads, head_dim)
    k = k.reshape(batch_size, seq_len, num_key_value_heads, head_dim)
    v = v.reshape(batch_size, seq_len, num_key_value_heads, head_dim)

    if q_norm_weight is not None:
        q = torch_rms_norm(q, q_norm_weight, eps)

    if k_norm_weight is not None:
        k = torch_rms_norm(k, k_norm_weight, eps)

    q, k = torch_apply_rotary_emb(
        q=q,
        k=k,
        position_ids=position_ids,
        theta=rope_theta,
        head_dim=head_dim,
    )

    groups = num_attention_heads // num_key_value_heads
    k = torch.repeat_interleave(k, groups, dim=2)
    v = torch.repeat_interleave(v, groups, dim=2)

    # (B, H, Q, D) @ (B, H, D, K) -> (B, H, Q, K)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-1, -2))
    scores = scores / math.sqrt(head_dim)

    causal_mask = torch.triu(
        torch.ones(
            (seq_len, seq_len),
            dtype=torch.bool,
            device=x.device,
        ),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

    probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)

    output = torch.matmul(probs, v)
    output = output.transpose(1, 2).contiguous()
    output = output.reshape(batch_size, seq_len, -1)

    output = F.linear(output, o_proj_weight)

    return residual + output


def torch_qwen_mlp(
    x: torch.Tensor,
    post_attn_norm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    residual = x

    x = torch_rms_norm(x, post_attn_norm_weight, eps)
    gate = F.linear(x, gate_proj_weight)
    up = F.linear(x, up_proj_weight)

    x = F.silu(gate) * up
    x = F.linear(x, down_proj_weight)

    return residual + x


def torch_qwen_decoder_layer(
    x: torch.Tensor,
    weights: dict,
    position_ids: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    x = torch_qwen_attention(
        x=x,
        input_norm_weight=weights["input_layernorm.weight"],
        q_proj_weight=weights["self_attn.q_proj.weight"],
        k_proj_weight=weights["self_attn.k_proj.weight"],
        v_proj_weight=weights["self_attn.v_proj.weight"],
        o_proj_weight=weights["self_attn.o_proj.weight"],
        q_norm_weight=weights.get("self_attn.q_norm.weight"),
        k_norm_weight=weights.get("self_attn.k_norm.weight"),
        position_ids=position_ids,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        rope_theta=config["rope_theta"],
        eps=config["rms_norm_eps"],
    )

    return torch_qwen_mlp(
        x=x,
        post_attn_norm_weight=weights["post_attention_layernorm.weight"],
        gate_proj_weight=weights["mlp.gate_proj.weight"],
        up_proj_weight=weights["mlp.up_proj.weight"],
        down_proj_weight=weights["mlp.down_proj.weight"],
        eps=config["rms_norm_eps"],
    )
