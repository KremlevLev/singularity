def validate_qwen_config(config: dict) -> None:
    hidden_size = config["hidden_size"]
    q_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]

    assert hidden_size == q_heads * head_dim, (
        f"hidden_size={hidden_size}, но "
        f"num_attention_heads * head_dim={q_heads * head_dim}"
    )

    assert q_heads % kv_heads == 0, (
        f"Q-heads={q_heads} должны делиться на KV-heads={kv_heads}"
    )

    assert head_dim % 2 == 0, (
        f"head_dim={head_dim} должен быть чётным для RoPE"
    )

    assert config["num_hidden_layers"] > 0
    assert config["intermediate_size"] > hidden_size

def assert_layer_keys(weight_map: dict, layer_idx: int, has_qk_norm: bool) -> None:
    prefix = f"model.layers.{layer_idx}"

    expected = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]

    if has_qk_norm:
        expected.extend([
            f"{prefix}.self_attn.q_norm.weight",
            f"{prefix}.self_attn.k_norm.weight",
        ])

    missing = [key for key in expected if key not in weight_map]

    if missing:
        raise KeyError(
            "В index.json отсутствуют обязательные веса:\n"
            + "\n".join(f"  - {key}" for key in missing)
        )

    print(f"Layer {layer_idx}: все обязательные ключи весов найдены.")
