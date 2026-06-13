from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MLAConfig:
    latent_kv_dim: int = 512
    q_lora_rank: int = 64
    kv_lora_rank: int = 512


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 16
    shared_experts: int = 2
    routed_experts: int = 14
    top_k: int = 2
    expert_capacity_factor: float = 1.25
    noisy_gating: bool = True
    router_z_loss_coef: float = 1e-4
    load_balance_coef: float = 1e-3


@dataclass(frozen=True)
class QuantizationConfig:
    enabled: bool = False
    dtype: str = "int4"
    qat: bool = True


@dataclass(frozen=True)
class SingularityConfig:
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_layers: int = 64
    num_heads: int = 32
    num_kv_heads: int = 8
    max_seq_len: int = 16384
    rope_theta: float = 500000.0
    tie_embeddings: bool = False
    mla: MLAConfig | None = None
    moe: MoEConfig | None = None
    dora_rank: int = 128
    quantization: QuantizationConfig | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SingularityConfig":
        data = dict(data or {})
        return cls(
            vocab_size=int(data.get("vocab_size", cls.vocab_size)),
            hidden_size=int(data.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(data.get("intermediate_size", cls.intermediate_size)),
            num_layers=int(data.get("num_layers", cls.num_layers)),
            num_heads=int(data.get("num_heads", cls.num_heads)),
            num_kv_heads=int(data.get("num_kv_heads", cls.num_kv_heads)),
            max_seq_len=int(data.get("max_seq_len", cls.max_seq_len)),
            rope_theta=float(data.get("rope_theta", cls.rope_theta)),
            tie_embeddings=bool(data.get("tie_embeddings", cls.tie_embeddings)),
            mla=MLAConfig(**(data.get("mla") or {})),
            moe=MoEConfig(**(data.get("moe") or {})),
            dora_rank=int(data.get("dora_rank", data.get("dora", {}).get("rank", cls.dora_rank))),
            quantization=QuantizationConfig(**(data.get("quantization") or {})),
        )

@classmethod
    def get_debug_config(cls) -> "SingularityConfig":
        """Возвращает крошечный конфиг для тестов на CPU."""
        return cls(
            vocab_size=1024,      # Маленький словарь для тестов
            hidden_size=128,      # Крошечная размерность
            intermediate_size=256,
            num_layers=2,         # Всего 2 слоя
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=256,
            mla=None,             # Отключаем тяжелые фичи для начала
            moe=None,
            dora_rank=8
        )

