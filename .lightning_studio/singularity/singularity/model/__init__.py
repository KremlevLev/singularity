from __future__ import annotations

__all__ = [
    "MLAConfig",
    "MoEConfig",
    "QuantizationConfig",
    "SingularityConfig",
    "SingularityTransformer",
]


def __getattr__(name: str):
    if name in {"MLAConfig", "MoEConfig", "QuantizationConfig", "SingularityConfig"}:
        from .config import MLAConfig, MoEConfig, QuantizationConfig, SingularityConfig

        return {
            "MLAConfig": MLAConfig,
            "MoEConfig": MoEConfig,
            "QuantizationConfig": QuantizationConfig,
            "SingularityConfig": SingularityConfig,
        }[name]
    if name == "SingularityTransformer":
        from .transformer import SingularityTransformer

        return SingularityTransformer
    raise AttributeError(name)
