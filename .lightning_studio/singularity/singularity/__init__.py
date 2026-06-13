from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["SingularityConfig", "SingularityTransformer", "__version__"]


def __getattr__(name: str):
    if name == "SingularityConfig":
        from .model.config import SingularityConfig

        return SingularityConfig
    if name == "SingularityTransformer":
        from .model.transformer import SingularityTransformer

        return SingularityTransformer
    raise AttributeError(name)
