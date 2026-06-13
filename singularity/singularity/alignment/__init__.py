from __future__ import annotations

__all__ = ["run_grpo", "verify_code_compiles", "verify_math_answer", "sample_tokens"]


def __getattr__(name: str):
    if name == "run_grpo":
        from .grpo import run_grpo

        return run_grpo
    if name in {"verify_code_compiles", "verify_math_answer"}:
        from .rewards import verify_code_compiles, verify_math_answer

        return {"verify_code_compiles": verify_code_compiles, "verify_math_answer": verify_math_answer}[name]
    if name == "sample_tokens":
        from .sampler import sample_tokens

        return sample_tokens
    raise AttributeError(name)
