from __future__ import annotations

__all__ = ["create_app", "run_api", "merge_dora_linear", "MCTSNode", "choose_best_branch"]


def __getattr__(name: str):
    if name in {"create_app", "run_api"}:
        from .api import create_app, run_api

        return {"create_app": create_app, "run_api": run_api}[name]
    if name == "merge_dora_linear":
        from .merge import merge_dora_linear

        return merge_dora_linear
    if name in {"MCTSNode", "choose_best_branch"}:
        from .search import MCTSNode, choose_best_branch

        return {"MCTSNode": MCTSNode, "choose_best_branch": choose_best_branch}[name]
    raise AttributeError(name)
