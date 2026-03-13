"""Eureka dynamic reward injection for roller skating.

The Eureka runner (scripts/skating/eureka_phase2.py) writes each LLM-generated
reward function to a .py file and sets the EUREKA_REWARD_FN_PATH environment
variable before launching a training subprocess.  This module provides the
Isaac Lab reward term that dynamically loads and calls that function.

Expected interface for generated reward functions
-------------------------------------------------
The file at EUREKA_REWARD_FN_PATH must define:

    def compute_reward(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...
        return total_reward, {"component_name": component_tensor, ...}

Where:
    total_reward : torch.Tensor shape (num_envs,)  — per-env scalar reward
    dict values  : same shape, used for TensorBoard logging (optional but helpful)

Caching
-------
The function is loaded once on first call (per unique EUREKA_REWARD_FN_PATH value)
and cached for the rest of the training run.  A safety clamp of [-20, 20] is
applied to prevent reward explosion from buggy generated code.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Module-level cache: avoid reloading on every env step
_cached_fn: Optional[Callable] = None
_cached_path: str = ""


def _load_eureka_fn(path: str) -> Optional[Callable]:
    """Load compute_reward from a generated .py file (cached by path).

    Uses exec()-based loading with torch pre-imported so that LLM-generated
    functions with ``-> tuple[torch.Tensor, ...]`` return annotations work even
    when the function body does ``import torch`` internally.  Python evaluates
    return annotations at function-definition time (PEP 3107), so torch must be
    present in the module namespace when the def statement is executed.
    """
    global _cached_fn, _cached_path

    if path == _cached_path:
        # Already attempted to load this path — return whatever we got
        return _cached_fn

    _cached_path = path
    _cached_fn = None

    try:
        with open(path, "r") as fh:
            source = fh.read()

        # Pre-import torch so type annotations like -> tuple[torch.Tensor, ...]
        # resolve correctly at function-definition time.
        import torch as _torch
        module_ns: dict = {"torch": _torch, "__file__": path, "__name__": "eureka_generated_reward"}
        exec(compile(source, path, "exec"), module_ns)  # type: ignore[arg-type]

        if "compute_reward" not in module_ns:
            raise AttributeError(
                f"Generated module at '{path}' must define 'compute_reward(env)'"
            )

        _cached_fn = module_ns["compute_reward"]
        print(f"[Eureka] Loaded reward function from: {path}")

    except Exception as exc:
        print(f"[Eureka] WARNING — failed to load reward function from '{path}': {exc}")

    return _cached_fn


def eureka_task_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Isaac Lab reward term that calls the Eureka LLM-generated reward function.

    The path to the generated .py file is read from the ``EUREKA_REWARD_FN_PATH``
    environment variable set by the Eureka runner before each training subprocess.
    The function is loaded lazily on the first call and then cached.

    Returns zeros if no path is set or if the generated function fails, so
    training continues safely even with a buggy reward function.
    """
    path = os.environ.get("EUREKA_REWARD_FN_PATH", "").strip()
    if not path:
        return torch.zeros(env.num_envs, device=env.device)

    fn = _load_eureka_fn(path)
    if fn is None:
        return torch.zeros(env.num_envs, device=env.device)

    try:
        reward, _components = fn(env)
        # Safety clamp: prevents exploding gradients if the LLM uses bad scales
        return reward.clamp(-20.0, 20.0).to(env.device)
    except Exception as exc:
        print(f"[Eureka] WARNING — reward function runtime error: {exc}")
        return torch.zeros(env.num_envs, device=env.device)
