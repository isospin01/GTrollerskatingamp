#!/usr/bin/env python3
"""NVIDIA Eureka reward search for roller-skating Phase 2+.

Implements the Eureka LLM-reward-generation loop:
  1. Send task description + env context to an LLM (GPT-4o by default)
  2. LLM generates a compute_reward(env) -> (Tensor, dict) function
  3. Train the skating policy using the generated reward (resume from Phase 1)
  4. Evaluate the trained policy (forward speed, fall rate, episode length)
  5. Feed metrics back to LLM for iterative improvement
  6. Repeat for K iterations; keep the best policy
  7. Record a demo video of the best policy

Prerequisites
-------------
  conda activate env_isaaclab
  pip install openai                  # for GPT-4o API calls
  export OPENAI_API_KEY=<your_key>    # or pass --api_key

Usage
-----
  # Full Eureka run (5 iterations, 1500 training iters each)
  python scripts/skating/eureka_phase2.py \\
      --phase1_run skating_phase1 \\
      --num_eureka_iters 5 \\
      --train_iters 1500 \\
      --eval_episodes 100

  # Quick smoke-test (2 iters, 200 training iters)
  python scripts/skating/eureka_phase2.py \\
      --phase1_run skating_phase1 \\
      --num_eureka_iters 2 \\
      --train_iters 200 \\
      --eval_episodes 20 \\
      --num_envs 512

  # Use a different LLM
  python scripts/skating/eureka_phase2.py \\
      --model gpt-4-turbo \\
      --phase1_run skating_phase1

Output
------
  eureka_output/
    iter_0/
      reward_fn.py         ← LLM-generated reward function
      training_stdout.txt  ← raw training subprocess output
      eval_metrics.json    ← evaluation results
      fitness.json         ← composite fitness score
    iter_1/ ...
    best/
      reward_fn.py         ← best reward function found
      iter.txt             ← which iteration this came from
    eureka_summary.json    ← all iterations, ranked by fitness
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "skating"
ENV_SOURCE_FILE = (
    REPO_ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/skating/robots/g1_29dof/eureka_env_cfg.py"
)

# The rollerskating repo ships its own unitree_rl_lab package with the Eureka environments
# registered.  The conda env may have a different (upstream) editable install active, so
# we prepend our local source tree to PYTHONPATH in every subprocess we launch.
_REPO_UNITREE_SRC = str(REPO_ROOT / "source" / "unitree_rl_lab")


def _make_subprocess_env(extras: Optional[dict] = None) -> dict:
    """Return os.environ copy with PYTHONPATH prepended to load the rollerskating package."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_REPO_UNITREE_SRC}:{existing}" if existing else _REPO_UNITREE_SRC
    if extras:
        env.update(extras)
    return env


sys.path.insert(0, str(SCRIPTS_DIR))
from eureka_prompts import SYSTEM_PROMPT, initial_prompt, feedback_prompt  # noqa: E402

sys.path.pop(0)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eureka LLM reward search for roller-skating Phase 2+.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--phase1_run",
        type=str,
        default=None,
        help="Name of the Phase 1 training run (directory under logs/rsl_rl/skating_phase1/). "
             "Use 'latest' to auto-select the most recent run. Required.",
    )
    p.add_argument("--num_eureka_iters", type=int, default=5,
                   help="Number of Eureka LLM → train → eval iterations.")
    p.add_argument("--train_iters", type=int, default=1500,
                   help="PPO training iterations per Eureka iteration.")
    p.add_argument("--eval_episodes", type=int, default=100,
                   help="Number of rollout episodes for evaluation.")
    p.add_argument("--num_envs", type=int, default=2048,
                   help="Parallel environments during training.")
    p.add_argument("--eval_envs", type=int, default=32,
                   help="Parallel environments during evaluation.")
    p.add_argument("--eval_device", type=str, default=None,
                   help="CUDA device for eval (default: auto-select GPU with most free memory). "
                        "Set to same as --device to use the training GPU.")
    p.add_argument("--eval_sleep", type=int, default=30,
                   help="Seconds to wait after training before launching eval, "
                        "to allow GPU memory to be released on shared servers.")
    p.add_argument(
        "--model", type=str, default="gpt-5.4-pro",
        help=(
            "LLM model for reward generation.  Provider inferred from name:\n"
            "  --- OpenAI (OPENAI_API_KEY) ---\n"
            "  gpt-5.4-pro                 → GPT-5.4 Thinking (recommended — strongest)\n"
            "  gpt-5.4                     → GPT-5.4 standard\n"
            "  o3-mini                     → OpenAI o3-mini\n"
            "  gpt-4o                      → GPT-4o\n"
            "  --- Anthropic (ANTHROPIC_API_KEY) ---\n"
            "  claude-3-7-sonnet-20250219  → Claude 3.7 Sonnet\n"
            "  claude-3-5-sonnet-20241022  → Claude 3.5 Sonnet\n"
            "  --- Google (GOOGLE_API_KEY) ---\n"
            "  gemini-2.0-flash            → Gemini 2.0 Flash\n"
        ),
    )
    p.add_argument(
        "--api_key", type=str, default=None,
        help="API key for the chosen provider (overrides env vars: "
             "ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY).",
    )
    p.add_argument("--output_dir", type=str, default="eureka_output",
                   help="Directory to save Eureka outputs (relative to repo root).")
    p.add_argument("--resume_iter", type=int, default=None,
                   help="Resume from this Eureka iteration (re-uses existing reward_fn.py).")
    p.add_argument("--record_best", action="store_true", default=True,
                   help="Record a demo video of the best policy after all iterations.")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="CUDA device for training and eval subprocesses.")
    p.add_argument("--headless", action="store_true", default=True,
                   help="Run training/eval headless (no GUI).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LLM interface  (OpenAI · Anthropic · Google Gemini)
# ---------------------------------------------------------------------------
#
# Model name conventions used by this script:
#   "gpt-4o"                          → OpenAI (default)
#   "gpt-4o-mini"                     → OpenAI
#   "o3-mini"                         → OpenAI
#   "claude-3-7-sonnet-20250219"      → Anthropic  (recommended: strongest coder)
#   "claude-3-5-sonnet-20241022"      → Anthropic
#   "gemini-2.0-flash"                → Google Gemini
#   "gemini-2.0-pro-exp"              → Google Gemini
#
# Provider is inferred from the model name prefix.
# API keys:  OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY
# Or pass --api_key for any single provider.

def _infer_provider(model: str) -> str:
    """Return 'anthropic', 'google', or 'openai' based on model name."""
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "google"
    return "openai"  # gpt-*, o1, o3, o4, gpt-5.4, gpt-5.4-pro, etc.


def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten chat messages into a single prompt string for base completion models."""
    parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _call_openai(messages: list[dict], model: str, api_key: str) -> tuple[str, int, int]:
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")
    # 3-minute timeout per call; reasoning models (gpt-5.x, o1, o3, o4) may think longer.
    client = openai.OpenAI(api_key=api_key, timeout=180.0)

    m = model.lower()
    # Reasoning/chat models: gpt-5.x, o1, o3, o4 use max_completion_tokens, no temperature.
    is_reasoning = m.startswith(("o1", "o3", "o4"))
    # gpt-5.x may be either a chat model (gpt-5.4) or a base completion model (gpt-5.4-pro).
    # We try chat first; if the API returns a 404 "not a chat model" error we fall back to
    # the legacy completions endpoint automatically.
    is_gpt5 = m.startswith("gpt-5")

    if is_reasoning:
        kwargs: dict = {"model": model, "messages": messages, "max_completion_tokens": 4000}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens

    if is_gpt5:
        # Try chat completions first; fall back to legacy completions if refused.
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=4000
            )
            return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens
        except Exception as e:
            if "not a chat model" not in str(e).lower() and "404" not in str(e):
                raise
            # Base/completion model — use legacy endpoint.
            prompt = _messages_to_prompt(messages)
            resp = client.completions.create(model=model, prompt=prompt, max_tokens=4000)
            text = resp.choices[0].text.strip()
            in_toks = resp.usage.prompt_tokens
            out_toks = resp.usage.completion_tokens
            return text, in_toks, out_toks

    # Standard chat models (gpt-4o, gpt-4-turbo, etc.)
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=3000, temperature=0.7
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_anthropic(messages: list[dict], model: str, api_key: str) -> tuple[str, int, int]:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    # Anthropic separates system from user/assistant turns
    system_text = ""
    turns = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    resp = client.messages.create(
        model=model,
        max_tokens=3000,
        system=system_text.strip() or anthropic.NOT_GIVEN,
        messages=turns,
    )
    text = resp.content[0].text
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_google(messages: list[dict], model: str, api_key: str) -> tuple[str, int, int]:
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("pip install google-generativeai")
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model_name=model)
    # Flatten to a single prompt string (Gemini free-tier doesn't support multi-turn system)
    full_prompt = "\n\n".join(
        f"[{m['role'].upper()}]\n{m['content']}" for m in messages
    )
    resp = gmodel.generate_content(full_prompt)
    text = resp.text
    # Gemini doesn't always expose token counts reliably
    in_toks = getattr(getattr(resp, "usage_metadata", None), "prompt_token_count", 0)
    out_toks = getattr(getattr(resp, "usage_metadata", None), "candidates_token_count", 0)
    return text, in_toks, out_toks


def call_llm(messages: list[dict], model: str, api_key: str) -> str:
    """Send messages to the appropriate LLM provider and return the response text.

    Provider is inferred from the model name:
      claude-*  → Anthropic   (needs ANTHROPIC_API_KEY)
      gemini-*  → Google      (needs GOOGLE_API_KEY)
      *         → OpenAI      (needs OPENAI_API_KEY)
    """
    provider = _infer_provider(model)

    # Resolve API key: explicit arg > provider-specific env var > generic OPENAI_API_KEY
    key = api_key
    if not key:
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "google":    "GOOGLE_API_KEY",
            "openai":    "OPENAI_API_KEY",
        }
        key = os.environ.get(env_map[provider], "") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            f"No API key found for provider '{provider}'.\n"
            f"  Set {env_map.get(provider, 'OPENAI_API_KEY')} environment variable."
        )

    dispatch = {"openai": _call_openai, "anthropic": _call_anthropic, "google": _call_google}

    # Thread-based timeout: httpx keepalive keeps TCP connections ESTAB even when
    # the server is stalled, so a plain socket/read timeout never fires.  We wrap
    # each attempt in a ThreadPoolExecutor and use shutdown(wait=False) to abandon
    # the stuck thread rather than blocking in the context-manager __exit__.
    import concurrent.futures
    _LLM_TIMEOUT_S = 180.0  # 3 minutes per attempt

    last_exc: Exception = RuntimeError("LLM call failed")
    for attempt in range(1, 4):  # up to 3 attempts
        print(
            f"  [LLM] Calling {model} [{provider}] ({len(messages)} messages)"
            + (f"  [attempt {attempt}/3]" if attempt > 1 else "") + "…",
            flush=True,
        )
        t0 = time.time()
        _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        _fut = _ex.submit(dispatch[provider], messages, model, key)
        try:
            text, in_toks, out_toks = _fut.result(timeout=_LLM_TIMEOUT_S)
            _ex.shutdown(wait=False)
            elapsed = time.time() - t0
            print(f"  [LLM] Done in {elapsed:.1f}s  (in={in_toks}, out={out_toks})")
            return text
        except concurrent.futures.TimeoutError:
            _ex.shutdown(wait=False)
            elapsed = time.time() - t0
            print(f"  [LLM] TIMEOUT on attempt {attempt} after {elapsed:.1f}s", flush=True)
            last_exc = TimeoutError(f"LLM call timed out after {_LLM_TIMEOUT_S}s")
        except Exception as exc:
            _ex.shutdown(wait=False)
            elapsed = time.time() - t0
            print(f"  [LLM] ERROR on attempt {attempt} after {elapsed:.1f}s: {exc}", flush=True)
            last_exc = exc
        if attempt < 3:
            wait = 30 * attempt
            print(f"  [LLM] Retrying in {wait}s…", flush=True)
            time.sleep(wait)
    raise last_exc


def extract_code_block(llm_response: str) -> Optional[str]:
    """Extract the first ```python ... ``` block from the LLM response."""
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    if not matches:
        # Fall back: try to find def compute_reward directly
        if "def compute_reward" in llm_response:
            # Return everything from the first import / def onwards
            idx = llm_response.find("import")
            if idx == -1:
                idx = llm_response.find("def compute_reward")
            return llm_response[idx:]
        return None
    return matches[0].strip()


def validate_code(code: str) -> tuple[bool, str]:
    """Check that the code compiles and defines compute_reward."""
    try:
        compiled = compile(code, "<eureka_generated>", "exec")
        ns: dict = {}
        exec(compiled, ns)  # noqa: S102
        if "compute_reward" not in ns:
            return False, "Function 'compute_reward' not defined in generated code."
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    except Exception as exc:
        return False, f"Runtime error during validation: {exc}"


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

_ISAAC_PYTHON: Optional[str] = None

def _conda_python() -> str:
    """Return the Python interpreter for the Isaac Lab conda environment.

    Search order:
      1. ISAACLAB_PYTHON env var  (user override)
      2. env_isaaclab conda env   (standard install)
      3. Current sys.executable   (fallback — may not have Isaac Lab)
    """
    global _ISAAC_PYTHON
    if _ISAAC_PYTHON is not None:
        return _ISAAC_PYTHON

    # User override
    override = os.environ.get("ISAACLAB_PYTHON")
    if override and os.path.isfile(override):
        _ISAAC_PYTHON = override
        return _ISAAC_PYTHON

    # Look for env_isaaclab in conda
    import shutil
    candidates = [
        # Standard miniconda/anaconda paths
        os.path.expanduser("~/miniconda3/envs/env_isaaclab/bin/python"),
        os.path.expanduser("~/anaconda3/envs/env_isaaclab/bin/python"),
        os.path.expanduser("~/conda/envs/env_isaaclab/bin/python"),
    ]
    # Also try conda run to locate it dynamically
    conda_bin = shutil.which("conda")
    if conda_bin:
        try:
            result = subprocess.run(
                [conda_bin, "run", "-n", "env_isaaclab", "which", "python"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                candidates.insert(0, result.stdout.strip())
        except Exception:
            pass

    for c in candidates:
        if c and os.path.isfile(c):
            _ISAAC_PYTHON = c
            print(f"[Eureka] Using Isaac Lab Python: {c}")
            return _ISAAC_PYTHON

    print("[Eureka] WARNING: Could not find env_isaaclab Python; falling back to sys.executable.")
    print("  Set ISAACLAB_PYTHON=/path/to/env_isaaclab/bin/python to override.")
    _ISAAC_PYTHON = sys.executable
    return _ISAAC_PYTHON


def _best_free_gpu(label: str = "training") -> str:
    """Return the CUDA device string for the GPU with the most free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            free = [int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip()]
            # Print all GPUs so the user can see the full picture
            for i, f in enumerate(free):
                print(f"  [GPU] cuda:{i}  {f:6d} MiB free")
            best = max(range(len(free)), key=lambda i: free[i])
            print(f"  [GPU] Auto-selected cuda:{best} ({free[best]} MiB free) for {label}.")
            return f"cuda:{best}"
    except Exception:
        pass
    return "cuda:0"


def find_phase1_checkpoint(phase1_run: str) -> Optional[pathlib.Path]:
    """Locate the most recent model checkpoint from a Phase 1 run.

    Searches in order:
      1. REPO_ROOT/logs/rsl_rl/skating_phase1/           (new location)
      2. ~/unitree_rl_lab/logs/rsl_rl/skating_phase1/    (legacy location)
      3. ~/unitree_rl_lab/logs/rsl_rl/skating_phase1_v1/ (legacy v1 variant)
    """
    candidate_bases = [
        REPO_ROOT / "logs" / "rsl_rl" / "skating_phase1",
        pathlib.Path.home() / "unitree_rl_lab" / "logs" / "rsl_rl" / "skating_phase1",
        pathlib.Path.home() / "unitree_rl_lab" / "logs" / "rsl_rl" / "skating_phase1_v1",
    ]

    for log_base in candidate_bases:
        if not log_base.exists():
            continue

        # Only consider timestamp-style run directories (YYYY-MM-DD_HH-MM-SS)
        _ts_pat = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
        run_dirs = sorted(
            [d for d in log_base.iterdir() if d.is_dir() and _ts_pat.match(d.name)],
            key=lambda p: p.name,  # lexicographic = chronological for ISO timestamps
            reverse=True,
        )

        if phase1_run != "latest":
            run_dirs = [d for d in run_dirs if phase1_run in d.name] or run_dirs

        for run_dir in run_dirs:
            ckpts = sorted(
                run_dir.glob("model_*.pt"),
                key=lambda p: int(re.search(r"(\d+)", p.name).group(1)),
            )
            if ckpts:
                print(f"[Eureka] Found Phase 1 checkpoint: {ckpts[-1]}")
                return ckpts[-1]
            fallback = run_dir / "model.pt"
            if fallback.exists():
                return fallback

    return None


def run_training(
    iter_dir: pathlib.Path,
    reward_fn_path: pathlib.Path,
    phase1_checkpoint: Optional[pathlib.Path],
    train_iters: int,
    num_envs: int,
    device: str,
    headless: bool,
    experiment_name: str,
) -> tuple[bool, str]:
    """Launch train.py as a subprocess for one Eureka iteration.

    Returns (success, stdout_tail).
    """
    # Always pick the GPU with the most free memory; the user's --device is the fallback
    # only when nvidia-smi is unavailable.  This avoids OOM crashes on shared servers
    # where the default cuda:0 may be nearly full.
    train_device = _best_free_gpu("training")

    cmd = [
        _conda_python(),
        str(SCRIPTS_DIR / "train.py"),
        "--task", "Unitree-G1-Skating-Eureka-v0",
        "--num_envs", str(num_envs),
        "--max_iterations", str(train_iters),
        "--experiment_name", experiment_name,
        "--device", train_device,
    ]
    if headless:
        cmd += ["--headless"]
    if phase1_checkpoint is not None:
        # Use --resume_path to pass the absolute checkpoint path directly,
        # bypassing the logs/ tree lookup (the checkpoint may be in a different repo).
        cmd += ["--resume_path", str(phase1_checkpoint)]

    env = _make_subprocess_env({"EUREKA_REWARD_FN_PATH": str(reward_fn_path)})

    stdout_path = iter_dir / "training_stdout.txt"
    print(f"  [Train] Launching training (max_iters={train_iters}, envs={num_envs})…")
    print(f"  [Train] Log → {stdout_path}")

    t0 = time.time()
    with open(stdout_path, "w") as fout:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fout,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - t0
    print(f"  [Train] Done in {elapsed / 60:.1f} min  exit_code={result.returncode}")

    # Grab the last 60 lines for LLM feedback
    with open(stdout_path) as f:
        lines = f.readlines()
    tail = "".join(lines[-60:])

    # Isaac Sim may swallow Python exceptions and still exit 0; detect via stdout.
    stdout_text = "".join(lines)
    crashed = result.returncode != 0 or "Traceback (most recent call last)" in stdout_text
    if crashed:
        if "out of memory" in stdout_text.lower() or "cudaErrorMemoryAllocation" in stdout_text:
            print(f"  [Train] ERROR — GPU out of memory on {train_device}.")
            print(f"  [Train]   Try reducing --num_envs (currently {num_envs}).")
            print(f"  [Train]   Example: NUM_ENVS=1024 bash scripts/skating/run_eureka.sh")
        elif result.returncode == 0:
            print("  [Train] WARNING — subprocess exited 0 but stdout contains a traceback; treating as failure.")

    return not crashed, tail


def run_eval(
    iter_dir: pathlib.Path,
    experiment_name: str,
    reward_fn_path: pathlib.Path,
    eval_episodes: int,
    eval_envs: int,
    device: str,
    headless: bool,
    eval_device: Optional[str] = None,
) -> Optional[dict]:
    """Launch play.py as a subprocess to evaluate the trained policy.

    Returns a metrics dict or None if evaluation failed.
    """
    # Use a separate GPU for eval if specified, otherwise pick the one with most free memory
    _eval_device = eval_device or _best_free_gpu("evaluation")

    cmd = [
        _conda_python(),
        str(SCRIPTS_DIR / "play.py"),
        "--task", "Unitree-G1-Skating-Eureka-v0",
        "--num_envs", str(eval_envs),
        "--eval_episodes", str(eval_episodes),
        "--load_experiment", experiment_name,
        "--load_run", "latest",
        "--device", _eval_device,
    ]
    if headless:
        cmd += ["--headless"]

    env = _make_subprocess_env({"EUREKA_REWARD_FN_PATH": str(reward_fn_path)})

    eval_stdout = iter_dir / "eval_stdout.txt"
    print(f"  [Eval] Running {eval_episodes} episodes ({eval_envs} envs)…")

    with open(eval_stdout, "w") as fout:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fout,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        print("  [Eval] WARNING — evaluation subprocess failed.")
        return None

    # play.py saves eval_metrics.csv to logs/rsl_rl/<experiment_name>/eval_metrics.csv
    csv_path = REPO_ROOT / "logs" / "rsl_rl" / experiment_name / "eval_metrics.csv"
    if not csv_path.exists():
        print(f"  [Eval] WARNING — metrics CSV not found at {csv_path}")
        return None

    metrics: dict[str, list[float]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                metrics.setdefault(k, []).append(float(v))

    summary = {
        "mean_forward_speed":   _mean(metrics.get("forward_speed", [])),
        "mean_lateral_drift":   _mean(metrics.get("lateral_drift", [])),
        "mean_episode_length":  _mean(metrics.get("episode_length", [])),
        "fall_rate_pct":        _mean(metrics.get("fall", [])) * 100.0,
    }
    summary["fitness_score"] = compute_fitness(summary)
    return summary


def run_record_demo(
    best_iter_dir: pathlib.Path,
    experiment_name: str,
    reward_fn_path: pathlib.Path,
    device: str,
) -> None:
    """Record a demo video of the best policy."""
    cmd = [
        _conda_python(),
        str(SCRIPTS_DIR / "record_demo.py"),
        "--task", "Unitree-G1-Skating-Eureka-v0",
        "--load_experiment", experiment_name,
        "--load_run", "latest",
        "--clip", "straight",
        "--device", device,
    ]
    env = _make_subprocess_env({"EUREKA_REWARD_FN_PATH": str(reward_fn_path)})

    print("  [Demo] Recording demo video for best policy…")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def compute_fitness(eval_metrics: dict) -> float:
    """Combine eval metrics into a single fitness score in [0, 1].

    Weights tuned so that:
      - Surviving (not falling) is the top priority
      - Speed is important but capped at 3.5 m/s
      - Short lateral drift is rewarded
      - Long episodes are rewarded
    """
    speed   = eval_metrics.get("mean_forward_speed",  0.0)
    lat     = eval_metrics.get("mean_lateral_drift",   0.5)
    ep_len  = eval_metrics.get("mean_episode_length",  0.0)
    fall_pct = eval_metrics.get("fall_rate_pct",       100.0)

    survival   = max(0.0, 1.0 - fall_pct / 100.0)          # 0–1
    speed_norm = min(speed / 3.5, 1.0)                      # 0–1  (3.5 m/s = perfect)
    lat_norm   = max(0.0, 1.0 - lat / 0.5)                  # 0–1  (0.5 m/s drift = zero)
    len_norm   = min(ep_len / 20.0, 1.0)                     # 0–1  (20 s = perfect)

    return 0.4 * survival + 0.35 * speed_norm + 0.15 * lat_norm + 0.10 * len_norm


def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


# ---------------------------------------------------------------------------
# Training summary from stdout
# ---------------------------------------------------------------------------

def parse_training_summary(stdout_tail: str) -> dict:
    """Extract a few key numbers from the training subprocess stdout."""
    summary: dict = {"stdout_tail": stdout_tail, "converged": False}

    # Look for mean reward lines (RSL-RL format: "Mean episode rew: X.XX")
    reward_matches = re.findall(r"Mean episode rew[:\s]+([+-]?\d+\.?\d*)", stdout_tail)
    if reward_matches:
        rewards = [float(r) for r in reward_matches]
        first, last = rewards[0], rewards[-1]
        if last > first * 1.05:
            trend = f"improving  ({first:.2f} → {last:.2f})"
        elif last < first * 0.95:
            trend = f"degrading  ({first:.2f} → {last:.2f})"
        else:
            trend = f"stable     ({first:.2f} → {last:.2f})"
        summary["reward_trend"] = trend
        summary["converged"] = abs(last - first) < 0.1 * abs(first + 1e-6)
    else:
        summary["reward_trend"] = "no data found in stdout"

    # Episode length
    ep_matches = re.findall(r"Mean episode length[:\s]+([+-]?\d+\.?\d*)", stdout_tail)
    if ep_matches:
        summary["mean_episode_length_s"] = f"{float(ep_matches[-1]):.1f}"

    return summary


# ---------------------------------------------------------------------------
# Main Eureka loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    provider = _infer_provider(args.model)
    _KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}
    api_key = args.api_key or os.environ.get(_KEY_ENV[provider]) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            f"ERROR: No API key found for provider '{provider}' (model: {args.model}).\n"
            f"  Set {_KEY_ENV[provider]} environment variable or pass --api_key."
        )
        sys.exit(1)

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate Phase 1 checkpoint
    phase1_run_arg = args.phase1_run or "latest"
    phase1_checkpoint = find_phase1_checkpoint(phase1_run_arg)
    if phase1_checkpoint is None:
        print(
            f"WARNING: Phase 1 checkpoint not found (run='{phase1_run_arg}').\n"
            "  Training will start from scratch (no Phase 1 warm-start)."
        )
    else:
        print(f"[Eureka] Phase 1 checkpoint: {phase1_checkpoint}")

    # Read env source for LLM context
    env_source_excerpt = ENV_SOURCE_FILE.read_text()[:4000]  # first ~4000 chars

    # Conversation history for multi-turn LLM dialogue
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    best_fitness = -1.0
    best_iter = -1
    all_results: list[dict] = []

    print(f"\n{'='*70}")
    print(f"  Eureka Roller-Skating Reward Search")
    print(f"  Iterations : {args.num_eureka_iters}")
    print(f"  Train iters: {args.train_iters} per Eureka iter")
    print(f"  LLM model  : {args.model}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*70}\n")

    for eureka_iter in range(args.num_eureka_iters):
        iter_dir = output_dir / f"iter_{eureka_iter}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        experiment_name = f"eureka_iter_{eureka_iter}"

        print(f"\n{'─'*60}")
        print(f"  Eureka iteration {eureka_iter + 1} / {args.num_eureka_iters}")
        print(f"{'─'*60}")

        # ── 1. Generate reward function via LLM ───────────────────────────────
        reward_fn_path = iter_dir / "reward_fn.py"

        if args.resume_iter is not None and eureka_iter < args.resume_iter and reward_fn_path.exists():
            print(f"  [LLM] Skipping (resuming from iter {args.resume_iter})")
            code = reward_fn_path.read_text()
        else:
            if eureka_iter == 0:
                user_msg = initial_prompt(env_source_excerpt)
            else:
                prev_result = all_results[-1]
                user_msg = feedback_prompt(
                    prev_code=prev_result["reward_code"],
                    iteration=eureka_iter,
                    train_summary=prev_result.get("train_summary", {}),
                    eval_metrics=prev_result.get("eval_metrics"),
                )

            messages.append({"role": "user", "content": user_msg})
            response_text = call_llm(messages, model=args.model, api_key=api_key)
            messages.append({"role": "assistant", "content": response_text})

            # Save raw LLM response
            (iter_dir / "llm_response.txt").write_text(response_text)

            code = extract_code_block(response_text)
            if code is None:
                print("  [LLM] ERROR — could not extract a code block from LLM response.")
                print("         Skipping this iteration.")
                all_results.append({
                    "iter": eureka_iter,
                    "reward_code": "",
                    "train_summary": {},
                    "eval_metrics": None,
                    "fitness": 0.0,
                    "error": "code extraction failed",
                })
                continue

            # Validate syntax
            valid, err = validate_code(code)
            if not valid:
                print(f"  [LLM] WARNING — generated code has errors: {err}")
                print("         Attempting to continue anyway (errors may be env-specific).")

            # Add a standard header to generated code
            header = textwrap.dedent(f"""\
                # Eureka-generated reward function for roller skating
                # Iteration {eureka_iter}, generated {datetime.now().isoformat()}
                # Model: {args.model}

                import math
                import torch
                from isaaclab.managers import SceneEntityCfg
                try:
                    from unitree_rl_lab.tasks.skating.mdp import (
                        forward_velocity_tracking_exp,
                        lateral_velocity_penalty,
                        skate_glide_continuity,
                        push_off_rhythm,
                        skate_foot_edge_contact,
                        upright_orientation_l2,
                        base_height_skating_l2,
                        skate_energy,
                        skate_action_rate,
                    )
                except ImportError:
                    pass  # helpers unavailable outside Isaac Sim

            """)
            # Only prepend header if code doesn't already have imports
            if "import torch" not in code:
                code = header + code

            reward_fn_path.write_text(code)
            print(f"  [LLM] Reward function saved → {reward_fn_path}")

        # ── 2. Train with generated reward ────────────────────────────────────
        success, stdout_tail = run_training(
            iter_dir=iter_dir,
            reward_fn_path=reward_fn_path,
            phase1_checkpoint=phase1_checkpoint,
            train_iters=args.train_iters,
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            experiment_name=experiment_name,
        )
        train_summary = parse_training_summary(stdout_tail)
        (iter_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2))

        if not success:
            print("  [Train] Training subprocess failed — skipping eval.")
            result_entry = {
                "iter": eureka_iter,
                "reward_code": reward_fn_path.read_text(),
                "train_summary": train_summary,
                "eval_metrics": None,
                "fitness": 0.0,
                "error": "training failed",
            }
            all_results.append(result_entry)
            continue

        # ── 3. Evaluate trained policy ─────────────────────────────────────────
        # Wait for GPU memory to be released on shared servers before launching eval
        if args.eval_sleep > 0:
            print(f"  [Eval] Waiting {args.eval_sleep}s for GPU memory to settle…")
            time.sleep(args.eval_sleep)

        eval_metrics = run_eval(
            iter_dir=iter_dir,
            experiment_name=experiment_name,
            reward_fn_path=reward_fn_path,
            eval_episodes=args.eval_episodes,
            eval_envs=args.eval_envs,
            device=args.device,
            headless=args.headless,
            eval_device=args.eval_device,
        )

        fitness = eval_metrics["fitness_score"] if eval_metrics else 0.0
        (iter_dir / "eval_metrics.json").write_text(
            json.dumps(eval_metrics or {}, indent=2)
        )
        (iter_dir / "fitness.json").write_text(
            json.dumps({"fitness": fitness}, indent=2)
        )

        # ── 4. Track best ──────────────────────────────────────────────────────
        if fitness > best_fitness:
            best_fitness = fitness
            best_iter = eureka_iter
            best_dir = output_dir / "best"
            best_dir.mkdir(exist_ok=True)
            shutil.copy(reward_fn_path, best_dir / "reward_fn.py")
            (best_dir / "iter.txt").write_text(
                f"Best from iteration {eureka_iter}  fitness={fitness:.4f}\n"
            )
            print(f"  [Best] New best! iter={eureka_iter}  fitness={fitness:.4f}")

        result_entry = {
            "iter": eureka_iter,
            "reward_code": reward_fn_path.read_text(),
            "train_summary": train_summary,
            "eval_metrics": eval_metrics,
            "fitness": fitness,
        }
        all_results.append(result_entry)

        # Print iteration summary
        if eval_metrics:
            print(
                f"\n  ── Iteration {eureka_iter} summary ──\n"
                f"     Forward speed : {eval_metrics['mean_forward_speed']:.2f} m/s\n"
                f"     Fall rate     : {eval_metrics['fall_rate_pct']:.1f}%\n"
                f"     Episode len   : {eval_metrics['mean_episode_length']:.1f} s\n"
                f"     Fitness score : {fitness:.4f}\n"
            )

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Eureka search complete")
    print(f"  Best iteration : {best_iter}  (fitness={best_fitness:.4f})")
    print(f"  Best reward fn : {output_dir / 'best' / 'reward_fn.py'}")
    print(f"{'='*70}\n")

    # Save full summary
    summary_path = output_dir / "eureka_summary.json"
    summary_data = [
        {k: v for k, v in r.items() if k != "reward_code"}
        for r in all_results
    ]
    summary_data.sort(key=lambda x: x.get("fitness", 0), reverse=True)
    summary_path.write_text(json.dumps(summary_data, indent=2))
    print(f"Summary saved → {summary_path}")

    # ── Record demo of best policy ────────────────────────────────────────────
    if args.record_best and best_iter >= 0:
        print("\n[Demo] Recording demo video for best policy…")
        best_reward_fn = output_dir / "best" / "reward_fn.py"
        run_record_demo(
            best_iter_dir=output_dir / f"iter_{best_iter}",
            experiment_name=f"eureka_iter_{best_iter}",
            reward_fn_path=best_reward_fn,
            device=args.device,
        )


if __name__ == "__main__":
    main()
