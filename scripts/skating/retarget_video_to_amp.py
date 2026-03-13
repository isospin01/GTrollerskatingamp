"""Roller-skating video → GENMO SMPL pose → G1-29DOF AMP reference motion.

End-to-end pipeline:

  Step 1  yt-dlp         : Download YouTube/local video → skating_video.mp4
  Step 2  Frame extract  : ffmpeg / OpenCV → frames/ (JPEG at --fps)
  Step 3  SMPL inference : GENMO/GEM (primary) or 4D-Humans (fallback)
                             → SMPL body_pose (T, 63), global_orient (T, 3), transl (T, 3)
  Step 4  Retarget       : smpl_to_g1_retarget.SMPLtoG1Retargeter
                             → G1-29DOF joint angles + root kinematics
  Step 5  AMP obs build  : build_amp_obs_from_retarget()
                             → (T, 85) float32 — same layout as gen_skating_reference.py
  Step 6  Save           : skating_reference_human.npz

Usage::

    # Recommended: default output path (replaces skating_reference_human.npz)
    python scripts/skating/retarget_video_to_amp.py \\
        --video_url https://www.youtube.com/shorts/4b-bjwTWPRA

    # With explicit checkpoint for GENMO/GEM:
    python scripts/skating/retarget_video_to_amp.py \\
        --video_url https://www.youtube.com/shorts/4b-bjwTWPRA \\
        --genmo_ckpt /path/to/gem_checkpoint.ckpt \\
        --fps 30 --output data/my_reference.npz

    # Skip download if video already exists:
    python scripts/skating/retarget_video_to_amp.py \\
        --video_path /path/to/existing/video.mp4

    # Use 4D-Humans (HMR2) backend instead of GENMO:
    python scripts/skating/retarget_video_to_amp.py \\
        --video_url https://www.youtube.com/shorts/4b-bjwTWPRA \\
        --backend hmr2

    # Combine with existing analytical reference (augment, not replace):
    python scripts/skating/retarget_video_to_amp.py \\
        --video_url https://www.youtube.com/shorts/4b-bjwTWPRA \\
        --merge_analytical

Setup (run once before using this script)::

    # GENMO / GEM (primary backend — NVlabs research model):
    pip install git+https://github.com/NVlabs/GENMO.git
    # Download checkpoint from the GENMO release page or HuggingFace

    # 4D-Humans / HMR2 (fallback backend — easier to install):
    pip install 4d-humans
    # Or: pip install git+https://github.com/shubham-goel/4D-Humans.git

    # Always needed:
    pip install yt-dlp smplx scipy

Notes on GENMO vs 4D-Humans:
  - GENMO: Diffusion-based, better global consistency (good camera handling,
    long sequences); outputs SMPL-X; needs GPU.
  - 4D-Humans (HMR2): Transformer-based, per-frame with temporal smoothing;
    outputs SMPL; very robust, well-tested.
  Both output SMPL body_pose + global_orient + transl; the retargeter handles both.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

# Ensure this repo's scripts/ is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_SKATING = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_SKATING))

from smpl_to_g1_retarget import SMPLtoG1Retargeter, build_amp_obs_from_retarget

# ─── Default paths ─────────────────────────────────────────────────────────────

_DEFAULT_OUTPUT = _REPO_ROOT / "source" / "unitree_rl_lab" / "unitree_rl_lab" / "data" / "skating_reference_human.npz"
_DEFAULT_ANALYTICAL = _REPO_ROOT / "source" / "unitree_rl_lab" / "unitree_rl_lab" / "data" / "skating_reference.npz"
_DEFAULT_WORKDIR = _REPO_ROOT / "data" / "retarget_workdir"


# ─── Step 1: Download video ────────────────────────────────────────────────────

def download_video(url: str, out_path: Path, fps: int) -> Path:
    """Download a video from URL using yt-dlp.

    Args:
        url:      YouTube URL or any yt-dlp-compatible URL.
        out_path: Desired output path for the .mp4 file.
        fps:      Target FPS to re-encode to (ensures consistent frame timing).

    Returns:
        Path to the downloaded .mp4 file.
    """
    if out_path.exists():
        print(f"[download] Video already exists at {out_path}, skipping download.")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.%(ext)s")

    print(f"[download] Downloading video from: {url}")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(tmp_path),
        url,
    ]
    subprocess.run(cmd, check=True)

    # Find the downloaded file (yt-dlp fills in %(ext)s)
    candidates = list(out_path.parent.glob("*.tmp.*"))
    if not candidates:
        raise RuntimeError("yt-dlp download failed — no output file found.")
    downloaded = candidates[0]

    # Re-encode at target FPS using ffmpeg for consistent frame timing
    print(f"[download] Re-encoding to {fps} fps → {out_path}")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(downloaded),
        "-vf", f"fps={fps}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",  # no audio
        str(out_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    downloaded.unlink(missing_ok=True)
    print(f"[download] Saved: {out_path}")
    return out_path


# ─── Step 2: Extract frames ────────────────────────────────────────────────────

def extract_frames(video_path: Path, frames_dir: Path, fps: int) -> list[Path]:
    """Extract JPEG frames from a video using ffmpeg.

    Args:
        video_path: Path to input video file.
        frames_dir: Directory to write frame_NNNNNN.jpg files.
        fps:        Frames per second to extract.

    Returns:
        Sorted list of extracted frame paths.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / "frame_%06d.jpg"

    existing = sorted(frames_dir.glob("frame_*.jpg"))
    if existing:
        print(f"[frames] Found {len(existing)} existing frames in {frames_dir}, skipping extraction.")
        return existing

    print(f"[frames] Extracting frames at {fps} fps from {video_path.name} ...")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    print(f"[frames] Extracted {len(frames)} frames → {frames_dir}")
    return frames


# ─── Step 3a: GENMO/GEM backend ───────────────────────────────────────────────

def infer_smpl_genmo(
    video_path: Path,
    frames: list[Path],
    fps: int,
    genmo_ckpt: str | None,
    cache_pkl: Path,
) -> dict:
    """Run GENMO/GEM to estimate SMPL body pose from video.

    GENMO (NVlabs/GENMO, arXiv:2505.01425) is a diffusion-based model that
    takes monocular video and outputs SMPL-X body parameters.

    Installation::

        pip install git+https://github.com/NVlabs/GENMO.git

    Args:
        video_path:  Path to the input video.
        frames:      List of extracted frame paths (used for 2D keypoint detection).
        fps:         Video FPS.
        genmo_ckpt:  Path to GENMO model checkpoint (or None to use default).
        cache_pkl:   Path to cache the raw GENMO output as a .pkl file.

    Returns:
        dict with keys:
          body_pose     (T, 63) float32
          global_orient (T, 3)  float32
          transl        (T, 3)  float32
    """
    if cache_pkl.exists():
        print(f"[genmo] Loading cached SMPL output from {cache_pkl}")
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)

    print("[genmo] Running GENMO/GEM inference ...")

    try:
        # ── Attempt GENMO import (NVlabs/GENMO) ──────────────────────────────
        # The GENMO repo exposes a high-level inference API.
        # Two common patterns observed in NVlabs research repos:
        #   Pattern A: from genmo.infer import run_inference
        #   Pattern B: model.infer(video_path) → smplx_output
        # We try both; install from https://github.com/NVlabs/GENMO

        smpl_data = _try_genmo_api(str(video_path), genmo_ckpt, fps)

    except Exception as e:
        # Broad catch: GENMO may fail with ImportError, AttributeError, RuntimeError
        # depending on installed version; fall through to HMR2 in all cases.
        print(f"[genmo] GENMO unavailable ({type(e).__name__}: {e}). "
              f"Trying 4D-Humans (HMR2) fallback...")
        smpl_data = _try_hmr2_api(frames, fps, cache_pkl.parent)

    cache_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_pkl, "wb") as f:
        pickle.dump(smpl_data, f)
    print(f"[genmo] Cached SMPL output to {cache_pkl}")
    return smpl_data


def _try_genmo_api(video_path: str, ckpt: str | None, fps: int) -> dict:
    """Try to run GENMO inference using its Python API.

    GENMO (https://github.com/NVlabs/GENMO) is a generalist human motion model
    that estimates SMPL-X parameters from monocular video using a diffusion
    framework conditioned on detected 2D keypoints.

    The exact API depends on the installed version; we try the most likely patterns.
    If GENMO is installed but the API has changed, edit this function to match
    the actual API in the installed version.
    """
    # ── Pattern A: high-level infer() function ─────────────────────────────
    try:
        from genmo.infer import run_video_inference  # type: ignore[import]
        output = run_video_inference(video_path=video_path, checkpoint=ckpt, fps=fps)
        return _parse_genmo_output(output)
    except ImportError:
        pass

    # ── Pattern B: GEM model class ────────────────────────────────────────
    try:
        from gem import GEM  # type: ignore[import]
        model = GEM.from_pretrained(ckpt) if ckpt else GEM.from_pretrained()
        output = model.infer_video(video_path, fps=fps)
        return _parse_genmo_output(output)
    except ImportError:
        pass

    # ── Pattern C: CLI-based wrapper ──────────────────────────────────────
    # Some GENMO versions are invoked as a CLI tool; wrap with subprocess
    import tempfile, json
    out_json = tempfile.mktemp(suffix=".json")
    cmd = ["python", "-m", "genmo.infer",
           "--video", video_path,
           "--output", out_json,
           "--fps", str(fps)]
    if ckpt:
        cmd += ["--checkpoint", ckpt]
    subprocess.run(cmd, check=True)
    with open(out_json) as f:
        raw = json.load(f)
    return _parse_genmo_output(raw)


def _parse_genmo_output(output: object) -> dict:
    """Normalise GENMO output to our standard dict format.

    GENMO may return a dict, a dataclass, or a Namespace depending on version.
    We handle all cases by duck-typing.
    """
    def _get(obj, *keys):
        for key in keys:
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
            elif hasattr(obj, key):
                return getattr(obj, key)
        raise KeyError(f"Could not find any of {keys} in GENMO output")

    body_pose     = np.asarray(_get(output, "body_pose",     "poses_body")).astype(np.float32)
    global_orient = np.asarray(_get(output, "global_orient", "poses_root", "root_orient")).astype(np.float32)
    transl        = np.asarray(_get(output, "transl",        "trans")).astype(np.float32)

    # Ensure shapes: body_pose → (T, 63), global_orient → (T, 3), transl → (T, 3)
    T = body_pose.shape[0]
    if body_pose.ndim == 3:
        body_pose = body_pose.reshape(T, -1)   # (T, 21, 3) → (T, 63)

    # SMPL-X body_pose has 21 joints (63 values) but some models give 23 (69 values);
    # truncate to 21 joints to match SMPL convention
    if body_pose.shape[1] == 69:
        body_pose = body_pose[:, :63]
    elif body_pose.shape[1] not in (63,):
        raise ValueError(f"Unexpected body_pose dim: {body_pose.shape[1]} (expected 63)")

    return dict(body_pose=body_pose, global_orient=global_orient, transl=transl)


# ─── Step 3b: 4D-Humans / HMR2 fallback backend ──────────────────────────────

def _try_hmr2_api(frames: list[Path], fps: int, work_dir: Path) -> dict:
    """Run 4D-Humans (HMR2) to estimate SMPL body pose per frame.

    4D-Humans is a widely-used video-to-SMPL tool based on ViT + transformer.
    Install: pip install 4d-humans
    GitHub : https://github.com/shubham-goel/4D-Humans

    Args:
        frames:   List of frame image paths (sorted in temporal order).
        fps:      Frames per second.
        work_dir: Directory to store intermediate files.

    Returns:
        dict with body_pose (T, 63), global_orient (T, 3), transl (T, 3).
    """
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT  # type: ignore[import]
    import cv2
    import torch
    import torchvision.transforms as tvT
    from scipy.spatial.transform import Rotation as R_scipy

    print(f"[hmr2] Loading HMR2 model from {DEFAULT_CHECKPOINT} ...")
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    body_poses, global_orients, transls = [], [], []

    # ImageNet mean/std in [0, 1] range (ToTensor already maps [0,255]→[0,1])
    _IMG_MEAN = [0.485, 0.456, 0.406]
    _IMG_STD  = [0.229, 0.224, 0.225]
    tfm = tvT.Compose([
        tvT.ToTensor(),
        tvT.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
    ])

    print(f"[hmr2] Processing {len(frames)} frames ...")

    # Simple frame-by-frame inference (no tracking)
    for i, frame_path in enumerate(frames):
        if i % 30 == 0:
            print(f"  Frame {i}/{len(frames)} ...")
        img = cv2.imread(str(frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_t = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model({"img": img_t})

        # body_pose: (1, 23, 3, 3) rotation matrices
        # global_orient: (1, 1, 3, 3) or (1, 3, 3) depending on HMR2 version
        bp = out["pred_smpl_params"]["body_pose"].cpu().numpy()
        go = out["pred_smpl_params"]["global_orient"].cpu().numpy()
        tr_val = out.get("pred_cam_t", out.get("pred_cam", None))
        tr = tr_val.cpu().numpy() if tr_val is not None else np.zeros((1, 3), dtype=np.float32)

        # Convert rotation matrices → axis-angle
        # bp[0]: (23, 3, 3) or similar → flatten, take first 21 joints (63 values)
        bp_mat = bp[0]  # remove batch dim: (23 or 21, 3, 3)
        bp_aa = R_scipy.from_matrix(bp_mat).as_rotvec().reshape(-1)[:63]  # (63,)

        # global_orient: remove all leading batch/singleton dims until shape is (3, 3)
        go_mat = go
        while go_mat.ndim > 2:
            go_mat = go_mat[0]   # strip leading dims: (1,1,3,3)→(1,3,3)→(3,3)
        go_aa = R_scipy.from_matrix(go_mat).as_rotvec()  # (3,)

        body_poses.append(bp_aa.astype(np.float32))
        global_orients.append(go_aa.astype(np.float32))
        transls.append(tr[0].astype(np.float32))

    return dict(
        body_pose=np.stack(body_poses, axis=0),       # (T, 63)
        global_orient=np.stack(global_orients, axis=0),  # (T, 3)
        transl=np.stack(transls, axis=0),             # (T, 3)
    )


# ─── Step 3c: GVHMR backend ──────────────────────────────────────────────────
#
# GVHMR (https://github.com/zju3dv/GVHMR) is the video estimation backbone
# that GENMO/GEM builds upon. It outputs globally-consistent SMPL-X params
# from monocular video. This backend runs the GVHMR demo script as a
# subprocess (since GVHMR uses its own conda env) then parses the .pt output.
#
# Setup (run once):
#   git clone https://github.com/zju3dv/GVHMR ~/GVHMR
#   conda create -y -n gvhmr python=3.10
#   conda activate gvhmr
#   cd ~/GVHMR && pip install -r requirements.txt && pip install -e .
#   # Download checkpoints to ~/GVHMR/inputs/checkpoints/ (see INSTALL.md)
#   # Download SMPL_NEUTRAL.pkl → inputs/checkpoints/body_models/smpl/
#   # Download SMPLX_NEUTRAL.npz → inputs/checkpoints/body_models/smplx/

_GVHMR_REPO = Path.home() / "GVHMR"
_GVHMR_DEMO = _GVHMR_REPO / "tools" / "demo" / "demo.py"
_GVHMR_CKPT_DIR = _GVHMR_REPO / "inputs" / "checkpoints"


def infer_smpl_gvhmr(video_path: Path, fps: int, cache_pkl: Path) -> dict:
    """Run GVHMR (GENMO's video estimation backbone) to estimate SMPL pose.

    Invokes GVHMR's demo script in a subprocess (using its dedicated conda env
    or the current env if GVHMR is installed), then parses the saved .pt file
    to extract body_pose, global_orient, and transl.

    Args:
        video_path: Path to the input video.
        fps:        Video FPS (informational; GVHMR reads FPS from the video).
        cache_pkl:  Path to cache the parsed output as a .pkl file.

    Returns:
        dict with body_pose (T, 63), global_orient (T, 3), transl (T, 3).
    """
    if cache_pkl.exists():
        print(f"[gvhmr] Loading cached SMPL output from {cache_pkl}")
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)

    if not _GVHMR_DEMO.exists():
        raise RuntimeError(
            f"GVHMR demo script not found at {_GVHMR_DEMO}. "
            "Clone from: git clone https://github.com/zju3dv/GVHMR ~/GVHMR"
        )

    # Output directory for GVHMR results
    gvhmr_out = cache_pkl.parent / "gvhmr_output"
    gvhmr_out.mkdir(parents=True, exist_ok=True)

    # Check if results already exist anywhere under the output dir (GVHMR
    # places results in a subdirectory named after the video stem)
    existing_pts = list(gvhmr_out.rglob("hmr4d_results.pt"))

    if not existing_pts:
        print("[gvhmr] Running GVHMR demo inference ...")
        # Try gvhmr conda env first, fall back to current Python env
        conda_envs = ["gvhmr", None]
        for env in conda_envs:
            cmd = (
                ["conda", "run", "-n", env, "--no-capture-output",
                 "python", str(_GVHMR_DEMO)]
                if env else
                ["python", str(_GVHMR_DEMO)]
            )
            cmd += [
                "--video", str(video_path),
                "--output_root", str(gvhmr_out),
                "-s",   # --static_cam: skip DPVO for speed
            ]
            try:
                print(f"[gvhmr] Trying env: {env or 'current'}")
                subprocess.run(cmd, cwd=str(_GVHMR_REPO))  # no check= : rendering may fail
            except FileNotFoundError as e:
                print(f"[gvhmr] env={env} not found: {e}")
                continue

            # Check if results were produced (may have failed only at render)
            existing_pts = list(gvhmr_out.rglob("hmr4d_results.pt"))
            if existing_pts:
                print(f"[gvhmr] hmr4d_results.pt found after env={env} run")
                break
            print(f"[gvhmr] No results after env={env}, trying next env...")

    if not existing_pts:
        raise RuntimeError(
            f"hmr4d_results.pt not found under {gvhmr_out} after all attempts. "
            "GVHMR inference failed. Check GVHMR setup."
        )

    # GVHMR may place results in outputs/demo/<video_stem>/hmr4d_results.pt
    pt_candidates = existing_pts
    results_pt = pt_candidates[0]
    print(f"[gvhmr] Parsing results from {results_pt}")

    # Parse GVHMR output ──────────────────────────────────────────────────────
    import torch
    pred = torch.load(str(results_pt), map_location="cpu")

    # Use global params (world-frame) for robot training
    smpl_params = pred.get("smpl_params_global", pred.get("smpl_params_incam"))
    if smpl_params is None:
        raise KeyError(
            f"Expected 'smpl_params_global' or 'smpl_params_incam' in "
            f"hmr4d_results.pt. Available keys: {list(pred.keys())}"
        )

    # body_pose: (T, 21, 3) or (T, 63) axis-angle
    body_pose = smpl_params["body_pose"]
    global_orient = smpl_params["global_orient"]
    transl = smpl_params["transl"]

    # Convert torch → numpy, handle rotation matrix vs axis-angle
    body_pose_np = body_pose.numpy() if hasattr(body_pose, "numpy") else np.array(body_pose)
    go_np = global_orient.numpy() if hasattr(global_orient, "numpy") else np.array(global_orient)
    tr_np = transl.numpy() if hasattr(transl, "numpy") else np.array(transl)

    # If rotation matrices (T, 21, 3, 3) → convert to axis-angle
    from scipy.spatial.transform import Rotation as _R
    if body_pose_np.ndim == 4 and body_pose_np.shape[-2:] == (3, 3):
        T, J = body_pose_np.shape[:2]
        body_pose_np = _R.from_matrix(body_pose_np.reshape(-1, 3, 3)).as_rotvec().reshape(T, J * 3)
    elif body_pose_np.ndim == 3 and body_pose_np.shape[-1] == 3:
        T, J, _ = body_pose_np.shape
        body_pose_np = body_pose_np.reshape(T, J * 3)

    if go_np.ndim == 3 and go_np.shape[-2:] == (3, 3):
        go_np = _R.from_matrix(go_np).as_rotvec()
    elif go_np.ndim == 2 and go_np.shape[-1] != 3:
        go_np = _R.from_matrix(go_np.reshape(-1, 3, 3)).as_rotvec()

    # Ensure float32 and clip body_pose to first 21 joints (63 values)
    body_pose_np = body_pose_np.astype(np.float32)[:, :63]
    go_np = go_np.astype(np.float32)
    tr_np = tr_np.astype(np.float32)

    smpl_data = dict(body_pose=body_pose_np, global_orient=go_np, transl=tr_np)
    T_frames = body_pose_np.shape[0]
    print(f"[gvhmr] Extracted {T_frames} frames of global SMPL params")

    # Cache
    cache_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_pkl, "wb") as f:
        pickle.dump(smpl_data, f)
    print(f"[gvhmr] Cached to {cache_pkl}")
    return smpl_data


# ─── Step 3 dispatcher ────────────────────────────────────────────────────────

def infer_smpl(
    backend: str,
    video_path: Path,
    frames: list[Path],
    fps: int,
    genmo_ckpt: str | None,
    work_dir: Path,
) -> dict:
    """Dispatch SMPL inference to the chosen backend.

    Args:
        backend:    One of "genmo", "hmr2", "gvhmr".
        video_path: Path to the downloaded video.
        frames:     Extracted frame paths.
        fps:        Video FPS.
        genmo_ckpt: Optional GENMO checkpoint path.
        work_dir:   Working directory for caches.

    Returns:
        dict with body_pose (T, 63), global_orient (T, 3), transl (T, 3).
    """
    cache_pkl = work_dir / f"smpl_output_{backend}.pkl"

    if backend == "genmo":
        return infer_smpl_genmo(video_path, frames, fps, genmo_ckpt, cache_pkl)
    elif backend == "hmr2":
        if cache_pkl.exists():
            with open(cache_pkl, "rb") as f:
                return pickle.load(f)
        result = _try_hmr2_api(frames, fps, work_dir)
        with open(cache_pkl, "wb") as f:
            pickle.dump(result, f)
        return result
    elif backend == "gvhmr":
        return infer_smpl_gvhmr(video_path, fps, cache_pkl)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: genmo, hmr2, gvhmr")


# ─── Step 4 + 5: Retarget and build AMP obs ───────────────────────────────────

def retarget_and_build_amp(
    smpl_data: dict,
    fps: int,
    smooth_window: int,
) -> np.ndarray:
    """Retarget SMPL pose to G1, smooth joints, and build 85-dim AMP obs.

    Args:
        smpl_data:     dict from infer_smpl — body_pose, global_orient, transl.
        fps:           Video FPS (used for velocity computation).
        smooth_window: Moving-average window for temporal smoothing of joint
                       angles (odd integer; 1 = no smoothing).

    Returns:
        (T, 85) float32 AMP observation array.
    """
    body_pose     = smpl_data["body_pose"].astype(np.float32)
    global_orient = smpl_data["global_orient"].astype(np.float32)
    transl        = smpl_data["transl"].astype(np.float32)

    T = body_pose.shape[0]
    print(f"[retarget] Input: {T} frames at {fps} fps "
          f"({T / fps:.1f} s)")

    retargeter = SMPLtoG1Retargeter(fps=float(fps))
    result = retargeter.retarget(body_pose, global_orient, transl)

    # Optional temporal smoothing of joint angles and velocities
    if smooth_window > 1:
        sw = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        print(f"[retarget] Applying temporal smoothing (window={sw}) ...")
        from scipy.ndimage import uniform_filter1d
        result["joint_pos"] = uniform_filter1d(result["joint_pos"], sw, axis=0)
        result["joint_vel"] = uniform_filter1d(result["joint_vel"], sw, axis=0)
        result["lin_vel_b"] = uniform_filter1d(result["lin_vel_b"], sw, axis=0)
        result["ang_vel_b"] = uniform_filter1d(result["ang_vel_b"], sw, axis=0)

    amp_obs = build_amp_obs_from_retarget(result)
    print(f"[retarget] AMP obs shape: {amp_obs.shape}  dim={amp_obs.shape[1]}")
    return amp_obs


# ─── Merge with analytical reference ─────────────────────────────────────────

def merge_with_analytical(
    human_amp_obs: np.ndarray,
    analytical_path: Path,
) -> np.ndarray:
    """Concatenate human reference with existing analytical reference.

    This augments the expert buffer with both human-style and robot-optimal
    motions, which can improve discriminator robustness.

    Args:
        human_amp_obs:    (T, 85) float32 from video retargeting.
        analytical_path:  Path to existing analytical skating_reference.npz.

    Returns:
        Combined (T_human + T_analytical, 85) float32 array.
    """
    if not analytical_path.exists():
        print(f"[merge] Analytical reference not found at {analytical_path}, skipping merge.")
        return human_amp_obs

    data = np.load(analytical_path)
    analytical = data["amp_obs"]
    if analytical.shape[1] != 85:
        print(f"[merge] Analytical amp_obs dim={analytical.shape[1]} != 85, skipping merge.")
        return human_amp_obs

    merged = np.concatenate([human_amp_obs, analytical], axis=0)
    print(f"[merge] Merged: {len(human_amp_obs)} human + {len(analytical)} analytical = {len(merged)} total frames")
    return merged


# ─── Save ─────────────────────────────────────────────────────────────────────

def save_amp_npz(
    amp_obs: np.ndarray,
    output_path: Path,
    fps: int,
    video_url: str,
    backend: str,
    merged: bool,
) -> None:
    """Save the AMP observation array to .npz with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    T = amp_obs.shape[0]
    meta = (
        f"Human skating AMP reference: {T} frames at {fps} fps "
        f"({T / fps:.1f}s), backend={backend}, "
        f"source={video_url}, merged_analytical={merged}"
    )
    np.savez(output_path, amp_obs=amp_obs, fps=np.float32(fps), metadata=meta)
    print(f"\n[save] Saved AMP reference to: {output_path}")
    print(f"       Frames : {T}")
    print(f"       Dim    : {amp_obs.shape[1]}")
    print(f"       FPS    : {fps}")
    print(f"       Duration: {T / fps:.1f} s")


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def print_diagnostics(amp_obs: np.ndarray, fps: int) -> None:
    """Print statistics on the generated AMP obs to catch obvious retargeting errors."""
    print("\n── AMP obs diagnostics ──────────────────────────────────────")
    joint_pos = amp_obs[:, :29]
    joint_vel = amp_obs[:, 29:58] / 0.05   # undo the 0.05 scaling
    lin_vel_b = amp_obs[:, 61:64] / 0.1
    ang_vel_b = amp_obs[:, 64:67] / 0.2

    print(f"  joint_pos  : mean={joint_pos.mean():.3f}  std={joint_pos.std():.3f}  "
          f"range=[{joint_pos.min():.3f}, {joint_pos.max():.3f}]")
    print(f"  joint_vel  : mean={joint_vel.mean():.3f}  std={joint_vel.std():.3f}  "
          f"range=[{joint_vel.min():.3f}, {joint_vel.max():.3f}]")
    print(f"  lin_vel_b  : mean={lin_vel_b.mean(axis=0)}  std={lin_vel_b.std(axis=0)}")
    print(f"  ang_vel_b  : mean={ang_vel_b.mean(axis=0)}  std={ang_vel_b.std(axis=0)}")

    fwd_speed = lin_vel_b[:, 0]  # body-frame x = forward
    print(f"  fwd speed  : mean={fwd_speed.mean():.2f}  max={fwd_speed.max():.2f} m/s")

    # Check for NaN/Inf
    n_nan = np.isnan(amp_obs).sum()
    n_inf = np.isinf(amp_obs).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  ⚠ WARNING: {n_nan} NaN, {n_inf} Inf values detected — check retargeting!")
    else:
        print("  NaN/Inf : none ✓")
    print("────────────────────────────────────────────────────────────\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retarget human skating video → GENMO SMPL → G1 AMP reference motion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_url",  type=str, help="YouTube/web URL to download the video")
    src.add_argument("--video_path", type=str, help="Path to an already-downloaded video file")

    # SMPL backend
    p.add_argument(
        "--backend", type=str, default="gvhmr",
        choices=["genmo", "hmr2", "gvhmr"],
        help="SMPL pose estimation backend (default: gvhmr — GENMO's video estimation backbone)",
    )
    p.add_argument(
        "--gvhmr_repo", type=str, default=str(Path.home() / "GVHMR"),
        help="Path to cloned GVHMR repo (used when --backend=gvhmr)",
    )
    p.add_argument(
        "--genmo_ckpt", type=str, default=None,
        help="Path to GENMO/GEM model checkpoint (optional if GENMO uses default path)",
    )

    # Processing
    p.add_argument("--fps",            type=int,   default=30,   help="Video FPS for extraction/inference")
    p.add_argument("--smooth_window",  type=int,   default=5,    help="Temporal smoothing window (1=off)")
    p.add_argument("--merge_analytical", action="store_true",    help="Also include analytical skating_reference.npz frames")

    # Paths
    p.add_argument(
        "--output", type=str, default=str(_DEFAULT_OUTPUT),
        help="Output .npz path",
    )
    p.add_argument(
        "--workdir", type=str, default=str(_DEFAULT_WORKDIR),
        help="Working directory for intermediate files (video, frames, SMPL cache)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.workdir)
    work_dir.mkdir(parents=True, exist_ok=True)

    video_url = args.video_url or "local"

    # Allow overriding GVHMR repo path at runtime via module globals
    if args.backend == "gvhmr" and hasattr(args, "gvhmr_repo"):
        global _GVHMR_REPO, _GVHMR_DEMO, _GVHMR_CKPT_DIR
        _GVHMR_REPO = Path(args.gvhmr_repo)
        _GVHMR_DEMO = _GVHMR_REPO / "tools" / "demo" / "demo.py"
        _GVHMR_CKPT_DIR = _GVHMR_REPO / "inputs" / "checkpoints"

    # ── Step 1: Get/download video ─────────────────────────────────────────
    if args.video_path:
        video_path = Path(args.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        print(f"[input] Using existing video: {video_path}")
    else:
        video_path = work_dir / "skating_video.mp4"
        video_path = download_video(args.video_url, video_path, args.fps)

    # ── Step 2: Extract frames ─────────────────────────────────────────────
    frames_dir = work_dir / "frames"
    frames = extract_frames(video_path, frames_dir, args.fps)
    if not frames:
        raise RuntimeError("No frames extracted from video.")
    print(f"[frames] Total: {len(frames)} frames  ({len(frames) / args.fps:.1f} s @ {args.fps} fps)")

    # ── Step 3: SMPL inference ─────────────────────────────────────────────
    smpl_data = infer_smpl(
        backend=args.backend,
        video_path=video_path,
        frames=frames,
        fps=args.fps,
        genmo_ckpt=args.genmo_ckpt,
        work_dir=work_dir,
    )
    T_smpl = smpl_data["body_pose"].shape[0]
    print(f"[smpl] Got {T_smpl} frames of SMPL pose data")

    # Sanity check: ensure frame counts match
    if abs(T_smpl - len(frames)) > 5:
        print(f"[smpl] ⚠ SMPL frame count ({T_smpl}) differs from extracted frames "
              f"({len(frames)}) by > 5 — SMPL backend may have dropped frames.")

    # ── Step 4 + 5: Retarget → AMP obs ────────────────────────────────────
    amp_obs = retarget_and_build_amp(smpl_data, args.fps, args.smooth_window)

    # ── Optional: Merge with analytical reference ─────────────────────────
    if args.merge_analytical:
        amp_obs = merge_with_analytical(amp_obs, _DEFAULT_ANALYTICAL)

    # ── Diagnostics ───────────────────────────────────────────────────────
    print_diagnostics(amp_obs, args.fps)

    # ── Step 6: Save ──────────────────────────────────────────────────────
    save_amp_npz(
        amp_obs, Path(args.output), args.fps,
        video_url=video_url,
        backend=args.backend,
        merged=args.merge_analytical,
    )

    print("\n✓ Done. To train AMP PPO with this reference, run:")
    print(
        "  conda run -n env_isaaclab python scripts/skating/train.py \\\n"
        "      --task Unitree-G1-Skating-AMP-v0 \\\n"
        "      --num_envs 2048 --headless \\\n"
        "      --resume_path ~/unitree_rl_lab/logs/rsl_rl/skating_phase1/"
        "2026-03-08_13-46-21/model_1999.pt \\\n"
        "      --experiment_name skating_amp --max_iterations 3000"
    )


if __name__ == "__main__":
    main()
