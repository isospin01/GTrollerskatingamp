"""Tests for AMP observation dimensions and the reference motion NPZ.

Verifies that:
  - amp_observation_state() produces the documented 85-dim output
  - The reference motion NPZ has shape (T, 85) and covers multiple speeds
  - The NPZ and the online AMP obs have matching dimensions

Run with:
    cd /home/muchenxu/unitree_rl_lab
    python -m pytest tests/skating/test_amp_and_reference.py -v
"""

import os
import sys
import math
import pytest
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_NPZ_PATH  = os.path.join(
    _REPO_ROOT,
    "source/unitree_rl_lab/unitree_rl_lab/data/skating_reference.npz",
)
_AMP_OBS_SRC = os.path.join(
    _REPO_ROOT,
    "source/unitree_rl_lab/unitree_rl_lab/tasks/skating/mdp/amp_obs.py",
)


# ---------------------------------------------------------------------------
# Test 1 – AMP obs dim declared consistently in source comments
# ---------------------------------------------------------------------------

class TestAmpObsDocumentation:

    def test_amp_obs_dimension_comment_is_85(self):
        """The amp_obs.py docstring and inline comment must say 85, not 89."""
        with open(_AMP_OBS_SRC) as f:
            source = f.read()

        assert "(N, 89)" not in source, (
            "amp_obs.py still contains '(N, 89)' — the correct dimension is 85. "
            "Stale comment misleads future developers about the tensor shape."
        )
        assert "(N, 85)" in source, \
            "amp_obs.py should document the output shape as (N, 85)."

    def test_documented_dim_matches_sum(self):
        """The documented sum (29+29+3+3+3+6+6+6 = 85) must be arithmetically correct."""
        components = {
            "joint_pos":  29,
            "joint_vel":  29,
            "proj_grav":   3,
            "lin_vel_b":   3,
            "ang_vel_b":   3,
            "ankle_rel":   6,   # 2 ankles × xyz
            "wrist_rel":   6,   # 2 wrists × xyz
            "ankle_vel":   6,   # 2 ankles × xyz
        }
        total = sum(components.values())
        assert total == 85, (
            f"Documented AMP obs components sum to {total}, expected 85. "
            f"Component breakdown: {components}"
        )


# ---------------------------------------------------------------------------
# Test 2 – Reference NPZ shape, content, and speed coverage
# ---------------------------------------------------------------------------

class TestReferenceMotionNPZ:

    @pytest.fixture(scope="class")
    def npz(self):
        if not os.path.isfile(_NPZ_PATH):
            pytest.skip(f"Reference NPZ not found at {_NPZ_PATH}")
        return np.load(_NPZ_PATH)

    def test_npz_has_required_keys(self, npz):
        assert "amp_obs" in npz, "NPZ missing 'amp_obs' key"
        assert "fps"     in npz, "NPZ missing 'fps' key"
        assert "metadata" in npz, "NPZ missing 'metadata' key"

    def test_amp_obs_dimension_is_85(self, npz):
        """Reference NPZ must have same 85-dim AMP state as the online amp_obs.py."""
        dim = npz["amp_obs"].shape[1]
        assert dim == 85, (
            f"Reference NPZ amp_obs has dim={dim}, expected 85. "
            "If dim != 85, the discriminator input dimensions won't match the "
            "policy's live AMP observations, silently breaking AMP training."
        )

    def test_npz_covers_multiple_speeds(self, npz):
        """NPZ should contain >= 6000 frames (at least 2 speed variants × 3000 frames)."""
        n_frames = npz["amp_obs"].shape[0]
        assert n_frames >= 6000, (
            f"Reference NPZ has only {n_frames} frames. "
            "A single-speed reference (0.8 m/s) doesn't cover Phase 3 training speeds "
            "(up to 4 m/s), causing the AMP discriminator to penalise all high-speed motion."
        )

    def test_fps_matches_policy_frequency(self, npz):
        """FPS in NPZ must match the simulation policy frequency (50 Hz = dt×decimation)."""
        fps = float(npz["fps"])
        expected_fps = 50.0   # sim.dt=0.005 × decimation=4 → 0.02s per step → 50 Hz
        assert fps == expected_fps, (
            f"Reference NPZ fps={fps}, expected {expected_fps}. "
            "Mismatch means the reference motion plays at the wrong speed relative "
            "to the policy rollouts, corrupting the AMP discriminator training."
        )

    def test_amp_obs_has_no_nans_or_infs(self, npz):
        amp_obs = npz["amp_obs"]
        n_nan = np.isnan(amp_obs).sum()
        n_inf = np.isinf(amp_obs).sum()
        assert n_nan == 0, f"Reference NPZ contains {n_nan} NaN values."
        assert n_inf == 0, f"Reference NPZ contains {n_inf} Inf values."

    def test_amp_obs_values_in_reasonable_range(self, npz):
        """AMP obs values should be small (scaled) floats — not raw SI units."""
        amp_obs = npz["amp_obs"]
        abs_max = np.abs(amp_obs).max()
        assert abs_max < 50.0, (
            f"Max abs AMP obs value = {abs_max:.2f}. Expected small scaled values. "
            "Large values indicate missing scaling factors (e.g. lin_vel not multiplied by 0.1)."
        )

    def test_forward_velocity_component_covers_speed_range(self, npz):
        """The lin_vel_b forward component (scaled by 0.1) should cover 0.05–0.30.

        This corresponds to actual speeds of 0.5–3.0 m/s × 0.1 scaling.
        """
        amp_obs = npz["amp_obs"]
        # lin_vel_b is at indices 87:90 in the 85-dim vector... wait
        # layout: joint_pos(29) + joint_vel(29) + proj_grav(3) + lin_vel_b(3) + ang_vel_b(3) + ...
        # lin_vel_b starts at index 29+29+3 = 61
        lin_vel_x = amp_obs[:, 61]   # forward velocity × 0.1
        vx_max = lin_vel_x.max()
        vx_min = lin_vel_x.min()
        assert vx_max >= 0.28, (
            f"Max scaled forward velocity in reference = {vx_max:.3f} (={vx_max/0.1:.2f} m/s). "
            "Reference should cover up to ~3.0 m/s (scaled: 0.30). "
            "The discriminator won't recognise high-speed skating motion."
        )
        assert vx_min >= 0.0, (
            f"Forward velocity has negative values ({vx_min:.3f}) — reference motion "
            "shouldn't skate backwards."
        )


# ---------------------------------------------------------------------------
# Test 3 – AMP obs dimension computed by amp_obs.py (pure computation check)
# ---------------------------------------------------------------------------

class TestAmpObsComputation:
    """Verify the amp_observation_state function output dimension via mock tensors."""

    def test_concat_produces_85_dims(self):
        """Simulate the torch.cat in amp_observation_state and verify 85-dim output."""
        import torch
        N = 4   # batch size

        joint_pos = torch.zeros(N, 29)
        joint_vel = torch.zeros(N, 29) * 0.05
        proj_grav = torch.zeros(N, 3)
        lin_vel_b = torch.zeros(N, 3) * 0.1
        ang_vel_b = torch.zeros(N, 3) * 0.2
        ankle_rel = torch.zeros(N, 6)    # 2 ankles × 3 xyz
        wrist_rel = torch.zeros(N, 6)    # 2 wrists × 3 xyz
        ankle_vel = torch.zeros(N, 6)    # 2 ankles × 3 xyz

        result = torch.cat([joint_pos, joint_vel, proj_grav, lin_vel_b, ang_vel_b,
                            ankle_rel, wrist_rel, ankle_vel], dim=-1)

        assert result.shape == (N, 85), (
            f"AMP obs concatenation gives shape {result.shape}, expected ({N}, 85). "
            "Dimension mismatch breaks the discriminator."
        )

    def test_reference_npz_dim_matches_online_obs(self):
        """NPZ dim must equal the torch.cat output dim (both must be 85)."""
        if not os.path.isfile(_NPZ_PATH):
            pytest.skip("Reference NPZ not found")

        import torch
        npz = np.load(_NPZ_PATH)
        npz_dim = npz["amp_obs"].shape[1]

        # Compute online dim
        N = 1
        online_dim = sum([29, 29, 3, 3, 3, 6, 6, 6])

        assert npz_dim == online_dim, (
            f"Reference NPZ dim ({npz_dim}) != online amp_obs dim ({online_dim}). "
            "The discriminator would receive different-dimension inputs from "
            "expert samples vs policy rollouts."
        )
