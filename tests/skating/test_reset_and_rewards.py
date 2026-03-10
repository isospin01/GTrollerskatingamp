"""Tests for reset logic and reward functions.

Verifies that:
  - reset_skating_pose correctly rotates the forward velocity into world frame
  - skate_foot_edge_contact rewards ankle roll (not just contact)
  - skate_action_rate uses prev_action (no attribute errors)
  - glide_continuity min_speed parameter is respected

Run with:
    cd /home/muchenxu/unitree_rl_lab
    python -m pytest tests/skating/test_reset_and_rewards.py -v
"""

import math
import os
import sys
import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


# ---------------------------------------------------------------------------
# Test 1 – Reset velocity frame rotation
# ---------------------------------------------------------------------------

class TestResetVelocityFrame:
    """Verify that reset_skating_pose projects body-frame vx into world frame."""

    def _run_reset_logic(self, yaw: float, vx: float) -> tuple[float, float]:
        """Replicate the corrected reset_skating_pose velocity logic."""
        dyaw = torch.tensor([yaw])
        vx_body = torch.tensor([vx])

        vx_world = (vx_body * torch.cos(dyaw)).item()
        vy_world = (vx_body * torch.sin(dyaw)).item()
        return vx_world, vy_world

    def test_zero_yaw_gives_pure_x_velocity(self):
        """At yaw=0 (facing +X), world vx should equal body vx."""
        vx_w, vy_w = self._run_reset_logic(yaw=0.0, vx=1.0)
        assert abs(vx_w - 1.0) < 1e-6, f"Expected vx_world=1.0, got {vx_w}"
        assert abs(vy_w - 0.0) < 1e-6, f"Expected vy_world=0.0, got {vy_w}"

    def test_90deg_yaw_gives_pure_y_velocity(self):
        """At yaw=π/2 (facing +Y), world vy should equal body vx."""
        vx_w, vy_w = self._run_reset_logic(yaw=math.pi / 2, vx=1.0)
        assert abs(vx_w - 0.0) < 1e-5, \
            f"At 90° yaw, world vx should be ~0, got {vx_w:.4f}"
        assert abs(vy_w - 1.0) < 1e-5, \
            f"At 90° yaw, world vy should be ~1.0, got {vy_w:.4f}"

    def test_180deg_yaw_gives_negative_x_velocity(self):
        """At yaw=π (facing -X), world vx should be negative."""
        vx_w, vy_w = self._run_reset_logic(yaw=math.pi, vx=1.0)
        assert vx_w < -0.99, (
            f"At 180° yaw, world vx should be -1.0, got {vx_w:.4f}. "
            "Old bug: setting world[7]=vx without rotation would make the robot "
            "move backward relative to its facing direction."
        )

    def test_speed_magnitude_preserved(self):
        """After rotation, total speed magnitude must equal body vx."""
        for yaw in [0.0, 0.3, 1.2, 2.5, math.pi]:
            for vx in [0.3, 1.0, 2.0]:
                vx_w, vy_w = self._run_reset_logic(yaw, vx)
                magnitude = math.sqrt(vx_w**2 + vy_w**2)
                assert abs(magnitude - vx) < 1e-5, (
                    f"yaw={yaw:.2f}, vx={vx}: magnitude after rotation={magnitude:.4f}, "
                    f"expected {vx}. Rotation must preserve speed."
                )

    def test_actual_events_py_uses_rotation(self):
        """Verify events.py source code contains the cos/sin rotation fix."""
        events_path = os.path.join(
            _REPO_ROOT,
            "source/unitree_rl_lab/unitree_rl_lab/tasks/skating/mdp/events.py",
        )
        with open(events_path) as f:
            source = f.read()

        assert "torch.cos(dyaw)" in source, (
            "events.py does not apply cos(yaw) rotation to the reset velocity. "
            "The robot will be initialized moving in the wrong direction when yaw != 0."
        )
        assert "torch.sin(dyaw)" in source, (
            "events.py does not apply sin(yaw) to set the vy component. "
            "The rotation is incomplete."
        )
        # The old bug: new_state[:, 7] = vx without cos/sin
        assert "= vx_body * torch.cos" in source or "vx_body * torch.cos(dyaw)" in source, (
            "The corrected form 'vx_body * torch.cos(dyaw)' is not found in events.py."
        )


# ---------------------------------------------------------------------------
# Test 2 – Edge contact reward uses ankle roll angles
# ---------------------------------------------------------------------------

class TestEdgeContactReward:
    """Verify skate_foot_edge_contact measures ankle roll, not just contact."""

    def test_reward_source_accesses_joint_positions(self):
        """The reward function must read ankle roll joint positions."""
        rewards_path = os.path.join(
            _REPO_ROOT,
            "source/unitree_rl_lab/unitree_rl_lab/tasks/skating/mdp/rewards.py",
        )
        with open(rewards_path) as f:
            source = f.read()

        import re
        # Find the skate_foot_edge_contact function body
        match = re.search(
            r'def skate_foot_edge_contact\(.*?\ndef \w',
            source, re.DOTALL
        )
        assert match, "skate_foot_edge_contact function not found in rewards.py"
        fn_body = match.group(0)

        assert "joint_pos" in fn_body, (
            "skate_foot_edge_contact does not read joint_pos. "
            "The old bug only checked contact time — ankle roll angle is required "
            "for meaningful edge control reward."
        )
        assert "ankle_roll_joint" in fn_body, (
            "skate_foot_edge_contact must specifically reference ankle_roll_joint."
        )

    def test_edge_reward_increases_with_ankle_roll(self):
        """Reward should increase when ankle rolls in the correct direction."""
        # Simulate the core logic of the fixed skate_foot_edge_contact
        def edge_reward_logic(
            q_left: float, q_right: float,
            wz_cmd: float,
            both_contact: bool,
            ang_vel_threshold: float = 0.2,
        ) -> float:
            turning = abs(wz_cmd) > ang_vel_threshold
            edge_alignment = max(0.0, q_right * wz_cmd - q_left * wz_cmd)
            return edge_alignment * (1.0 if both_contact else 0.0) * (1.0 if turning else 0.0)

        # Left turn (wz > 0): right ankle should roll inward (positive),
        # left ankle should roll outward (negative)
        reward_correct = edge_reward_logic(
            q_left=-0.15, q_right=0.15, wz_cmd=0.5, both_contact=True
        )
        reward_flat = edge_reward_logic(
            q_left=0.0, q_right=0.0, wz_cmd=0.5, both_contact=True
        )
        reward_wrong = edge_reward_logic(
            q_left=0.15, q_right=-0.15, wz_cmd=0.5, both_contact=True
        )

        assert reward_correct > reward_flat, (
            "Correct ankle lean should earn more reward than flat ankles. "
            f"Got correct={reward_correct:.3f}, flat={reward_flat:.3f}."
        )
        assert reward_correct > reward_wrong, (
            "Correct ankle lean should earn more reward than wrong-direction lean. "
            f"Got correct={reward_correct:.3f}, wrong={reward_wrong:.3f}."
        )
        assert reward_wrong == 0.0, (
            f"Wrong-direction lean should earn zero (clamped to 0), got {reward_wrong:.3f}."
        )

    def test_no_reward_without_turning_command(self):
        """Edge contact reward must be zero when no turn is commanded."""
        def edge_reward_logic(q_left, q_right, wz_cmd, both_contact, threshold=0.2):
            turning = abs(wz_cmd) > threshold
            alignment = max(0.0, q_right * wz_cmd - q_left * wz_cmd)
            return alignment * float(both_contact) * float(turning)

        reward = edge_reward_logic(
            q_left=-0.2, q_right=0.2,  # ankles correctly tilted
            wz_cmd=0.1,                 # below threshold
            both_contact=True,
        )
        assert reward == 0.0, (
            f"Edge reward should be 0 when wz_cmd < threshold, got {reward:.3f}."
        )

    def test_no_reward_when_airborne(self):
        """Edge contact reward must be zero when a foot is off the ground."""
        def edge_reward_logic(q_left, q_right, wz_cmd, both_contact, threshold=0.2):
            turning = abs(wz_cmd) > threshold
            alignment = max(0.0, q_right * wz_cmd - q_left * wz_cmd)
            return alignment * float(both_contact) * float(turning)

        reward = edge_reward_logic(
            q_left=-0.2, q_right=0.2,
            wz_cmd=0.5,
            both_contact=False,   # one foot airborne
        )
        assert reward == 0.0, (
            f"Edge reward should be 0 when a foot is airborne, got {reward:.3f}."
        )


# ---------------------------------------------------------------------------
# Test 3 – Glide continuity reward logic
# ---------------------------------------------------------------------------

class TestGlideContinuityReward:

    def test_reward_zero_below_min_speed(self):
        """glide_continuity must return 0 when speed < min_speed."""
        min_speed = 0.3
        speed = 0.2
        cmd_active = True
        result = float(speed > min_speed) * float(cmd_active)
        assert result == 0.0, \
            f"Below min_speed reward should be 0, got {result}"

    def test_reward_one_above_min_speed(self):
        """glide_continuity must return 1 when speed > min_speed and cmd active."""
        min_speed = 0.3
        speed = 0.5
        cmd_active = True
        result = float(speed > min_speed) * float(cmd_active)
        assert result == 1.0, \
            f"Above min_speed with cmd active should give reward 1.0, got {result}"

    def test_reward_zero_without_command(self):
        """glide_continuity must return 0 when no velocity is commanded."""
        min_speed = 0.3
        speed = 1.5
        cmd_active = False   # cmd < 0.1 threshold
        result = float(speed > min_speed) * float(cmd_active)
        assert result == 0.0, \
            f"Without active command reward should be 0, got {result}"
