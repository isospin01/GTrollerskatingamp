"""Tests for configuration consistency across the skating task.

Verifies that all phase configurations are internally consistent:
  - Phase clock periods match between observations and rewards
  - CriticCfg has concatenate_terms=True
  - Phase 2 reset velocity is compatible with glide_continuity threshold
  - Init height accounts for wheel geometry
  - Target height in the height reward is consistent with init height
  - Dead code removed

Run with:
    cd /home/muchenxu/unitree_rl_lab
    python -m pytest tests/skating/test_config_consistency.py -v
"""

import ast
import math
import os
import sys
import pytest

# Add source to path so we can import without installing
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_SRC = os.path.join(_REPO_ROOT, "source/unitree_rl_lab")
sys.path.insert(0, _SRC)

ENV_CFG_PATH = os.path.join(
    _REPO_ROOT,
    "source/unitree_rl_lab/unitree_rl_lab/tasks/skating/robots/g1_29dof/skating_env_cfg.py",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_source(path: str) -> str:
    with open(path) as f:
        return f.read()


def _find_all_period_values_in_source(source: str) -> list[str]:
    """Return all "period": <value> assignments found in the source text."""
    import re
    return re.findall(r'"period"\s*:\s*([\d.]+)', source)


# ── Test 1: Phase clock period consistency ────────────────────────────────────

class TestPhaseClockPeriod:

    def test_observation_phase_periods_are_1_2(self):
        """Both PolicyCfg and CriticCfg skating_phase_signal must use period=1.2."""
        source = _read_source(ENV_CFG_PATH)
        import re
        # Find all skating_phase_signal ObsTerm definitions with their period param
        blocks = re.findall(
            r'skating_phase\s*=\s*ObsTerm\(.*?params\s*=\s*\{.*?"period"\s*:\s*([\d.]+)',
            source, re.DOTALL
        )
        assert len(blocks) >= 2, \
            "Expected at least 2 skating_phase ObsTerms (policy + critic), found fewer."
        for period_str in blocks:
            period = float(period_str)
            assert period == 1.2, (
                f"skating_phase_signal period is {period} but must be 1.2. "
                "Policy sees a 1.2s clock and must be graded by the same clock."
            )

    def test_push_off_reward_period_matches_obs_period(self):
        """push_off_rhythm reward period must equal the observation skating_phase period."""
        source = _read_source(ENV_CFG_PATH)
        import re

        obs_periods = re.findall(
            r'skating_phase\s*=\s*ObsTerm\(.*?params\s*=\s*\{.*?"period"\s*:\s*([\d.]+)',
            source, re.DOTALL
        )
        reward_periods = re.findall(
            r'push_off\s*=\s*RewTerm\(.*?params\s*=\s*\{.*?"period"\s*:\s*([\d.]+)',
            source, re.DOTALL
        )

        assert obs_periods, "No skating_phase ObsTerm period found"
        assert reward_periods, "No push_off RewTerm period found"

        obs_period    = float(obs_periods[0])
        reward_period = float(reward_periods[0])

        assert obs_period == reward_period, (
            f"Observation phase period ({obs_period}) != reward phase period ({reward_period}). "
            "These must match or the policy can never be in-phase with the reward."
        )

    def test_no_mismatched_period_values(self):
        """All period values in the env cfg must be either 1.2 (gait) or sim.dt related."""
        source = _read_source(ENV_CFG_PATH)
        all_periods = _find_all_period_values_in_source(source)
        gait_periods = [p for p in all_periods if float(p) != 0.005]  # exclude sim.dt
        bad = [p for p in gait_periods if float(p) != 1.2]
        assert not bad, (
            f"Found unexpected gait period values: {bad}. All gait periods should be 1.2."
        )


# ── Test 2: CriticCfg concatenate_terms ──────────────────────────────────────

class TestCriticCfgConcatenation:

    def test_critic_cfg_has_concatenate_terms_true(self):
        """CriticCfg.__post_init__ must set concatenate_terms=True."""
        source = _read_source(ENV_CFG_PATH)
        import re

        # Find the CriticCfg class definition and its __post_init__
        # Look for the pattern within what should be the CriticCfg class
        critic_block = re.search(
            r'class CriticCfg\(ObsGroup\):(.*?)class AmpCfg',
            source, re.DOTALL
        )
        assert critic_block, "CriticCfg class not found in skating_env_cfg.py"

        block_text = critic_block.group(1)
        assert "concatenate_terms = True" in block_text, (
            "CriticCfg.__post_init__ is missing 'self.concatenate_terms = True'. "
            "Without this, the critic observation group may return a dict instead of "
            "a flat tensor, causing PPO to receive wrong observation shapes."
        )

    def test_policy_cfg_has_concatenate_terms_true(self):
        """PolicyCfg.__post_init__ must also set concatenate_terms=True (sanity check)."""
        source = _read_source(ENV_CFG_PATH)
        import re
        policy_block = re.search(
            r'class PolicyCfg\(ObsGroup\):(.*?)class CriticCfg',
            source, re.DOTALL
        )
        assert policy_block, "PolicyCfg class not found"
        assert "concatenate_terms = True" in policy_block.group(1), \
            "PolicyCfg missing concatenate_terms = True"


# ── Test 3: Phase 2 reset velocity vs glide_continuity ───────────────────────

class TestPhase2ResetGlideCompatibility:

    def test_phase2_min_speed_reachable_from_reset_velocity(self):
        """Phase 2 glide_continuity min_speed must be <= Phase 2 reset max velocity.

        The glide reward must be reachable within the first push cycle
        to provide learning signal from episode start.
        """
        source = _read_source(ENV_CFG_PATH)
        import re

        # Find Phase2RewardsCfg.glide_continuity min_speed
        # It's defined inside Phase2RewardsCfg so look after that class marker
        phase2_rewards_match = re.search(
            r'class Phase2RewardsCfg.*?glide_continuity\s*=\s*RewTerm\(.*?'
            r'"min_speed"\s*:\s*([\d.]+)',
            source, re.DOTALL
        )
        assert phase2_rewards_match, \
            "Phase2RewardsCfg.glide_continuity with min_speed not found"
        min_speed = float(phase2_rewards_match.group(1))

        # Find Phase2EventCfg.reset_base init_velocity_range x max
        phase2_event_match = re.search(
            r'class Phase2EventCfg.*?init_velocity_range.*?"x"\s*:\s*\(([\d.]+)\s*,\s*([\d.]+)\)',
            source, re.DOTALL
        )
        assert phase2_event_match, \
            "Phase2EventCfg reset_base init_velocity_range not found"
        reset_max_vx = float(phase2_event_match.group(2))

        assert min_speed <= reset_max_vx + 0.3, (
            f"Phase 2 glide_continuity min_speed={min_speed} m/s is too far above "
            f"the reset max velocity={reset_max_vx} m/s. "
            "The robot would need to push off significantly before earning any glide reward, "
            "reducing the learning signal at episode start."
        )

    def test_phase2_min_speed_lower_than_phase1(self):
        """Phase 2 glide min_speed must be <= Phase 1's to allow meaningful training from rest."""
        source = _read_source(ENV_CFG_PATH)
        import re

        # Phase 1 glide_continuity (in Phase1RewardsCfg)
        phase1_match = re.search(
            r'class Phase1RewardsCfg.*?glide_continuity\s*=\s*RewTerm\(.*?'
            r'"min_speed"\s*:\s*([\d.]+)',
            source, re.DOTALL
        )
        assert phase1_match, "Phase1RewardsCfg.glide_continuity not found"
        phase1_min = float(phase1_match.group(1))

        phase2_match = re.search(
            r'class Phase2RewardsCfg.*?glide_continuity\s*=\s*RewTerm\(.*?'
            r'"min_speed"\s*:\s*([\d.]+)',
            source, re.DOTALL
        )
        assert phase2_match, "Phase2RewardsCfg.glide_continuity override not found"
        phase2_min = float(phase2_match.group(1))

        assert phase2_min <= phase1_min, (
            f"Phase 2 glide min_speed ({phase2_min}) should be <= Phase 1 ({phase1_min}). "
            "Phase 2 starts from rest and must earn the glide reward by pushing off."
        )


# ── Test 4: Init height and target height geometry correctness ────────────────

class TestHeightGeometryConsistency:
    """Verify the init height accounts for the wheel geometry offset."""

    # Constants from skate_attachment.py
    WHEEL_RADIUS = 0.038
    SPHERE_FOOT_BOTTOM_BELOW_ANKLE = 0.050  # from code comment
    WHEEL_CENTER_BELOW_ANKLE = SPHERE_FOOT_BOTTOM_BELOW_ANKLE + WHEEL_RADIUS  # 0.088m
    WHEEL_BOTTOM_BELOW_ANKLE = WHEEL_CENTER_BELOW_ANKLE + WHEEL_RADIUS        # 0.126m
    EXTRA_HEIGHT_NEEDED = WHEEL_BOTTOM_BELOW_ANKLE - SPHERE_FOOT_BOTTOM_BELOW_ANKLE  # 0.076m
    G1_BASE_HEIGHT = 0.80  # standard G1 root height with sphere feet on floor
    MIN_CORRECT_INIT_HEIGHT = G1_BASE_HEIGHT + EXTRA_HEIGHT_NEEDED  # 0.876m

    def _get_init_height(self) -> float:
        unitree_path = os.path.join(
            _REPO_ROOT,
            "source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py",
        )
        source = _read_source(unitree_path)
        import re
        # Find UNITREE_G1_29DOF_SKATE_CFG init_state pos z
        match = re.search(
            r'UNITREE_G1_29DOF_SKATE_CFG.*?pos\s*=\s*\([^)]*?([\d.]+)\s*\)',
            source, re.DOTALL
        )
        if not match:
            pytest.fail("Could not find UNITREE_G1_29DOF_SKATE_CFG init_state pos in unitree.py")
        return float(match.group(1))

    def test_init_height_accounts_for_wheel_geometry(self):
        """Init root height must be >= G1_base + (wheel_bottom - sphere_foot_bottom)."""
        init_height = self._get_init_height()
        assert init_height >= self.MIN_CORRECT_INIT_HEIGHT - 0.005, (
            f"Init height {init_height:.3f} m is too low. "
            f"Wheels extend {self.WHEEL_BOTTOM_BELOW_ANKLE:.3f} m below ankle vs "
            f"{self.SPHERE_FOOT_BOTTOM_BELOW_ANKLE:.3f} m for sphere feet, requiring "
            f">= {self.MIN_CORRECT_INIT_HEIGHT:.3f} m root height to avoid wheel-ground penetration. "
            "Penetration causes PhysX to violently push the robot at episode start."
        )

    def test_target_height_consistent_with_init_height(self):
        """The base_height reward target must be at or below the init height."""
        source = _read_source(ENV_CFG_PATH)
        import re
        target_match = re.search(
            r'base_height_skating_l2.*?"target_height"\s*:\s*([\d.]+)',
            source, re.DOTALL
        )
        assert target_match, "base_height_skating_l2 target_height not found"
        target = float(target_match.group(1))

        init_height = self._get_init_height()
        assert target <= init_height + 0.01, (
            f"target_height ({target:.3f}) must be <= init_height ({init_height:.3f}). "
            "A target above init height means the robot is immediately penalised at spawn."
        )
        assert target >= self.MIN_CORRECT_INIT_HEIGHT - 0.08, (
            f"target_height ({target:.3f}) m is too low. With skates, the robot root should "
            f"be near {self.MIN_CORRECT_INIT_HEIGHT:.3f} m when standing on wheels."
        )


# ── Test 5: No dead _REFERENCE_MOTION_NPZ in env cfg ─────────────────────────

class TestNoDeadCode:

    def test_no_unused_reference_motion_npz_in_env_cfg(self):
        """The dead _REFERENCE_MOTION_NPZ variable must be removed from skating_env_cfg.py."""
        source = _read_source(ENV_CFG_PATH)
        assert "_REFERENCE_MOTION_NPZ" not in source, (
            "_REFERENCE_MOTION_NPZ is defined in skating_env_cfg.py but never used. "
            "It was dead code — the runner reads the path from rsl_rl_ppo_cfg.py instead."
        )
