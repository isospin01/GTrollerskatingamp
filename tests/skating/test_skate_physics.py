"""Tests for roller-skate physics geometry attachment.

These tests verify that when ``attach_skates_to_robot`` runs on a real USD stage,
it creates collision geometry with the correct physical properties.  Tests that
require Isaac Sim / Omniverse (``omni.usd``, ``pxr``) are automatically skipped
when those packages are not available (e.g. in a plain Python environment).

Run inside the env_isaaclab conda environment for full coverage:
    conda activate env_isaaclab
    cd /home/muchenxu/unitree_rl_lab
    python -m pytest tests/skating/test_skate_physics.py -v
"""

import math
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pxr_available() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except ImportError:
        return False


def _omni_available() -> bool:
    try:
        import omni.usd  # noqa: F401
        return True
    except ImportError:
        return False


requires_pxr  = pytest.mark.skipif(not _pxr_available(),  reason="pxr (USD) not installed")
requires_omni = pytest.mark.skipif(not _omni_available(), reason="omni.usd not available (requires Isaac Sim)")


# ---------------------------------------------------------------------------
# Test 1 – Wheel geometry constants are self-consistent
# ---------------------------------------------------------------------------

class TestWheelGeometryConstants:
    """Pure-Python tests: no USD/Omniverse needed."""

    def _import_constants(self):
        import importlib.util, sys, os
        spec = importlib.util.spec_from_file_location(
            "skate_attachment",
            os.path.join(
                os.path.dirname(__file__),
                "../../source/unitree_rl_lab/unitree_rl_lab/tasks/skating/mdp/skate_attachment.py",
            ),
        )
        # The module imports omni/pxr only inside functions — safe to load
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_wheel_bottom_below_sphere_foot(self):
        """Wheel bottom must extend BELOW the sphere foot so wheels are the primary contact."""
        mod = self._import_constants()
        sphere_foot_bottom_below_ankle = 0.050   # from code comment in skate_attachment.py
        wheel_bottom_below_ankle = mod._WHEEL_CENTER_BELOW_ANKLE_APPROX() + mod._WHEEL_RADIUS

        assert wheel_bottom_below_ankle > sphere_foot_bottom_below_ankle, (
            f"Wheel bottom ({wheel_bottom_below_ankle:.3f} m below ankle) must be lower than "
            f"sphere foot ({sphere_foot_bottom_below_ankle:.3f} m below ankle). "
            "If not, the robot will still contact the floor via its bare sphere feet."
        )

    def test_collision_cylinder_positions_match_visual_wheels(self):
        """Collision cylinder X positions should be centred between the visual wheel pairs."""
        mod = self._import_constants()
        wx = mod._WHEEL_X
        expected_front_x = (wx[0] + wx[1]) / 2
        expected_back_x  = (wx[2] + wx[3]) / 2
        assert abs(mod._COL_FRONT_X - expected_front_x) < 1e-6
        assert abs(mod._COL_BACK_X  - expected_back_x)  < 1e-6

    def test_collision_cylinder_height_covers_both_wheels(self):
        """Collision cylinder height must span at least two wheel widths (+ gap)."""
        mod = self._import_constants()
        wx = mod._WHEEL_X
        wheel_pair_span = abs(wx[0] - wx[1])
        assert mod._COL_HEIGHT >= wheel_pair_span, (
            f"Collision cylinder height ({mod._COL_HEIGHT:.4f} m) should span the "
            f"front wheel pair ({wheel_pair_span:.4f} m)."
        )


# ---------------------------------------------------------------------------
# Test 2 – Physics material has correct friction values
# ---------------------------------------------------------------------------

@requires_pxr
class TestWheelPhysicsMaterial:

    def _make_in_memory_stage(self):
        from pxr import Usd
        return Usd.Stage.CreateInMemory()

    def test_material_has_low_static_friction(self):
        from pxr import UsdPhysics
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                        "../../source/unitree_rl_lab"))
        from unitree_rl_lab.tasks.skating.mdp.skate_attachment import _ensure_wheel_physics_material

        stage = self._make_in_memory_stage()
        mat_path = "/Test/WheelMat"
        _ensure_wheel_physics_material(stage, mat_path)

        prim = stage.GetPrimAtPath(mat_path)
        assert prim.IsValid(), "Physics material prim was not created"

        physics_mat = UsdPhysics.MaterialAPI(prim)
        static_friction = physics_mat.GetStaticFrictionAttr().Get()
        dynamic_friction = physics_mat.GetDynamicFrictionAttr().Get()

        assert static_friction <= 0.05, (
            f"Wheel static friction {static_friction} is too high for skating "
            f"(should be <= 0.05, got {static_friction}). "
            "High friction will prevent gliding."
        )
        assert dynamic_friction <= 0.03, (
            f"Wheel dynamic friction {dynamic_friction} is too high (should be <= 0.03)."
        )
        assert dynamic_friction <= static_friction, (
            "Dynamic friction must not exceed static friction."
        )

    def test_material_uses_min_combine_mode(self):
        """'min' combine mode ensures wheel friction dominates over the rink surface."""
        try:
            from pxr import PhysxSchema
        except ImportError:
            pytest.skip("PhysxSchema not available")

        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                        "../../source/unitree_rl_lab"))
        from unitree_rl_lab.tasks.skating.mdp.skate_attachment import _ensure_wheel_physics_material

        stage = self._make_in_memory_stage()
        mat_path = "/Test/WheelMatCombine"
        _ensure_wheel_physics_material(stage, mat_path)

        prim = stage.GetPrimAtPath(mat_path)
        physx_mat = PhysxSchema.PhysxMaterialAPI(prim)
        combine_mode = physx_mat.GetFrictionCombineModeAttr().Get()
        assert combine_mode == "min", (
            f"Friction combine mode should be 'min' so wheel friction dominates, got '{combine_mode}'."
        )

    def test_collision_api_applied_to_cylinder(self):
        """_apply_collision_and_material must apply CollisionAPI to geometry."""
        from pxr import UsdGeom, UsdPhysics, Gf
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                        "../../source/unitree_rl_lab"))
        from unitree_rl_lab.tasks.skating.mdp.skate_attachment import (
            _ensure_wheel_physics_material, _apply_collision_and_material,
        )

        stage = self._make_in_memory_stage()
        mat_path = "/Test/WheelMat2"
        _ensure_wheel_physics_material(stage, mat_path)

        cyl = UsdGeom.Cylinder.Define(stage, "/Test/Cylinder")
        _apply_collision_and_material(cyl.GetPrim(), stage, mat_path)

        assert UsdPhysics.CollisionAPI(cyl.GetPrim()), (
            "UsdPhysics.CollisionAPI was not applied to the collision cylinder."
        )

    def test_full_geometry_creates_collision_root(self):
        """_add_skate_geometry must create a SkateCollision/ prim under the ankle path."""
        from pxr import UsdGeom, UsdPhysics
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                        "../../source/unitree_rl_lab"))
        from unitree_rl_lab.tasks.skating.mdp.skate_attachment import (
            _ensure_wheel_physics_material, _add_skate_geometry,
        )

        stage = self._make_in_memory_stage()
        ankle_path = "/Robot/left_ankle_roll_link"
        stage.DefinePrim(ankle_path, "Xform")
        mat_path = "/Looks/SkateWheelMaterial"
        _ensure_wheel_physics_material(stage, mat_path)

        _add_skate_geometry(stage, ankle_path, mat_path,
                            (0.08, 0.05, 0.02), (0.15, 0.15, 0.15), (0.7, 0.7, 0.7))

        # SkateVisual must exist
        assert stage.GetPrimAtPath(f"{ankle_path}/SkateVisual").IsValid(), \
            "SkateVisual Xform not found"

        # SkateCollision must exist with two wheel cylinders
        col_root = f"{ankle_path}/SkateCollision"
        assert stage.GetPrimAtPath(col_root).IsValid(), \
            "SkateCollision Xform not created — wheels have NO physics!"

        for name in ("WheelFront", "WheelBack"):
            col_prim = stage.GetPrimAtPath(f"{col_root}/{name}")
            assert col_prim.IsValid(), \
                f"SkateCollision/{name} prim not found"
            assert UsdPhysics.CollisionAPI(col_prim), \
                f"CollisionAPI not applied to SkateCollision/{name}"

    def test_collision_wheels_are_invisible(self):
        """Collision cylinders must be invisible — visuals are in SkateVisual/."""
        from pxr import UsdGeom, UsdPhysics
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                        "../../source/unitree_rl_lab"))
        from unitree_rl_lab.tasks.skating.mdp.skate_attachment import (
            _ensure_wheel_physics_material, _add_skate_geometry,
        )

        stage = self._make_in_memory_stage()
        ankle_path = "/Robot/right_ankle_roll_link"
        stage.DefinePrim(ankle_path, "Xform")
        mat_path = "/Looks/SkateWheelMaterialVis"
        _ensure_wheel_physics_material(stage, mat_path)

        _add_skate_geometry(stage, ankle_path, mat_path,
                            (0.08, 0.05, 0.02), (0.15, 0.15, 0.15), (0.7, 0.7, 0.7))

        for name in ("WheelFront", "WheelBack"):
            col_prim = stage.GetPrimAtPath(f"{ankle_path}/SkateCollision/{name}")
            imageable = UsdGeom.Imageable(col_prim)
            visibility = imageable.ComputeVisibility()
            assert visibility == UsdGeom.Tokens.invisible, \
                f"SkateCollision/{name} should be invisible, got '{visibility}'"
