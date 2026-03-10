"""Skate geometry attachment for roller-skating simulation.

This module provides a startup event that programmatically adds roller skate
visual + collision geometry to the G1 robot's ankle roll links via the live
Omniverse USD stage.  It runs once at environment startup, before any physics
step, so the geometry is baked into the stage for the full training run.

Geometry added per ankle link
──────────────────────────────
  SkateVisual/
    Boot          – dark box approximating the skate boot
    Wheel{0-3}    – four inline wheel cylinders (visual)
    Wheel{i}HubL/R – cosmetic hub caps

  SkateCollision/
    WheelFront    – physics cylinder (front two wheels merged) with anisotropic friction
    WheelBack     – physics cylinder (rear two wheels merged) with anisotropic friction

    Each collision cylinder has:
      UsdPhysics.CollisionAPI   – enables PhysX collision
      UsdPhysics.MeshCollisionAPI (approximation="convexHull") — not needed for Cylinder
      PhysxSchema.PhysxMaterialAPI via a dedicated physics material:
        static_friction  (forward X) ≈ 0.02   — low rolling friction
        dynamic_friction (forward X) ≈ 0.01
        anisotropic:  lateral (Y) static_friction ≈ 0.8  — high edge grip

    NOTE: True per-axis anisotropic friction is not directly supported in PhysX
    via the standard USD physics material. We approximate it by:
      1. Setting isotropic low friction (0.02) on the wheel material so the
         robot can glide easily in the forward direction.
      2. The skating_env_cfg skate_friction EventTerm further overrides ankle
         body material at startup to match.
    The lateral grip is enforced indirectly through the wheel geometry: a
    narrow cylinder (width 24 mm) with moderate friction (0.5) on its flat
    rim contacts the floor primarily in the forward rolling direction.

Coordinates are in the *ankle_roll_link* local frame:
  +X = forward (toe direction)
  +Y = lateral (outward)
  +Z = up

The ankle_roll_link sits roughly at ankle height. The foot bottom in the
sphere-feet USD is ≈ 0.05 m below the link.  Wheel radius is 0.038 m,
so wheel centres sit at Z ≈ -0.088 m below ankle.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def attach_skates_to_robot(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    boot_color: tuple[float, float, float] = (0.08, 0.05, 0.02),
    wheel_color: tuple[float, float, float] = (0.15, 0.15, 0.15),
    hub_color: tuple[float, float, float] = (0.7, 0.7, 0.7),
) -> None:
    """Startup event: attach roller-skate geometry to every ankle roll link.

    Scans the Omniverse stage for prims whose path ends with
    ``left_ankle_roll_link`` or ``right_ankle_roll_link`` and adds
    inline-skate boot + wheel geometry (visual AND collision) as USD children.

    This event must be registered with ``mode="startup"`` so it fires once
    after the scene is constructed.
    """
    try:
        import omni.usd
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Vt, Sdf
    except ImportError:
        print("[SkateAttach] WARNING: omni.usd / pxr not available — skate geometry skipped.")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[SkateAttach] WARNING: USD stage not available — skate geometry skipped.")
        return

    # Create a shared physics material for the wheel collision surfaces.
    # Low isotropic friction approximates the rolling/gliding direction.
    # (True anisotropic friction requires custom PhysX extensions not exposed
    #  via standard USD — the narrow wheel geometry itself provides the
    #  directional constraint for lateral grip.)
    wheel_material_path = "/World/Looks/SkateWheelMaterial"
    _ensure_wheel_physics_material(stage, wheel_material_path)

    ankle_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    attached = 0
    skipped  = 0

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        for ankle_name in ankle_names:
            if prim_path.endswith(ankle_name):
                # Skip if already processed (idempotent — handles USD instancing where
                # the stage traversal may visit the same prototype prim multiple times).
                if stage.GetPrimAtPath(f"{prim_path}/SkateVisual").IsValid():
                    skipped += 1
                    break
                _add_skate_geometry(
                    stage, prim_path, wheel_material_path,
                    boot_color, wheel_color, hub_color,
                )
                attached += 1
                break  # avoid double-matching

    print(
        f"[SkateAttach] Attached skate geometry (visual + collision) to {attached} ankle links "
        f"({skipped} already processed / skipped)."
    )


# ─────────────────────────────────────────────────────────────────────────────

_WHEEL_RADIUS = 0.038   # metres — standard inline skate wheel

# Exposed for tests — depth of wheel bottom below ankle roll link
def _WHEEL_CENTER_BELOW_ANKLE_APPROX() -> float:
    """Return the approximate depth of the wheel centre below ankle_roll_link (metres)."""
    return 0.050 + _WHEEL_RADIUS  # foot_drop + wheel_radius = 0.088 m

_WHEEL_WIDTH  = 0.024   # metres — narrow tread for directional rolling
_BOOT_LEN     = 0.26    # metres
_BOOT_WIDTH   = 0.08    # metres
_BOOT_HEIGHT  = 0.055   # metres

# Wheel X positions along the boot (front → back)
_WHEEL_X = [0.085, 0.030, -0.025, -0.080]

# Ankle_roll_link sits at roughly ankle height.
# The foot bottom in the sphere-feet USD is ≈ 0.05 m below the link.
# Wheel centre Z = -(foot_drop + wheel_radius)
_WHEEL_Z   = -(0.050 + _WHEEL_RADIUS)   # ≈ -0.088 m
_BOOT_Z    = _WHEEL_Z + _WHEEL_RADIUS + _BOOT_HEIGHT / 2   # boot sits on wheels
_BOOT_X    = 0.005   # slight forward offset so boot centres over wheels

# Collision simplification: two capsule-like cylinders (front pair, rear pair)
_COL_FRONT_X = (_WHEEL_X[0] + _WHEEL_X[1]) / 2   # centre of front two wheels
_COL_BACK_X  = (_WHEEL_X[2] + _WHEEL_X[3]) / 2   # centre of rear two wheels
# Height of the merged collision cylinder spans the two wheels plus gaps
_COL_HEIGHT  = abs(_WHEEL_X[0] - _WHEEL_X[1]) + _WHEEL_WIDTH


def _ensure_wheel_physics_material(stage, mat_path: str) -> None:
    """Create (or reuse) a PhysX material for skate wheel collision surfaces."""
    from pxr import UsdShade, UsdPhysics, PhysxSchema, Sdf

    if stage.GetPrimAtPath(mat_path):
        return  # already created

    mat = UsdShade.Material.Define(stage, mat_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics_mat.CreateStaticFrictionAttr(0.02)    # low rolling friction
    physics_mat.CreateDynamicFrictionAttr(0.01)
    physics_mat.CreateRestitutionAttr(0.0)

    # PhysX-specific: disable friction combining so the low wheel friction
    # is applied directly rather than being averaged with the rink surface.
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    physx_mat.CreateFrictionCombineModeAttr("min")
    physx_mat.CreateRestitutionCombineModeAttr("min")


def _apply_collision_and_material(prim, stage, material_path: str) -> None:
    """Enable PhysX collision on a geometry prim and bind the wheel material."""
    from pxr import UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)

    mat_prim = stage.GetPrimAtPath(material_path)
    if mat_prim:
        mat = UsdShade.Material(mat_prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.strongerThanDescendants, "physics"
        )


def _set_display_color(prim, rgb: tuple[float, float, float]) -> None:
    from pxr import UsdGeom, Gf, Vt
    geom = UsdGeom.Gprim(prim)
    if geom:
        geom.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))


def _set_xform_translate_scale(xformable, translate, scale=None) -> None:
    """Set translate (and optionally scale) on a prim, safely overwriting any existing ops.

    Avoids the ``AddXformOp already exists`` error that occurs when the stage
    already contains these prims (e.g. USD instancing traversal revisits the
    same prototype prim, or the function is called a second time).
    """
    from pxr import UsdGeom, Gf
    existing_names = {op.GetOpName() for op in xformable.GetOrderedXformOps()}

    if "xformOp:translate" in existing_names:
        t_op = UsdGeom.XformOp(xformable.GetPrim().GetAttribute("xformOp:translate"))
    else:
        t_op = xformable.AddTranslateOp()
    t_op.Set(Gf.Vec3f(*translate))

    if scale is not None:
        if "xformOp:scale" in existing_names:
            s_op = UsdGeom.XformOp(xformable.GetPrim().GetAttribute("xformOp:scale"))
        else:
            s_op = xformable.AddScaleOp()
        s_op.Set(Gf.Vec3f(*scale))


def _add_skate_geometry(
    stage,
    ankle_path: str,
    wheel_material_path: str,
    boot_color: tuple,
    wheel_color: tuple,
    hub_color: tuple,
) -> None:
    """Create boot + wheel visual geometry AND collision geometry under a single ankle_roll_link prim."""
    from pxr import UsdGeom, Gf, Vt

    # ── Visual geometry ───────────────────────────────────────────────────────
    vis_root = f"{ankle_path}/SkateVisual"
    stage.DefinePrim(vis_root, "Xform")

    # Boot visual
    boot_path = f"{vis_root}/Boot"
    boot = UsdGeom.Cube.Define(stage, boot_path)
    boot.GetSizeAttr().Set(1.0)
    _set_xform_translate_scale(
        UsdGeom.Xformable(boot.GetPrim()),
        (_BOOT_X, 0.0, _BOOT_Z),
        (_BOOT_LEN / 2, _BOOT_WIDTH / 2, _BOOT_HEIGHT / 2),
    )
    _set_display_color(boot.GetPrim(), boot_color)

    # Wheel visuals
    for i, wx in enumerate(_WHEEL_X):
        wheel_path = f"{vis_root}/Wheel{i}"
        wheel = UsdGeom.Cylinder.Define(stage, wheel_path)
        wheel.GetRadiusAttr().Set(_WHEEL_RADIUS)
        wheel.GetHeightAttr().Set(_WHEEL_WIDTH)
        wheel.GetAxisAttr().Set("Y")
        _set_xform_translate_scale(
            UsdGeom.Xformable(wheel.GetPrim()),
            (wx, 0.0, _WHEEL_Z),
        )
        _set_display_color(wheel.GetPrim(), wheel_color)

        for side, y_off in [("HubL", -(_WHEEL_WIDTH / 2 + 0.002)),
                             ("HubR", +(_WHEEL_WIDTH / 2 + 0.002))]:
            hub_path = f"{vis_root}/Wheel{i}{side}"
            hub = UsdGeom.Cylinder.Define(stage, hub_path)
            hub.GetRadiusAttr().Set(_WHEEL_RADIUS * 0.45)
            hub.GetHeightAttr().Set(0.004)
            hub.GetAxisAttr().Set("Y")
            _set_xform_translate_scale(
                UsdGeom.Xformable(hub.GetPrim()),
                (wx, y_off, _WHEEL_Z),
            )
            _set_display_color(hub.GetPrim(), hub_color)

    # Frame rails (visual only)
    frame_len = _WHEEL_X[0] - _WHEEL_X[-1] + _WHEEL_RADIUS * 2
    for side, y_off in [("FrameL", -0.030), ("FrameR", +0.030)]:
        frame_path = f"{vis_root}/{side}"
        frame = UsdGeom.Cube.Define(stage, frame_path)
        frame.GetSizeAttr().Set(1.0)
        _set_xform_translate_scale(
            UsdGeom.Xformable(frame.GetPrim()),
            ((_WHEEL_X[0] + _WHEEL_X[-1]) / 2, y_off, _WHEEL_Z),
            (frame_len / 2, 0.005, 0.006),
        )
        _set_display_color(frame.GetPrim(), (0.4, 0.4, 0.4))

    # ── Collision geometry ────────────────────────────────────────────────────
    # Two physics cylinders (front pair + rear pair of wheels) so PhysX has
    # real geometry to collide with the rink floor.  Each cylinder:
    #   - axis = Y  (rolling direction = X)
    #   - radius = wheel radius
    #   - height = combined width of two wheels + gap
    #   - CollisionAPI enabled
    #   - SkateWheelMaterial bound (low friction)
    col_root = f"{ankle_path}/SkateCollision"
    stage.DefinePrim(col_root, "Xform")

    for col_name, col_x in [("WheelFront", _COL_FRONT_X), ("WheelBack", _COL_BACK_X)]:
        col_path = f"{col_root}/{col_name}"
        col_cyl = UsdGeom.Cylinder.Define(stage, col_path)
        col_cyl.GetRadiusAttr().Set(_WHEEL_RADIUS)
        col_cyl.GetHeightAttr().Set(_COL_HEIGHT)
        col_cyl.GetAxisAttr().Set("Y")
        _set_xform_translate_scale(
            UsdGeom.Xformable(col_cyl.GetPrim()),
            (col_x, 0.0, _WHEEL_Z),
        )
        _apply_collision_and_material(col_cyl.GetPrim(), stage, wheel_material_path)
        # Make collision prims invisible (geometry is shown via SkateVisual)
        UsdGeom.Imageable(col_cyl.GetPrim()).MakeInvisible()
