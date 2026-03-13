"""SMPL/SMPL-X to Unitree G1-29DOF joint-angle retargeting.

Maps SMPL body_pose (21 joints × axis-angle, output of GENMO/HMR2/GVHMR) to
G1-29DOF joint angles by:

  1. Converting axis-angle rotations → 3×3 rotation matrices (scipy)
  2. Aligning coordinate frames:
       SMPL  : Y-up, X-right, Z-back  (camera looks toward person's face)
       G1    : Z-up, X-fwd,  Y-left   (robot's own body frame)
     via a fixed orthogonal alignment matrix T
  3. Extracting per-DOF scalar angles:
       - 3-DOF joints (hip, shoulder): ZXY Euler angles
       - 1-DOF joints (knee, elbow): Y-axis projection
       - 2-DOF joints (ankle, wrist): ZXY subset
  4. Clamping all angles to G1 URDF joint limits

Additional outputs for building AMP obs (matching amp_obs.py exactly):
  - Joint velocities (finite-difference)
  - Projected gravity in body frame
  - Root linear velocity in body frame
  - Root angular velocity in body frame
  - Approximate ankle/wrist positions relative to root (simplified FK)
  - Approximate ankle velocities

SMPL body_pose layout (T, 21, 3) — body_pose[:, j] = joint (j+1):
  body_pose[:, 0]  = L_Hip       body_pose[:, 1]  = R_Hip
  body_pose[:, 2]  = Spine1      body_pose[:, 3]  = L_Knee
  body_pose[:, 4]  = R_Knee      body_pose[:, 5]  = Spine2
  body_pose[:, 6]  = L_Ankle     body_pose[:, 7]  = R_Ankle
  body_pose[:, 8]  = Spine3      body_pose[:, 12] = L_Collar
  body_pose[:, 13] = R_Collar    body_pose[:, 15] = L_Shoulder
  body_pose[:, 16] = R_Shoulder  body_pose[:, 17] = L_Elbow
  body_pose[:, 18] = R_Elbow     body_pose[:, 19] = L_Wrist
  body_pose[:, 20] = R_Wrist

G1-29DOF joint order (must match tasks/skating/mdp/amp_obs.py):
  0  left_hip_pitch      6  right_hip_pitch   12 waist_yaw
  1  left_hip_roll       7  right_hip_roll    13 waist_roll
  2  left_hip_yaw        8  right_hip_yaw     14 waist_pitch
  3  left_knee           9  right_knee        15 left_shoulder_pitch
  4  left_ankle_pitch   10  right_ankle_pitch 16 left_shoulder_roll
  5  left_ankle_roll    11  right_ankle_roll  17 left_shoulder_yaw
                                              18 left_elbow
                                              19 left_wrist_roll
                                              20 left_wrist_pitch
                                              21 left_wrist_yaw
                                              22 right_shoulder_pitch
                                              23 right_shoulder_roll
                                              24 right_shoulder_yaw
                                              25 right_elbow
                                              26 right_wrist_roll
                                              27 right_wrist_pitch
                                              28 right_wrist_yaw
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ─── G1-29DOF joint limits (radians, from Unitree G1 29-DOF URDF) ────────────
# Order must match G1 joint ordering defined in mdp/amp_obs.py and gen_skating_reference.py
_JOINT_LIMITS: list[tuple[float, float]] = [
    (-1.57,  2.53),  #  0  left_hip_pitch
    (-0.52,  2.53),  #  1  left_hip_roll
    (-0.87,  0.87),  #  2  left_hip_yaw
    (-0.09,  2.87),  #  3  left_knee
    (-0.87,  0.52),  #  4  left_ankle_pitch
    (-0.26,  0.26),  #  5  left_ankle_roll
    (-1.57,  2.53),  #  6  right_hip_pitch
    (-2.53,  0.52),  #  7  right_hip_roll   (note: negative convention vs left)
    (-0.87,  0.87),  #  8  right_hip_yaw
    (-0.09,  2.87),  #  9  right_knee
    (-0.87,  0.52),  # 10  right_ankle_pitch
    (-0.26,  0.26),  # 11  right_ankle_roll
    (-2.62,  2.62),  # 12  waist_yaw
    (-0.52,  0.52),  # 13  waist_roll
    (-0.52,  0.52),  # 14  waist_pitch
    (-3.14,  3.14),  # 15  left_shoulder_pitch
    (-0.87,  2.53),  # 16  left_shoulder_roll
    (-2.97,  2.97),  # 17  left_shoulder_yaw
    (-1.57,  2.87),  # 18  left_elbow
    (-1.57,  1.57),  # 19  left_wrist_roll
    (-1.57,  1.57),  # 20  left_wrist_pitch
    (-1.57,  1.57),  # 21  left_wrist_yaw
    (-3.14,  3.14),  # 22  right_shoulder_pitch
    (-2.53,  0.87),  # 23  right_shoulder_roll
    (-2.97,  2.97),  # 24  right_shoulder_yaw
    (-1.57,  2.87),  # 25  right_elbow
    (-1.57,  1.57),  # 26  right_wrist_roll
    (-1.57,  1.57),  # 27  right_wrist_pitch
    (-1.57,  1.57),  # 28  right_wrist_yaw
]

JOINT_LIMIT_LO = np.array([lo for lo, _ in _JOINT_LIMITS], dtype=np.float32)
JOINT_LIMIT_HI = np.array([hi for _, hi in _JOINT_LIMITS], dtype=np.float32)

# ─── G1 default (neutral skating stance) joint positions ─────────────────────
# Matches gen_skating_reference.py::DEFAULT_JOINT_POS exactly so that the
# human reference blends with the analytical reference without discontinuity.
DEFAULT_JOINT_POS = np.zeros(29, dtype=np.float32)
DEFAULT_JOINT_POS[0]  = -0.15   # left_hip_pitch
DEFAULT_JOINT_POS[3]  =  0.40   # left_knee
DEFAULT_JOINT_POS[4]  = -0.25   # left_ankle_pitch
DEFAULT_JOINT_POS[6]  = -0.15   # right_hip_pitch
DEFAULT_JOINT_POS[9]  =  0.40   # right_knee
DEFAULT_JOINT_POS[10] = -0.25   # right_ankle_pitch
DEFAULT_JOINT_POS[16] =  0.25   # left_shoulder_roll
DEFAULT_JOINT_POS[23] = -0.25   # right_shoulder_roll
DEFAULT_JOINT_POS[18] =  0.97   # left_elbow
DEFAULT_JOINT_POS[25] =  0.97   # right_elbow

# ─── Frame alignment: SMPL (Y-up, X-right, Z-back) → G1 (Z-up, X-fwd, Y-left) ──
#
#  SMPL world frame: person stands along +Y, faces -Z, right arm along +X
#  G1   body  frame: robot  faces  +X,  left  along +Y,  up along +Z
#
#  Mapping:
#    G1.x = -SMPL.z   (forward  = -backward)
#    G1.y = -SMPL.x   (left     = -right)
#    G1.z = +SMPL.y   (up       = +up)
_T = np.array([[ 0,  0, -1],
               [-1,  0,  0],
               [ 0,  1,  0]], dtype=np.float64)
_TT = _T.T   # _T^{-1} = _T^T since _T is orthogonal

# ─── Approximate body link offsets in G1 default pose (root frame, metres) ───
# Used for simplified FK to generate ankle/wrist positions for AMP obs.
_L_ANKLE_DEFAULT = np.array([ 0.01,  0.10, -0.80], dtype=np.float32)
_R_ANKLE_DEFAULT = np.array([ 0.01, -0.10, -0.80], dtype=np.float32)
_L_WRIST_DEFAULT = np.array([-0.10,  0.30,  0.10], dtype=np.float32)
_R_WRIST_DEFAULT = np.array([-0.10, -0.30,  0.10], dtype=np.float32)

_FEMUR_LEN    = 0.38  # hip    → knee       (metres)
_SHANK_LEN    = 0.37  # knee   → ankle
_UPPER_ARM_LEN = 0.28  # shoulder → elbow
_LOWER_ARM_LEN = 0.24  # elbow    → wrist


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _aa_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) → rotation matrices (..., 3, 3)."""
    orig = axis_angle.shape[:-1]
    flat = axis_angle.reshape(-1, 3)
    return Rotation.from_rotvec(flat).as_matrix().reshape(*orig, 3, 3)


def _frame_align(R_smpl: np.ndarray) -> np.ndarray:
    """Transform rotation matrix from SMPL frame to G1 robot frame.

    R_g1 = T @ R_smpl @ T^T
    Works on batched inputs (..., 3, 3).
    """
    return _T @ R_smpl @ _TT


def _euler_zxy(R: np.ndarray) -> np.ndarray:
    """ZXY Euler angles from rotation matrix (..., 3, 3).

    Returns (..., 3) = [yaw(z), roll(x), pitch(y)].
    Suitable for extracting all three DOFs of a hip or shoulder joint.
    """
    r = Rotation.from_matrix(R.reshape(-1, 3, 3))
    return r.as_euler("ZXY", degrees=False).reshape(*R.shape[:-2], 3)


def _euler_zyx(R: np.ndarray) -> np.ndarray:
    """ZYX Euler angles from rotation matrix (..., 3, 3).

    Returns (..., 3) = [yaw(z), pitch(y), roll(x)].
    Used for the waist (spine) decomposition.
    """
    r = Rotation.from_matrix(R.reshape(-1, 3, 3))
    return r.as_euler("ZYX", degrees=False).reshape(*R.shape[:-2], 3)


def _proj_y(R: np.ndarray) -> np.ndarray:
    """Extract sagittal (Y-axis) rotation angle from rotation matrix (..., 3, 3).

    Computes atan2(-R[..., 2, 0], R[..., 0, 0]) which is the signed rotation
    about Y valid for nearly-pure sagittal joints (knee, elbow).
    """
    return np.arctan2(-R[..., 2, 0], R[..., 0, 0]).astype(np.float32)


def _compose(*Rs: np.ndarray) -> np.ndarray:
    """Left-to-right compose batched rotation matrices: R0 @ R1 @ ... ."""
    result = Rs[0]
    for R in Rs[1:]:
        result = result @ R
    return result


# ─── Public API ───────────────────────────────────────────────────────────────

class SMPLtoG1Retargeter:
    """Retargets SMPL/SMPL-X body-pose sequences to Unitree G1-29DOF joint angles.

    Supports SMPL (21-joint body pose) and SMPL-X (body pose identical structure).
    The resulting joint angles are expressed relative to the G1 default skating
    stance (DEFAULT_JOINT_POS) and clamped to URDF limits.

    Usage::

        retargeter = SMPLtoG1Retargeter(fps=30.0)
        result = retargeter.retarget(body_pose, global_orient, transl)

    Args:
        fps: Frames per second of the input sequence (used for velocity FD).
        scale: Global scale factor applied to root translation
            (SMPL outputs are in metres by default; set to 1.0 unless rescaling).
    """

    def __init__(self, fps: float = 30.0, scale: float = 1.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.scale = scale

    def retarget(
        self,
        body_pose: np.ndarray,      # (T, 63) or (T, 21, 3) axis-angle
        global_orient: np.ndarray,  # (T, 3)  pelvis axis-angle
        transl: np.ndarray,         # (T, 3)  root translation in world frame
    ) -> dict:
        """Retarget SMPL pose sequence to G1 joint angles and root kinematics.

        Args:
            body_pose: SMPL body joints axis-angle (T, 63) or (T, 21, 3).
            global_orient: Root/pelvis orientation axis-angle (T, 3).
            transl: Root translation in SMPL world frame (T, 3), metres.

        Returns:
            dict with keys:
              joint_pos         (T, 29) float32 — G1 joint angles (rad)
              joint_vel         (T, 29) float32 — finite-diff velocities (rad/s)
              proj_grav         (T, 3)  float32 — gravity projected into body frame
              lin_vel_b         (T, 3)  float32 — root linear velocity, body frame (m/s)
              ang_vel_b         (T, 3)  float32 — root angular velocity, body frame (rad/s)
              ankle_left_rel    (T, 3)  float32 — left  ankle pos relative to root (m)
              ankle_right_rel   (T, 3)  float32 — right ankle pos relative to root (m)
              wrist_left_rel    (T, 3)  float32 — left  wrist pos relative to root (m)
              wrist_right_rel   (T, 3)  float32 — right wrist pos relative to root (m)
              ankle_left_vel    (T, 3)  float32 — left  ankle velocity (m/s)
              ankle_right_vel   (T, 3)  float32 — right ankle velocity (m/s)
        """
        T = body_pose.shape[0]
        if body_pose.ndim == 2:
            body_pose = body_pose.reshape(T, 21, 3)

        body_rot = _aa_to_rotmat(body_pose)           # (T, 21, 3, 3)
        root_rot_smpl = _aa_to_rotmat(global_orient)  # (T, 3, 3)
        root_rot_g1 = _frame_align(root_rot_smpl)     # (T, 3, 3) in robot frame

        # ── Extract SMPL joint rotation matrices ──────────────────────────────
        R_lhip      = body_rot[:, 0]   # L_Hip
        R_rhip      = body_rot[:, 1]   # R_Hip
        R_spine1    = body_rot[:, 2]   # Spine1
        R_lknee     = body_rot[:, 3]   # L_Knee
        R_rknee     = body_rot[:, 4]   # R_Knee
        R_spine2    = body_rot[:, 5]   # Spine2
        R_lankle    = body_rot[:, 6]   # L_Ankle
        R_rankle    = body_rot[:, 7]   # R_Ankle
        R_spine3    = body_rot[:, 8]   # Spine3
        R_lcollar   = body_rot[:, 12]  # L_Collar
        R_rcollar   = body_rot[:, 13]  # R_Collar
        R_lshoulder = body_rot[:, 15]  # L_Shoulder
        R_rshoulder = body_rot[:, 16]  # R_Shoulder
        R_lelbow    = body_rot[:, 17]  # L_Elbow
        R_relbow    = body_rot[:, 18]  # R_Elbow
        R_lwrist    = body_rot[:, 19]  # L_Wrist
        R_rwrist    = body_rot[:, 20]  # R_Wrist

        q = DEFAULT_JOINT_POS[np.newaxis].repeat(T, axis=0).astype(np.float32)

        # ── Hip joints (3 DOF each) ───────────────────────────────────────────
        # ZXY decomposition after frame alignment:
        #   index 0 = Z-angle → hip_yaw     (internal/external rotation)
        #   index 1 = X-angle → hip_roll    (abduction/adduction)
        #   index 2 = Y-angle → hip_pitch   (flexion/extension)
        lhip_g1 = _frame_align(R_lhip)
        rhip_g1 = _frame_align(R_rhip)
        lhip_e  = _euler_zxy(lhip_g1)   # (T, 3): [yaw, roll, pitch]
        rhip_e  = _euler_zxy(rhip_g1)

        # hip_pitch: forward flexion is positive in G1, negative in SMPL (-Z in G1)
        q[:, 0] = -lhip_e[:, 2] + DEFAULT_JOINT_POS[0]  # left_hip_pitch
        q[:, 1] =  lhip_e[:, 1] + DEFAULT_JOINT_POS[1]  # left_hip_roll
        q[:, 2] =  lhip_e[:, 0] + DEFAULT_JOINT_POS[2]  # left_hip_yaw

        q[:, 6] = -rhip_e[:, 2] + DEFAULT_JOINT_POS[6]  # right_hip_pitch
        q[:, 7] = -rhip_e[:, 1] + DEFAULT_JOINT_POS[7]  # right_hip_roll (negated: opposite side)
        q[:, 8] =  rhip_e[:, 0] + DEFAULT_JOINT_POS[8]  # right_hip_yaw

        # ── Knee joints (1 DOF each — sagittal flexion) ───────────────────────
        q[:, 3] = _proj_y(_frame_align(R_lknee)) + DEFAULT_JOINT_POS[3]
        q[:, 9] = _proj_y(_frame_align(R_rknee)) + DEFAULT_JOINT_POS[9]

        # ── Ankle joints (2 DOF each — pitch + roll) ──────────────────────────
        lankle_e = _euler_zxy(_frame_align(R_lankle))   # [yaw, roll, pitch]
        rankle_e = _euler_zxy(_frame_align(R_rankle))

        q[:,  4] = lankle_e[:, 2] + DEFAULT_JOINT_POS[4]   # left_ankle_pitch
        q[:,  5] = lankle_e[:, 1] + DEFAULT_JOINT_POS[5]   # left_ankle_roll
        q[:, 10] = rankle_e[:, 2] + DEFAULT_JOINT_POS[10]  # right_ankle_pitch
        q[:, 11] = rankle_e[:, 1] + DEFAULT_JOINT_POS[11]  # right_ankle_roll

        # ── Waist (composed Spine1 + Spine2 + Spine3) ─────────────────────────
        spine_g1 = _frame_align(_compose(R_spine1, R_spine2, R_spine3))
        spine_e  = _euler_zyx(spine_g1)   # [yaw(z), pitch(y), roll(x)]

        q[:, 12] = spine_e[:, 0] + DEFAULT_JOINT_POS[12]  # waist_yaw
        q[:, 13] = spine_e[:, 2] + DEFAULT_JOINT_POS[13]  # waist_roll  (ZYX index 2)
        q[:, 14] = spine_e[:, 1] + DEFAULT_JOINT_POS[14]  # waist_pitch (ZYX index 1)

        # ── Shoulder joints (collar + shoulder, 3 DOF each) ───────────────────
        lsh_g1 = _frame_align(R_lcollar @ R_lshoulder)
        rsh_g1 = _frame_align(R_rcollar @ R_rshoulder)
        lsh_e  = _euler_zxy(lsh_g1)
        rsh_e  = _euler_zxy(rsh_g1)

        q[:, 15] =  lsh_e[:, 2] + DEFAULT_JOINT_POS[15]   # left_shoulder_pitch
        q[:, 16] =  lsh_e[:, 1] + DEFAULT_JOINT_POS[16]   # left_shoulder_roll
        q[:, 17] =  lsh_e[:, 0] + DEFAULT_JOINT_POS[17]   # left_shoulder_yaw
        q[:, 22] =  rsh_e[:, 2] + DEFAULT_JOINT_POS[22]   # right_shoulder_pitch
        q[:, 23] = -rsh_e[:, 1] + DEFAULT_JOINT_POS[23]   # right_shoulder_roll (negated)
        q[:, 24] =  rsh_e[:, 0] + DEFAULT_JOINT_POS[24]   # right_shoulder_yaw

        # ── Elbow joints (1 DOF each) ─────────────────────────────────────────
        q[:, 18] = _proj_y(_frame_align(R_lelbow)) + DEFAULT_JOINT_POS[18]
        q[:, 25] = _proj_y(_frame_align(R_relbow)) + DEFAULT_JOINT_POS[25]

        # ── Wrist joints (3 DOF each) ─────────────────────────────────────────
        lwr_e = _euler_zxy(_frame_align(R_lwrist))
        rwr_e = _euler_zxy(_frame_align(R_rwrist))

        q[:, 19] =  lwr_e[:, 1] + DEFAULT_JOINT_POS[19]   # left_wrist_roll
        q[:, 20] =  lwr_e[:, 2] + DEFAULT_JOINT_POS[20]   # left_wrist_pitch
        q[:, 21] =  lwr_e[:, 0] + DEFAULT_JOINT_POS[21]   # left_wrist_yaw
        q[:, 26] =  rwr_e[:, 1] + DEFAULT_JOINT_POS[26]   # right_wrist_roll
        q[:, 27] =  rwr_e[:, 2] + DEFAULT_JOINT_POS[27]   # right_wrist_pitch
        q[:, 28] =  rwr_e[:, 0] + DEFAULT_JOINT_POS[28]   # right_wrist_yaw

        # ── Clamp to G1 joint limits ──────────────────────────────────────────
        q = np.clip(q, JOINT_LIMIT_LO, JOINT_LIMIT_HI)

        # ── Joint velocities (central finite difference) ──────────────────────
        dq = np.zeros_like(q)
        dq[1:-1] = (q[2:] - q[:-2]) / (2.0 * self.dt)
        dq[0]    = (q[1] - q[0])    / self.dt
        dq[-1]   = (q[-1] - q[-2])  / self.dt

        # ── Projected gravity in body frame ───────────────────────────────────
        # g_world = [0, 0, -1] in G1 world; project into body via R^T @ g
        g_world_g1 = np.array([0.0, 0.0, -1.0])
        # root_rot_g1 is (T, 3, 3); R^T = transpose last two dims
        proj_grav = (root_rot_g1.transpose(0, 2, 1) @ g_world_g1).astype(np.float32)

        # ── Root linear velocity in body frame ────────────────────────────────
        transl_g1 = (transl.astype(np.float64) * self.scale @ _TT)  # SMPL → G1 world
        vel_w = np.zeros((T, 3))
        vel_w[1:-1] = (transl_g1[2:] - transl_g1[:-2]) / (2.0 * self.dt)
        vel_w[0]    = (transl_g1[1]  - transl_g1[0])   / self.dt
        vel_w[-1]   = (transl_g1[-1] - transl_g1[-2])  / self.dt
        # R^T @ v_world → body frame
        lin_vel_b = np.einsum("tij,tj->ti", root_rot_g1.transpose(0, 2, 1), vel_w).astype(np.float32)

        # ── Root angular velocity in body frame ───────────────────────────────
        ang_vel_b = _ang_vel_from_rotmat(root_rot_g1, self.dt)

        # ── Approximate ankle/wrist positions (simplified FK) ─────────────────
        ankle_l_rel, ankle_r_rel = self._ankle_positions(q)
        wrist_l_rel, wrist_r_rel = self._wrist_positions(q)

        # ── Ankle velocities (central FD) ─────────────────────────────────────
        def _central_fd(x: np.ndarray) -> np.ndarray:
            v = np.zeros_like(x)
            v[1:-1] = (x[2:] - x[:-2]) / (2.0 * self.dt)
            v[0]  = (x[1]  - x[0])   / self.dt
            v[-1] = (x[-1] - x[-2])  / self.dt
            return v

        ankle_l_vel = _central_fd(ankle_l_rel)
        ankle_r_vel = _central_fd(ankle_r_rel)

        return dict(
            joint_pos=q.astype(np.float32),
            joint_vel=dq.astype(np.float32),
            proj_grav=proj_grav,
            lin_vel_b=lin_vel_b,
            ang_vel_b=ang_vel_b,
            ankle_left_rel=ankle_l_rel.astype(np.float32),
            ankle_right_rel=ankle_r_rel.astype(np.float32),
            wrist_left_rel=wrist_l_rel.astype(np.float32),
            wrist_right_rel=wrist_r_rel.astype(np.float32),
            ankle_left_vel=ankle_l_vel.astype(np.float32),
            ankle_right_vel=ankle_r_vel.astype(np.float32),
        )

    # ── Private FK helpers ────────────────────────────────────────────────────

    def _ankle_positions(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simplified FK: ankle position relative to root from hip pitch/roll + knee.

        This is an approximation sufficient for the AMP obs — full FK would require
        the actual G1 URDF link lengths and joint axes.
        """
        T = q.shape[0]

        ankle_l = np.tile(_L_ANKLE_DEFAULT, (T, 1)).astype(np.float32)
        ankle_r = np.tile(_R_ANKLE_DEFAULT, (T, 1)).astype(np.float32)

        # Hip pitch delta (forward swing of femur → ankle moves forward)
        delta_lhp = q[:, 0] - DEFAULT_JOINT_POS[0]
        delta_rhp = q[:, 6] - DEFAULT_JOINT_POS[6]
        ankle_l[:, 0] += delta_lhp * _FEMUR_LEN * 0.5
        ankle_r[:, 0] += delta_rhp * _FEMUR_LEN * 0.5

        # Knee flexion delta (more bent → ankle moves up relative to root)
        delta_lk = q[:, 3] - DEFAULT_JOINT_POS[3]
        delta_rk = q[:, 9] - DEFAULT_JOINT_POS[9]
        ankle_l[:, 2] -= delta_lk * _SHANK_LEN * 0.25
        ankle_r[:, 2] -= delta_rk * _SHANK_LEN * 0.25

        # Hip roll → lateral ankle offset (key for skating edge control)
        ankle_l[:, 1] += q[:, 1] * 0.12   # left_hip_roll
        ankle_r[:, 1] -= q[:, 7] * 0.12   # right_hip_roll (negated)

        return ankle_l, ankle_r

    def _wrist_positions(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simplified FK: wrist position relative to root from shoulder + elbow angles."""
        T = q.shape[0]

        wrist_l = np.tile(_L_WRIST_DEFAULT, (T, 1)).astype(np.float32)
        wrist_r = np.tile(_R_WRIST_DEFAULT, (T, 1)).astype(np.float32)

        arm_reach = _UPPER_ARM_LEN + _LOWER_ARM_LEN

        # Shoulder pitch → forward/backward arm swing
        delta_lsp = q[:, 15] - DEFAULT_JOINT_POS[15]
        delta_rsp = q[:, 22] - DEFAULT_JOINT_POS[22]
        wrist_l[:, 0] += delta_lsp * arm_reach * 0.45
        wrist_r[:, 0] += delta_rsp * arm_reach * 0.45

        # Shoulder roll → lateral extension
        delta_lsr = q[:, 16] - DEFAULT_JOINT_POS[16]
        delta_rsr = q[:, 23] - DEFAULT_JOINT_POS[23]
        wrist_l[:, 1] += delta_lsr * _UPPER_ARM_LEN * 0.35
        wrist_r[:, 1] -= delta_rsr * _UPPER_ARM_LEN * 0.35  # negated: opposite side

        # Elbow flexion → vertical wrist position
        delta_le = q[:, 18] - DEFAULT_JOINT_POS[18]
        delta_re = q[:, 25] - DEFAULT_JOINT_POS[25]
        wrist_l[:, 2] += delta_le * _LOWER_ARM_LEN * 0.2
        wrist_r[:, 2] += delta_re * _LOWER_ARM_LEN * 0.2

        return wrist_l, wrist_r


def _ang_vel_from_rotmat(R: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity (body frame, rad/s) from rotation matrix sequence.

    Uses the skew-symmetric part of the incremental rotation dR ≈ I + dt·[ω]×.
    dR = R[t]^T @ R[t+1] is the incremental rotation in body frame.

    Args:
        R:  (T, 3, 3) rotation matrices in world frame.
        dt: Time step (seconds).

    Returns:
        (T, 3) float32 angular velocity in body frame.
    """
    T = R.shape[0]
    w = np.zeros((T, 3), dtype=np.float64)

    # Central differences on rotation matrices
    for t in range(1, T - 1):
        # Incremental rotation from t-1 → t+1 in body frame
        dR = R[t].T @ R[t + 1]
        s = 1.0 / (2.0 * dt)
        w[t, 0] = (dR[2, 1] - dR[1, 2]) * s
        w[t, 1] = (dR[0, 2] - dR[2, 0]) * s
        w[t, 2] = (dR[1, 0] - dR[0, 1]) * s

    # Forward / backward for boundary frames
    dR0 = R[0].T @ R[1]
    w[0, 0] = (dR0[2, 1] - dR0[1, 2]) / dt
    w[0, 1] = (dR0[0, 2] - dR0[2, 0]) / dt
    w[0, 2] = (dR0[1, 0] - dR0[0, 1]) / dt

    dRn = R[-2].T @ R[-1]
    w[-1, 0] = (dRn[2, 1] - dRn[1, 2]) / dt
    w[-1, 1] = (dRn[0, 2] - dRn[2, 0]) / dt
    w[-1, 2] = (dRn[1, 0] - dRn[0, 1]) / dt

    return w.astype(np.float32)


def build_amp_obs_from_retarget(result: dict) -> np.ndarray:
    """Pack retargeter output into the 85-dim AMP observation vector.

    The layout is **identical** to gen_skating_reference.py::build_amp_obs() and
    amp_obs.py::amp_observation_state(), so the resulting .npz is a drop-in
    replacement for the analytical skating_reference.npz.

    Layout (85 dims):
      [0:29]   joint_pos
      [29:58]  joint_vel × 0.05
      [58:61]  proj_grav
      [61:64]  lin_vel_b × 0.1
      [64:67]  ang_vel_b × 0.2
      [67:73]  ankle_left_rel || ankle_right_rel   (6 dims)
      [73:79]  wrist_left_rel || wrist_right_rel   (6 dims)
      [79:85]  ankle_left_vel×0.1 || ankle_right_vel×0.1  (6 dims)

    Args:
        result: Output dict from SMPLtoG1Retargeter.retarget().

    Returns:
        (T, 85) float32 AMP observation array.
    """
    jpos    = result["joint_pos"]                                         # (T, 29)
    jvel    = result["joint_vel"] * 0.05                                  # (T, 29)
    pgrav   = result["proj_grav"]                                         # (T, 3)
    linv    = result["lin_vel_b"] * 0.1                                   # (T, 3)
    angv    = result["ang_vel_b"] * 0.2                                   # (T, 3)
    ank_rel = np.concatenate([result["ankle_left_rel"],
                              result["ankle_right_rel"]], axis=-1)        # (T, 6)
    wri_rel = np.concatenate([result["wrist_left_rel"],
                              result["wrist_right_rel"]], axis=-1)        # (T, 6)
    ank_vel = np.concatenate([result["ankle_left_vel"] * 0.1,
                              result["ankle_right_vel"] * 0.1], axis=-1)  # (T, 6)

    amp_obs = np.concatenate(
        [jpos, jvel, pgrav, linv, angv, ank_rel, wri_rel, ank_vel], axis=-1
    )
    assert amp_obs.shape[1] == 85, f"Expected 85-dim AMP obs, got {amp_obs.shape[1]}"
    return amp_obs.astype(np.float32)  # (T, 85)
