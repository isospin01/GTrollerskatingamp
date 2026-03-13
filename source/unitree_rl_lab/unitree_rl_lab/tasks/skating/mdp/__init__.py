"""MDP components for roller skating task.

Imports everything from the standard locomotion mdp and extends it with
skating-specific reward functions and observations.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# Re-export locomotion mdp utilities (commands, curriculums, observations)
from unitree_rl_lab.tasks.locomotion.mdp import *  # noqa: F401, F403

# Skating-specific extensions (override any same-named symbols from above)
from .rewards import (  # noqa: F401
    forward_velocity_tracking_exp,
    lateral_velocity_penalty,
    skate_glide_continuity,
    push_off_rhythm,
    skate_foot_edge_contact,
    upright_orientation_l2,
    base_height_skating_l2,
    skate_energy,
    skate_action_rate,
    ang_vel_z_l2,
)
from .observations import (  # noqa: F401
    skating_phase_signal,
    foot_contact_forces_normalized,
    base_lin_vel_forward,
)
from .events import reset_skating_pose  # noqa: F401
from .amp_obs import amp_observation_state  # noqa: F401
from .skate_attachment import attach_skates_to_robot  # noqa: F401
from .curriculums import skating_lin_vel_cmd_levels, skating_ang_vel_cmd_levels  # noqa: F401
from .eureka_rewards import eureka_task_reward  # noqa: F401
