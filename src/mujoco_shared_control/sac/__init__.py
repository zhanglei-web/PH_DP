"""Frozen SAC v1 actor and online optimization core."""

from mujoco_shared_control.sac.actor import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    SACGaussianActor,
    configure_full_mean_path_distillation,
    freeze_for_mean_calibration,
    initialize_from_bc,
)
from mujoco_shared_control.sac.agent import SACCore, SACCoreConfig, bootstrap_mask, polyak_update
from mujoco_shared_control.sac.constrained_actor import (
    SACConstrainedGaussianActor,
    constrained_transform,
    project_to_admissible,
    radial_squash,
)
from mujoco_shared_control.sac.critic import SACCritic, TwinSACCritic
from mujoco_shared_control.sac.replay_buffer import ReplayBatch, SACReplayBuffer
from mujoco_shared_control.sac.policy_anchor import InitialPolicyAnchor, PolicyAnchorConfig
from mujoco_shared_control.sac.trainer import EntropyWarmStartConfig, SACTrainer, TrainingProtocol

__all__ = [
    "LOG_STD_MAX", "LOG_STD_MIN", "SACGaussianActor",
    "configure_full_mean_path_distillation", "freeze_for_mean_calibration",
    "initialize_from_bc",
    "SACConstrainedGaussianActor", "constrained_transform",
    "project_to_admissible", "radial_squash",
    "ReplayBatch", "SACCritic", "SACCore", "SACCoreConfig", "SACReplayBuffer",
    "TwinSACCritic", "bootstrap_mask", "polyak_update",
    "InitialPolicyAnchor", "PolicyAnchorConfig", "EntropyWarmStartConfig",
    "SACTrainer", "TrainingProtocol",
]
