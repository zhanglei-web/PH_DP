from __future__ import annotations

import numpy as np

from mujoco_shared_control import PickPlaceEnv
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.experts.interfaces import ExpertActionSpec


def test_expert_action_normalization_round_trip() -> None:
    spec = ExpertActionSpec()
    command = np.array([.01, -.02, .005, .02, -.04, .08, .03])
    np.testing.assert_allclose(spec.denormalize(spec.normalize(command)), command)


def test_adapter_clips_and_produces_joint_action() -> None:
    env = PickPlaceEnv(enable_camera=False)
    try:
        observation, _ = env.reset(seed=7, options={"randomize_object": False,
                                                    "randomize_goal": False})
        spec = ExpertActionSpec(max_translation_step_m=.01)
        adapter = ExpertCommandAdapter(env.ik_controller, spec)
        adapter.reset(observation["ee_pose"], observation["q_obs"])
        result = adapter.adapt(np.array([.1, 0, 0, 0, 0, 0, .08]))
        assert result.accepted
        assert result.action_clipped
        assert result.joint_target.shape == (8,)
        np.testing.assert_allclose(np.linalg.norm(result.clipped[:3]), .01)
    finally:
        env.close()
