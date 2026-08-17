from copy import deepcopy

import numpy as np
import torch

from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.sac.agent import SACCore
from mujoco_shared_control.sac.diagnostics import (
    constrained_local_step, q_action_gradient, restore_environment,
    sample_with_log_std_override, snapshot_environment,
)
from mujoco_shared_control.sac.replay_buffer import SACReplayBuffer
from mujoco_shared_control.sac.trainer import ACTOR_ARTIFACT


def test_log_std_override_does_not_mutate_actor() -> None:
    core = SACCore(ACTOR_ARTIFACT)
    before = deepcopy(core.actor.state_dict())
    action = sample_with_log_std_override(core.actor, torch.zeros(4, 42), -4.0)
    assert action.shape == (4, 7) and torch.isfinite(action).all()
    for name, value in core.actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_snapshot_restore_reproduces_identical_transition() -> None:
    env = PickPlaceEnv(enable_camera=False, reward_version="sac_reward_v1")
    spec = ExpertActionSpec()
    adapter = ExpertCommandAdapter(env.ik_controller, spec)
    try:
        obs, _ = env.reset(seed=420000)
        adapter.reset(obs["ee_pose"], obs["q_obs"])
        snap = snapshot_environment(env, adapter, 0)
        command = np.array([.1, 0, 0, 0, 0, 0, 1], np.float64)
        outputs = []
        for _ in range(2):
            restore_environment(env, adapter, snap)
            adapted = adapter.adapt(spec.denormalize(command))
            nxt, reward, terminated, truncated, info = env.step(adapted.joint_target)
            outputs.append((nxt["q_obs"], reward, terminated, truncated, info["phase"]))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        assert outputs[0][1:] == outputs[1][1:]
    finally:
        env.close()


def test_counterfactual_branches_share_initial_state_and_do_not_write_replay() -> None:
    env = PickPlaceEnv(enable_camera=False, reward_version="sac_reward_v1")
    spec = ExpertActionSpec(); adapter = ExpertCommandAdapter(env.ik_controller, spec)
    replay = SACReplayBuffer(8, seed=7)
    replay.add(np.zeros(42), np.zeros(7), 0.0, np.zeros(42), False, False)
    before = deepcopy(replay.state_dict())
    try:
        obs, _ = env.reset(seed=420001)
        adapter.reset(obs["ee_pose"], obs["q_obs"])
        snapshot = snapshot_environment(env, adapter, 0)
        initial_states = []
        for command in (np.zeros(7), np.array([1e-4, 0, 0, 0, 0, 0, 0])):
            restore_environment(env, adapter, snapshot)
            current = np.empty_like(snapshot.integration_state)
            import mujoco
            mujoco.mj_getState(
                env.model, env.data, current, mujoco.mjtState.mjSTATE_INTEGRATION
            )
            initial_states.append(current)
            adapted = adapter.adapt(spec.denormalize(command))
            env.step(adapted.joint_target)
        np.testing.assert_array_equal(initial_states[0], initial_states[1])
        after = replay.state_dict()
        assert after["size"] == before["size"] and after["position"] == before["position"]
        for name in ("observation", "action", "reward", "next_observation",
                     "terminated", "truncated"):
            np.testing.assert_array_equal(after[name], before[name])
    finally:
        env.close()


def test_local_perturbation_is_admissible_and_q_gradient_finite() -> None:
    core = SACCore(ACTOR_ARTIFACT)
    state = torch.zeros(8, 42)
    action = core.actor.deterministic_action(state)
    gradient = q_action_gradient(core, state, action)
    candidate = constrained_local_step(action, gradient, 1e-3)
    assert torch.isfinite(gradient).all()
    assert torch.all(torch.linalg.vector_norm(candidate[:, :3], dim=-1) <= 1)
    assert torch.all(torch.linalg.vector_norm(candidate[:, 3:6], dim=-1) <= 1)
    assert torch.all(candidate[:, 6].abs() <= 1)
