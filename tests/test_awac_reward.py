from __future__ import annotations

import numpy as np

from mujoco_shared_control.awac.reward import AWACRewardV1Config, _episode_reward
from mujoco_shared_control.experts.rule_pick_place import RuleExpertStage


def _state() -> np.ndarray:
    state = np.zeros(42, dtype=np.float32)
    state[14:17] = [0.5, 0.0, 0.4]
    state[22:25] = [0.5, 0.0, 0.2]
    state[29:32] = [0.6, 0.0, 0.2]
    return state


def test_open_gripper_has_only_step_penalty_without_new_milestone() -> None:
    config = AWACRewardV1Config()
    obs = np.stack([_state()] * 4)
    next_obs = obs.copy()
    milestones = np.zeros((4, 5), dtype=np.uint8)
    reward, suppressed = _episode_reward(
        obs, next_obs,
        np.full(4, int(RuleExpertStage.OPEN_GRIPPER), dtype=np.uint8),
        milestones, np.arange(4), config,
    )
    np.testing.assert_allclose(reward, config.step_penalty)
    assert sum(suppressed.values()) == 0


def test_milestone_bonus_is_one_shot_and_not_bridged_across_filtered_gap() -> None:
    config = AWACRewardV1Config()
    obs = np.stack([_state()] * 4)
    next_obs = obs.copy()
    milestones = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    reward, suppressed = _episode_reward(
        obs, next_obs,
        np.full(4, int(RuleExpertStage.CLOSE_GRIPPER), dtype=np.uint8),
        milestones, np.asarray([0, 1, 3, 4]), config,
    )
    np.testing.assert_allclose(
        reward,
        [config.step_penalty, config.step_penalty + config.grasp_bonus,
         config.step_penalty, config.step_penalty],
    )
    assert suppressed["lift"] == 1


def test_progress_is_small_and_clipped() -> None:
    config = AWACRewardV1Config()
    obs = np.stack([_state()])
    next_obs = obs.copy()
    next_obs[0, 14] = 0.6
    reward, _ = _episode_reward(
        obs, next_obs,
        np.asarray([int(RuleExpertStage.PRE_GRASP)]),
        np.zeros((1, 5), dtype=np.uint8), np.asarray([0]), config,
    )
    assert reward[0] <= config.progress_clip + config.step_penalty + 1e-7
    assert reward[0] >= -config.progress_clip + config.step_penalty - 1e-7
