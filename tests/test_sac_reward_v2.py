from __future__ import annotations

import copy

import numpy as np
import pytest

from mujoco_shared_control.tasks.sac_reward import (
    SAC_DISCOUNT_GAMMA,
    SACPhase,
    SACPickPlaceProtocol,
    SACRewardV1,
)
from mujoco_shared_control.tasks.sac_reward_v2 import (
    SACPickPlaceProtocolV2,
    SACRewardV2,
)


def observation(*, ee=(0, 0, 0), obj=(0, 0, 0), goal=(.5, 0, 0), grasped=False):
    def pose(xyz):
        value = np.eye(4); value[:3, 3] = xyz; return value
    return {"ee_pose": pose(ee), "object_pose": pose(obj), "goal_pose": pose(goal),
            "object_grasped": grasped}


@pytest.mark.parametrize("phase", [SACPhase.PRE_GRASP, SACPhase.GRASP, SACPhase.TRANSPORT])
def test_p1_p2_p3_are_exactly_v1(phase: SACPhase) -> None:
    before = observation(ee=(.1, 0, .2), obj=(0, 0, 0), goal=(.5, 0, .2), grasped=True)
    after = observation(ee=(.08, 0, .2), obj=(0, 0, 0), goal=(.5, 0, .2), grasped=True)
    kwargs = {"stable_grasp_event": True} if phase == SACPhase.GRASP else {}
    v1 = SACRewardV1().step(before, after, phase, **kwargs)
    v2 = SACRewardV2().step(before, after, phase, **kwargs)
    assert v2.reward == pytest.approx(v1.reward)
    assert v2.components == v1.components


def test_p4_place_progress_is_exactly_v1_before_release() -> None:
    before = observation(obj=(.6, 0, 0), goal=(.5, 0, 0), grasped=True)
    after = observation(obj=(.55, 0, 0), goal=(.5, 0, 0), grasped=True)
    v1 = SACRewardV1().step(before, after, SACPhase.PLACE_AND_RETREAT)
    v2 = SACRewardV2().step(before, after, SACPhase.PLACE_AND_RETREAT)
    assert v2.components.p4_place_progress == pytest.approx(v1.components.p4_place_progress)


def _armed_v2() -> SACRewardV2:
    reward = SACRewardV2(); held = observation(grasped=True)
    reward.step(held, held, SACPhase.GRASP, stable_grasp_event=True)
    return reward


def test_stable_release_is_success_without_place_or_retreat_reward() -> None:
    reward = _armed_v2(); goal = (.5, 0, 0)
    before = observation(ee=goal, obj=goal, goal=goal, grasped=True)
    after = observation(ee=goal, obj=goal, goal=goal, grasped=False)
    result = reward.step(before, after, SACPhase.PLACE_AND_RETREAT,
                         successful_release_event=True)
    assert result.components.place_event == 0
    assert result.components.retreat_progress == 0
    assert result.components.success_terminal == 10
    assert result.terminated and not result.truncated
    with pytest.raises(RuntimeError, match="terminal step"):
        reward.step(after, after, SACPhase.PLACE_AND_RETREAT)


def test_protocol_v2_requires_same_four_stable_release_steps_then_terminates() -> None:
    protocol = SACPickPlaceProtocolV2(); protocol.phase = SACPhase.PLACE_AND_RETREAT
    protocol.reward.had_stable_grasp = True
    goal = (.5, 0, 0)
    before = observation(obj=goal, goal=goal, grasped=True)
    released = observation(obj=goal, goal=goal, grasped=False)
    result = protocol.step(before, released)
    assert not result.terminated
    for _ in range(2):
        result = protocol.step(released, released)
        assert not result.terminated
    result = protocol.step(released, released)
    assert result.terminated and result.termination_reason == "task_success"
    assert result.reward == pytest.approx(10.0)
    assert result.components.place_event == result.components.retreat_progress == 0


def test_v2_illegal_drop_and_timeout_match_v1() -> None:
    for cls in (SACRewardV1, SACRewardV2):
        reward = cls(); held = observation(grasped=True)
        reward.step(held, held, SACPhase.GRASP, stable_grasp_event=True)
        drop = reward.step(held, observation(grasped=False), SACPhase.TRANSPORT)
        assert drop.components.illegal_drop == -5 and drop.terminated
        timeout = cls().step(observation(), observation(), SACPhase.PRE_GRASP, time_limit=True)
        assert timeout.truncated and not timeout.terminated
        assert timeout.components.failure_terminal == 0


def test_v1_protocol_and_reward_are_not_modified_by_v2() -> None:
    protocol = SACPickPlaceProtocol(); protocol.phase = SACPhase.PLACE_AND_RETREAT
    protocol.reward.had_stable_grasp = True
    goal = (.5, 0, 0); before = observation(obj=goal, goal=goal, grasped=True)
    released = observation(obj=goal, goal=goal, grasped=False)
    protocol.step(before, released)
    protocol.step(released, released); protocol.step(released, released)
    result = protocol.step(released, released)
    assert result.components.place_event == 3
    assert not result.terminated
    assert SAC_DISCOUNT_GAMMA == .995


def test_v2_reset_does_not_leak_runtime_state() -> None:
    protocol = SACPickPlaceProtocolV2()
    original = copy.deepcopy(protocol.reward.__dict__)
    protocol.reward.had_stable_grasp = True; protocol._stable_release_steps = 3
    protocol.reset()
    assert protocol.reward.__dict__ == original
    assert protocol._stable_release_steps == 0
