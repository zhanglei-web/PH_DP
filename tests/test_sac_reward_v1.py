from __future__ import annotations

import numpy as np
import pytest

from mujoco_shared_control.tasks.sac_reward import (
    SACPhase,
    SACPickPlaceProtocol,
    SACRewardV1,
)


def observation(
    *, ee=(0.0, 0.0, 0.0), obj=(0.0, 0.0, 0.0),
    goal=(0.5, 0.0, 0.0), grasped=False,
) -> dict:
    def pose(xyz):
        result = np.eye(4)
        result[:3, 3] = xyz
        return result
    return {
        "ee_pose": pose(ee), "object_pose": pose(obj), "goal_pose": pose(goal),
        "object_grasped": grasped,
    }


@pytest.mark.parametrize(
    ("next_ee", "sign"),
    [((0.0, 0.0, 0.006), 1), ((0.0, 0.0, -0.006), -1), ((0.0, 0.0, 0.0), 0)],
)
def test_p1_signed_progress(next_ee: tuple[float, ...], sign: int) -> None:
    reward = SACRewardV1().step(
        observation(), observation(ee=next_ee), SACPhase.PRE_GRASP
    ).components.p1_progress
    if sign:
        assert np.sign(reward) == sign
    else:
        assert reward == pytest.approx(0.0)


def test_stable_grasp_event_is_rewarded_once() -> None:
    reward = SACRewardV1()
    obs = observation(grasped=True)
    first = reward.step(obs, obs, SACPhase.GRASP, stable_grasp_event=True)
    second = reward.step(obs, obs, SACPhase.GRASP, stable_grasp_event=True)
    assert first.components.grasp_event == 2.0
    assert second.components.grasp_event == 0.0


def test_p3_target_and_signed_progress() -> None:
    goal = np.array([0.4, -0.2, 0.25])
    np.testing.assert_allclose(SACRewardV1.above_goal(goal), goal + [0, 0, 0.16])
    target = goal + [0, 0, 0.16]
    start = target + [0.1, 0, 0]
    closer = target + [0.05, 0, 0]
    farther = target + [0.15, 0, 0]
    positive = SACRewardV1().step(
        observation(ee=start, goal=goal), observation(ee=closer, goal=goal),
        SACPhase.TRANSPORT,
    ).components.p3_progress
    negative = SACRewardV1().step(
        observation(ee=start, goal=goal), observation(ee=farther, goal=goal),
        SACPhase.TRANSPORT,
    ).components.p3_progress
    assert positive > 0
    assert negative < 0


def armed_reward() -> SACRewardV1:
    result = SACRewardV1()
    obs = observation(grasped=True)
    result.step(obs, obs, SACPhase.GRASP, stable_grasp_event=True)
    return result


def test_illegal_drop_is_terminal_and_blocks_delayed_recovery() -> None:
    reward = armed_reward()
    result = reward.step(
        observation(grasped=True), observation(grasped=False), SACPhase.TRANSPORT
    )
    assert result.components.illegal_drop == -5.0
    assert result.terminated and not result.truncated
    with pytest.raises(RuntimeError, match="cannot accumulate"):
        reward.step(observation(), observation(), SACPhase.TRANSPORT)


def test_legal_place_event_once_and_early_release_has_no_bonus() -> None:
    goal = (0.5, 0.0, 0.0)
    outside = (0.7, 0.0, 0.0)
    early = armed_reward().step(
        observation(obj=outside, goal=goal, grasped=True),
        observation(obj=outside, goal=goal, grasped=False),
        SACPhase.PLACE_AND_RETREAT, successful_release_event=True,
    )
    assert early.components.place_event == 0.0
    assert early.components.illegal_drop == -5.0

    reward = armed_reward()
    before = observation(obj=goal, goal=goal, grasped=True)
    after = observation(obj=goal, goal=goal, grasped=False)
    first = reward.step(
        before, after, SACPhase.PLACE_AND_RETREAT,
        successful_release_event=True,
    )
    second = reward.step(
        after, after, SACPhase.PLACE_AND_RETREAT,
        successful_release_event=True,
    )
    assert first.components.place_event == 3.0
    assert second.components.place_event == 0.0


def test_retreat_progress_and_full_success_once() -> None:
    reward = armed_reward()
    goal = np.array([0.5, 0.0, 0.0])
    release = observation(ee=goal, obj=goal, goal=goal, grasped=True)
    released = observation(ee=goal, obj=goal, goal=goal, grasped=False)
    reward.step(
        release, released, SACPhase.PLACE_AND_RETREAT,
        successful_release_event=True,
    )
    target = goal + [0, 0, 0.16]
    result = reward.step(
        observation(ee=goal, obj=goal, goal=goal),
        observation(ee=target, obj=goal, goal=goal),
        SACPhase.PLACE_AND_RETREAT, full_success=True,
    )
    assert result.components.retreat_progress > 0
    assert result.components.success_terminal == 10.0
    assert result.terminated and not result.truncated
    with pytest.raises(RuntimeError):
        reward.step(released, released, SACPhase.PLACE_AND_RETREAT, full_success=True)


def test_time_limit_is_truncation_without_failure_penalty() -> None:
    result = SACRewardV1().step(
        observation(), observation(), SACPhase.PRE_GRASP, time_limit=True
    )
    assert not result.terminated and result.truncated
    assert result.components.failure_terminal == 0.0
    assert result.termination_reason == "time_limit"


def test_reset_isolates_events_and_initial_distances() -> None:
    reward = armed_reward()
    reward.reset()
    obs = observation(grasped=True)
    event = reward.step(obs, obs, SACPhase.GRASP, stable_grasp_event=True)
    assert event.components.grasp_event == 2.0
    assert reward.had_stable_grasp


def test_ground_truth_protocol_uses_four_frozen_phases() -> None:
    protocol = SACPickPlaceProtocol()
    assert list(SACPhase) == [
        SACPhase.PRE_GRASP, SACPhase.GRASP, SACPhase.TRANSPORT,
        SACPhase.PLACE_AND_RETREAT,
    ]
    obj = np.array([0.5, 0.0, 0.25])
    grasp = obj + [0, 0, 0.012]
    result = protocol.step(
        observation(ee=grasp, obj=obj), observation(ee=grasp, obj=obj)
    )
    assert result.next_phase == SACPhase.GRASP


def test_protocol_requires_eight_grasp_steps_and_four_release_steps() -> None:
    protocol = SACPickPlaceProtocol()
    protocol.phase = SACPhase.GRASP
    held = observation(grasped=True)
    for _ in range(7):
        result = protocol.step(held, held)
        assert result.components.grasp_event == 0.0
    result = protocol.step(held, held)
    assert result.components.grasp_event == 2.0
    assert result.next_phase == SACPhase.TRANSPORT

    protocol.phase = SACPhase.PLACE_AND_RETREAT
    goal = (0.5, 0.0, 0.0)
    before = observation(obj=goal, goal=goal, grasped=True)
    released = observation(obj=goal, goal=goal, grasped=False)
    result = protocol.step(before, released)
    assert result.components.place_event == 0.0
    for _ in range(2):
        result = protocol.step(released, released)
        assert result.components.place_event == 0.0
    result = protocol.step(released, released)
    assert result.components.place_event == 3.0
