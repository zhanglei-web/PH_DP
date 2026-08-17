from __future__ import annotations

import numpy as np

from mujoco_shared_control.awac.milestones import (
    GeometricTaskPhase, MilestoneTracker, phase_from_milestones,
)


def _state() -> np.ndarray:
    state = np.zeros(43, np.float32)
    state[14:17] = [0.5, 0.0, 0.36]
    state[21] = 0.0
    state[22:25] = [0.5, 0.0, 0.20]
    state[29:32] = [0.7, 0.0, 0.20]
    return state


def test_geometric_milestones_are_ordered_and_latched() -> None:
    tracker = MilestoneTracker()
    state = _state()
    np.testing.assert_array_equal(tracker.reset(state), np.zeros(5, bool))

    state[42] = 1
    np.testing.assert_array_equal(tracker.update(state).current, [1, 0, 0, 0, 0])
    state[24] = 0.30
    np.testing.assert_array_equal(tracker.update(state).current, [1, 1, 0, 0, 0])
    state[22:24] = [0.70, 0.0]
    np.testing.assert_array_equal(tracker.update(state).current, [1, 1, 1, 0, 0])
    state[42] = 0
    state[21] = 0.055
    state[24] = 0.20
    np.testing.assert_array_equal(tracker.update(state).current, [1, 1, 1, 1, 0])
    state[14:17] = [0.7, 0.0, 0.36]
    np.testing.assert_array_equal(tracker.update(state).current, [1, 1, 1, 1, 1])

    # Losing all instantaneous conditions cannot clear a cumulative episode bit.
    state[:] = _state()
    np.testing.assert_array_equal(tracker.update(state).current, np.ones(5, bool))


def test_release_requires_goal_containment_and_open_gripper() -> None:
    tracker = MilestoneTracker()
    state = _state()
    tracker.reset(state)
    state[42] = 1; tracker.update(state)
    state[24] = 0.30; tracker.update(state)
    state[22:25] = [0.70, 0.0, 0.30]; tracker.update(state)
    state[42] = 0
    state[21] = 0.0549
    state[24] = 0.20
    assert not tracker.update(state).current[3]
    state[21] = 0.055
    assert tracker.update(state).current[3]


def test_retreat_uses_rule_waypoint_and_tolerance() -> None:
    tracker = MilestoneTracker()
    state = _state(); tracker.reset(state)
    state[42] = 1; tracker.update(state)
    state[24] = 0.30; tracker.update(state)
    state[22:25] = [0.70, 0.0, 0.20]
    state[42] = 0; state[21] = 0.055; tracker.update(state)
    state[14:17] = [0.70, 0.0, 0.3681]
    assert not tracker.update(state).current[4]
    state[14:17] = [0.70, 0.0, 0.3680]
    assert tracker.update(state).current[4]


def test_geometric_phase_is_the_ordered_milestone_prefix() -> None:
    legal = (
        [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1],
    )
    assert [phase_from_milestones(value) for value in legal] == list(GeometricTaskPhase)
