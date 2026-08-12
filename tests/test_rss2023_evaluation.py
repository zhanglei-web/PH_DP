from __future__ import annotations

import numpy as np

from mujoco_shared_control.rss2023.evaluation import (
    align_action_quaternion_sign,
    corrupt_action_sequence,
    quaternion_error_degrees,
)


def _actions(count: int = 6) -> np.ndarray:
    actions = np.zeros((count, 8), dtype=np.float64)
    actions[:, 0] = np.arange(count)
    actions[:, 3] = 1.0
    actions[:, 7] = 0.08
    return actions


def test_noisy_pilot_replaces_every_action_at_probability_one() -> None:
    expert = _actions()
    pool = _actions(3) + np.array([10.0, 0, 0, 0, 0, 0, 0, 0])
    noisy = corrupt_action_sequence(
        expert,
        pilot="noisy",
        probability=1.0,
        random_action_pool=pool,
        rng=np.random.default_rng(0),
    )
    assert np.all(noisy[:, 0] >= 10.0)


def test_laggy_pilot_repeats_previous_action_at_probability_one() -> None:
    expert = _actions()
    laggy = corrupt_action_sequence(
        expert,
        pilot="laggy",
        probability=1.0,
        random_action_pool=expert,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_allclose(laggy, np.repeat(expert[:1], len(expert), axis=0))


def test_quaternion_error_handles_antipodal_representation() -> None:
    first = np.array([[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(quaternion_error_degrees(first, -first), 0.0)

    action = np.array([[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.08]])
    antipodal = action.copy()
    antipodal[:, 3:7] *= -1.0
    np.testing.assert_allclose(align_action_quaternion_sign(antipodal, action), action)
