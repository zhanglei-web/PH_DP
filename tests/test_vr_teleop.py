from __future__ import annotations

import numpy as np
import pytest

from mujoco_shared_control.control.vr_teleop import (
    PoseTargetFilter,
    RelativePoseMapper,
    controller_pose_from_xrt,
)
from mujoco_shared_control.utils.pose import make_pose


def test_relative_pose_mapper_preserves_initial_alignment() -> None:
    mapper = RelativePoseMapper()
    controller_initial = controller_pose_from_xrt([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    ee_initial = make_pose([0.3, -0.2, 0.5], np.eye(3))
    mapper.align(controller_initial, ee_initial)

    target = mapper.target(controller_initial)

    np.testing.assert_allclose(target, ee_initial)


def test_relative_pose_mapper_maps_translation_and_rotation() -> None:
    mapper = RelativePoseMapper(translation_scale=0.5)
    controller_initial = controller_pose_from_xrt([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ee_initial = make_pose([0.3, -0.2, 0.5], np.eye(3))
    mapper.align(controller_initial, ee_initial)

    half_turn_about_z = controller_pose_from_xrt([0.2, -0.4, 0.6, 0.0, 0.0, 1.0, 0.0])
    target = mapper.target(half_turn_about_z)

    np.testing.assert_allclose(target[:3, 3], [0.4, -0.4, 0.8])
    np.testing.assert_allclose(target[:3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-7)


def test_relative_pose_mapper_can_hold_initial_orientation() -> None:
    mapper = RelativePoseMapper(control_orientation=False)
    initial = controller_pose_from_xrt([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ee_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mapper.align(initial, make_pose([0.3, 0.0, 0.5], ee_rotation))

    rotated_controller = controller_pose_from_xrt(
        [0.1, 0.2, 0.3, 0.0, 0.0, 1.0, 0.0]
    )
    target = mapper.target(rotated_controller)

    np.testing.assert_allclose(target[:3, :3], ee_rotation)


def test_custom_xrt_axis_mapping_is_applied_to_translation() -> None:
    axes = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mapper = RelativePoseMapper(vr_to_world_axes=axes)
    initial = controller_pose_from_xrt([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    mapper.align(initial, np.eye(4))

    forward = mapper.target(
        controller_pose_from_xrt([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    )
    right = mapper.target(
        controller_pose_from_xrt([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    )
    up = mapper.target(
        controller_pose_from_xrt([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    )

    np.testing.assert_allclose(forward[:3, 3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(right[:3, 3], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(up[:3, 3], [0.0, 0.0, 1.0])


def test_xrt_pose_requires_nonzero_quaternion() -> None:
    with pytest.raises(ValueError, match="zero quaternion"):
        controller_pose_from_xrt([0.0] * 7)


def test_relative_pose_mapper_requires_realign_after_reset() -> None:
    mapper = RelativePoseMapper()
    controller_pose = controller_pose_from_xrt([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    mapper.align(controller_pose, np.eye(4))
    mapper.reset()

    assert not mapper.aligned
    with pytest.raises(RuntimeError, match="aligned"):
        mapper.target(controller_pose)


def test_pose_target_filter_limits_linear_velocity() -> None:
    pose_filter = PoseTargetFilter(
        update_period=0.02,
        position_time_constant=0.0,
        orientation_time_constant=0.0,
        max_linear_speed=0.5,
        max_angular_speed=2.0,
    )
    pose_filter.reset(np.eye(4))
    target = np.eye(4)
    target[0, 3] = 1.0

    filtered = pose_filter.update(target)

    np.testing.assert_allclose(filtered[:3, 3], [0.01, 0.0, 0.0])


def test_pose_target_filter_limits_angular_velocity() -> None:
    pose_filter = PoseTargetFilter(
        update_period=0.1,
        position_time_constant=0.0,
        orientation_time_constant=0.0,
        max_linear_speed=1.0,
        max_angular_speed=1.0,
    )
    pose_filter.reset(np.eye(4))
    target = make_pose([0.0, 0.0, 0.0], np.diag([-1.0, -1.0, 1.0]))

    filtered = pose_filter.update(target)

    expected = np.array(
        [
            [np.cos(0.1), -np.sin(0.1), 0.0],
            [np.sin(0.1), np.cos(0.1), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(filtered[:3, :3], expected, atol=1e-7)
