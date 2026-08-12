from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray


def make_pose(position: ArrayLike, rotation_matrix: ArrayLike) -> NDArray[np.float64]:
    """Build a 4x4 homogeneous transform from world-frame components."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return pose


def quaternion_to_matrix(quaternion_wxyz: ArrayLike) -> NDArray[np.float64]:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


def matrix_to_quaternion(rotation_matrix: ArrayLike) -> NDArray[np.float64]:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quaternion, np.asarray(rotation_matrix, dtype=np.float64).reshape(9)
    )
    return quaternion


def pose_to_position_quaternion(
    pose: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    transform = np.asarray(pose, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a (4, 4) pose, got {transform.shape}")
    return transform[:3, 3].copy(), matrix_to_quaternion(transform[:3, :3])

