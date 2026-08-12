from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from mujoco_shared_control.robots.franka import FrankaRobot
from mujoco_shared_control.utils.pose import make_pose, matrix_to_quaternion


def _require_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Required MuJoCo object '{name}' is missing")
    return object_id


class ObservationReader:
    """Centralizes all direct reads from MjData used by higher-level code."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot: FrankaRobot):
        self.model = model
        self.data = data
        self.robot = robot
        self.object_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.goal_site_id = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self.left_finger_body_id = _require_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self.right_finger_body_id = _require_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )

    def _body_pose(self, body_id: int) -> NDArray[np.float64]:
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.xquat[body_id])
        return make_pose(self.data.xpos[body_id], rotation)

    def _goal_pose(self) -> NDArray[np.float64]:
        return make_pose(
            self.data.site_xpos[self.goal_site_id],
            self.data.site_xmat[self.goal_site_id],
        )

    def _object_velocity(self) -> NDArray[np.float64]:
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.object_body_id,
            velocity,
            0,
        )
        return velocity

    def get_contact_details(self) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        finger_bodies = {
            self.left_finger_body_id: "left",
            self.right_finger_body_id: "right",
        }
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            side = None
            if body1 == self.object_body_id and body2 in finger_bodies:
                side = finger_bodies[body2]
            elif body2 == self.object_body_id and body1 in finger_bodies:
                side = finger_bodies[body1]
            if side is None:
                continue

            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            details.append(
                {
                    "side": side,
                    "position": contact.pos.copy(),
                    "normal": contact.frame[:3].copy(),
                    "distance": float(contact.dist),
                    "normal_force": float(max(force[0], 0.0)),
                    "geom1": mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                    ),
                    "geom2": mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                    ),
                }
            )
        return details

    def _contact_summary(self) -> dict[str, NDArray[np.float32]]:
        details = self.get_contact_details()
        left_forces = [item["normal_force"] for item in details if item["side"] == "left"]
        right_forces = [item["normal_force"] for item in details if item["side"] == "right"]
        return {
            "left": np.array([bool(left_forces)], dtype=np.float32),
            "right": np.array([bool(right_forces)], dtype=np.float32),
            "left_force": np.array([sum(left_forces)], dtype=np.float32),
            "right_force": np.array([sum(right_forces)], dtype=np.float32),
            "count": np.array([len(details)], dtype=np.int32),
        }

    def get_observation(self) -> dict[str, Any]:
        object_pose = self._body_pose(self.object_body_id)
        object_velocity = self._object_velocity()
        contact = self._contact_summary()
        ee_pose = self.robot.get_ee_pose()
        grasped = bool(
            contact["left"][0]
            and contact["right"][0]
            and contact["left_force"][0] > 0.1
            and contact["right_force"][0] > 0.1
            and np.linalg.norm(object_pose[:3, 3] - ee_pose[:3, 3]) < 0.12
        )
        return {
            "q_obs": self.robot.get_joint_positions(),
            "dq_obs": self.robot.get_joint_velocities(),
            "ee_pose": ee_pose,
            "gripper": np.array([self.robot.get_gripper_opening()], dtype=np.float64),
            "gripper_joint_positions": self.robot.get_gripper_joint_positions(),
            "gripper_joint_velocities": self.robot.get_gripper_joint_velocities(),
            "object_pose": object_pose,
            "goal_pose": self._goal_pose(),
            "object_linear_velocity": object_velocity[3:].copy(),
            "object_angular_velocity": object_velocity[:3].copy(),
            "contact": contact,
            "object_grasped": int(grasped),
            "timestamp": np.array([self.data.time], dtype=np.float64),
        }

    def get_policy_observation(self, raw_obs: dict[str, Any] | None = None) -> NDArray[np.float32]:
        obs = self.get_observation() if raw_obs is None else raw_obs

        def pose_vector(pose: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.concatenate((pose[:3, 3], matrix_to_quaternion(pose[:3, :3])))

        vector = np.concatenate(
            (
                obs["q_obs"],
                obs["dq_obs"],
                pose_vector(obs["ee_pose"]),
                obs["gripper"],
                pose_vector(obs["object_pose"]),
                pose_vector(obs["goal_pose"]),
                obs["object_linear_velocity"],
                obs["object_angular_velocity"],
            )
        )
        return vector.astype(np.float32)
