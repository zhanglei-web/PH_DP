from __future__ import annotations

import time

from geometry_msgs.msg import PoseStamped
import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

from mujoco_shared_control import PickPlaceEnv
from mujoco_shared_control.robots.franka import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES


class MujocoStateViewer(Node):
    """GLFW viewer that mirrors state published by the EGL simulation process."""

    def __init__(self) -> None:
        super().__init__("state_viewer")
        self.env = PickPlaceEnv(enable_camera=False)
        self.env.reset(seed=0, options={"randomize_object": False})
        object_joint_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint"
        )
        self._object_qpos_address = int(self.env.model.jnt_qposadr[object_joint_id])
        self._goal_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )
        self._goal_mocap_id = int(
            self.env.model.body_mocapid[self._goal_body_id]
        )
        self._latest_qpos = self.env.data.qpos.copy()
        self._latest_goal_position = self.env.data.mocap_pos[
            self._goal_mocap_id
        ].copy()
        self._latest_sim_time = 0.0
        self._rendered_sim_time = 0.0
        self._key_reset_requested = False
        self._last_reset_request_time = -np.inf
        self._reset_pub = self.create_publisher(Empty, "reset", 10)
        self._viewer = mujoco.viewer.launch_passive(
            self.env.model, self.env.data, key_callback=self._on_key
        )
        self.create_subscription(
            JointState, "joint_states", self._on_joint_state, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, "object/pose", self._on_object_pose, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, "goal/pose", self._on_goal_pose, qos_profile_sensor_data
        )
        self.create_timer(1.0 / 60.0, self._sync_viewer)
        self.get_logger().info(
            "GLFW state viewer ready; the Reset button or R key resets the ROS simulation"
        )

    def _on_joint_state(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in ARM_JOINT_NAMES):
            self._latest_qpos[self.env.robot.arm_qpos_indices] = [
                positions[name] for name in ARM_JOINT_NAMES
            ]
        if all(name in positions for name in GRIPPER_JOINT_NAMES):
            self._latest_qpos[self.env.robot.gripper_qpos_indices] = [
                positions[name] for name in GRIPPER_JOINT_NAMES
            ]

    def _on_object_pose(self, message: PoseStamped) -> None:
        pose = message.pose
        object_qpos = self._latest_qpos[
            self._object_qpos_address : self._object_qpos_address + 7
        ]
        object_qpos[:3] = [pose.position.x, pose.position.y, pose.position.z]
        object_qpos[3:] = [
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ]
        self._latest_sim_time = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )

    def _on_goal_pose(self, message: PoseStamped) -> None:
        position = message.pose.position
        self._latest_goal_position[:] = [position.x, position.y, position.z]

    def _on_key(self, keycode: int) -> None:
        # GLFW uses 259 for Backspace. R provides a discoverable alternative.
        if keycode in {ord("R"), ord("r"), 259}:
            self._key_reset_requested = True

    def _sync_viewer(self) -> None:
        if not self._viewer.is_running():
            return

        with self._viewer.lock():
            qpos_was_reset = np.allclose(
                self.env.data.qpos, self.env.model.qpos0, atol=1e-12, rtol=0.0
            ) and not np.allclose(
                self._latest_qpos, self.env.model.qpos0, atol=1e-6, rtol=0.0
            )
            time_was_reset = (
                self._rendered_sim_time > 0.05
                and self.env.data.time + 0.05 < self._rendered_sim_time
            )
            viewer_reset_clicked = qpos_was_reset or time_was_reset
            self.env.data.qpos[:] = self._latest_qpos
            self.env.data.time = self._latest_sim_time
            self.env.data.mocap_pos[
                self._goal_mocap_id
            ] = self._latest_goal_position
            self.env.data.mocap_quat[self._goal_mocap_id] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(self.env.model, self.env.data)
            self._rendered_sim_time = self._latest_sim_time

        now = time.monotonic()
        if (
            (viewer_reset_clicked or self._key_reset_requested)
            and now - self._last_reset_request_time >= 0.5
        ):
            self._reset_pub.publish(Empty())
            self._last_reset_request_time = now
            self.get_logger().info("Requested ROS simulation reset from MuJoCo viewer")
        self._key_reset_requested = False
        self._viewer.sync()

    def destroy_node(self) -> bool:
        self._viewer.close()
        self.env.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MujocoStateViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
