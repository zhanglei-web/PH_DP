"""Publish XRobotToolkit right-controller motion as MuJoCo ROS commands."""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty, Float64

from mujoco_shared_control.control.vr_teleop import (
    PoseTargetFilter,
    RelativePoseMapper,
    controller_pose_from_xrt,
)
from mujoco_shared_control.utils.pose import matrix_to_quaternion, quaternion_to_matrix


GRIPPER_OPENING_METERS = 0.08
VR_RAW_AXES_SIZE = 12
HUMAN_COMMAND_AXES_SIZE = 12
CONTROL_MODES = ("direct", "shared")


def _pose_matrix(message: PoseStamped) -> np.ndarray:
    """Convert a ROS pose to a homogeneous transform, validating its quaternion."""
    pose = message.pose
    position = np.array([pose.position.x, pose.position.y, pose.position.z])
    quaternion_xyzw = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    if not np.isfinite(position).all() or not np.isfinite(quaternion_xyzw).all():
        raise ValueError("end-effector pose contains non-finite values")
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm < 1e-8:
        raise ValueError("end-effector pose has a zero quaternion")
    target = np.eye(4, dtype=np.float64)
    target[:3, :3] = quaternion_to_matrix(quaternion_xyzw[[3, 0, 1, 2]] / norm)
    target[:3, 3] = position
    return target


def _pose_message(target: np.ndarray, stamp) -> PoseStamped:
    """Create a world-frame ROS pose message from a homogeneous transform."""
    quaternion_wxyz = matrix_to_quaternion(target[:3, :3])
    message = PoseStamped()
    message.header.stamp = stamp
    message.header.frame_id = "world"
    message.pose.position.x = float(target[0, 3])
    message.pose.position.y = float(target[1, 3])
    message.pose.position.z = float(target[2, 3])
    message.pose.orientation.x = float(quaternion_wxyz[1])
    message.pose.orientation.y = float(quaternion_wxyz[2])
    message.pose.orientation.z = float(quaternion_wxyz[3])
    message.pose.orientation.w = float(quaternion_wxyz[0])
    return message


class XrtTeleopNode(Node):
    """Convert XRT controller deltas into MuJoCo end-effector commands."""

    def __init__(self) -> None:
        super().__init__("xrt_teleop")
        self.declare_parameter("update_rate_hz", 50.0)
        self.declare_parameter("translation_scale", 1.0)
        self.declare_parameter(
            "vr_to_world_axes",
            [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.declare_parameter("control_orientation", True)
        self.declare_parameter("position_smoothing_time_constant", 0.08)
        self.declare_parameter("orientation_smoothing_time_constant", 0.10)
        self.declare_parameter("max_linear_speed", 0.5)
        self.declare_parameter("max_angular_speed", 2.0)
        self.declare_parameter("grip_threshold", 0.5)
        self.declare_parameter("robot_namespace", "mujoco")
        self.declare_parameter("control_mode", "direct")

        rate_hz = float(self.get_parameter("update_rate_hz").value)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("update_rate_hz must be a positive finite number")
        self._grip_threshold = float(self.get_parameter("grip_threshold").value)
        if not np.isfinite(self._grip_threshold) or not 0.0 <= self._grip_threshold <= 1.0:
            raise ValueError("grip_threshold must be a finite value in [0, 1]")
        self._control_mode = str(self.get_parameter("control_mode").value).lower()
        if self._control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}")
        axis_values = np.asarray(self.get_parameter("vr_to_world_axes").value)
        try:
            self._mapper = RelativePoseMapper(
                translation_scale=float(self.get_parameter("translation_scale").value),
                vr_to_world_axes=axis_values.reshape(3, 3),
                control_orientation=bool(
                    self.get_parameter("control_orientation").value
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid teleoperation mapping parameters: {error}") from error
        try:
            self._target_filter = PoseTargetFilter(
                update_period=1.0 / rate_hz,
                position_time_constant=float(
                    self.get_parameter("position_smoothing_time_constant").value
                ),
                orientation_time_constant=float(
                    self.get_parameter("orientation_smoothing_time_constant").value
                ),
                max_linear_speed=float(
                    self.get_parameter("max_linear_speed").value
                ),
                max_angular_speed=float(
                    self.get_parameter("max_angular_speed").value
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid pose filter parameters: {error}") from error

        robot_namespace = str(self.get_parameter("robot_namespace").value).strip("/")
        topic_prefix = f"/{robot_namespace}" if robot_namespace else ""
        self._ee_pose: np.ndarray | None = None
        self._sdk_initialized = False

        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as error:
            raise RuntimeError(
                "xrobotoolkit_sdk is unavailable; add the xrt environment's site-packages "
                "directory to PYTHONPATH"
            ) from error
        self._xrt = xrt
        self._xrt.init()
        self._sdk_initialized = True

        self._ee_pose_sub = self.create_subscription(
            PoseStamped,
            f"{topic_prefix}/ee_pose",
            self._on_ee_pose,
            qos_profile_sensor_data,
        )
        self._reset_sub = self.create_subscription(
            Empty, f"{topic_prefix}/reset", self._on_reset, 10
        )
        self._reset_event_sub = self.create_subscription(
            Empty, f"{topic_prefix}/reset_event", self._on_reset, 10
        )
        self._ee_pose_pub = self.create_publisher(
            PoseStamped, f"{topic_prefix}/ee_pose_command", 10
        )
        self._gripper_pub = self.create_publisher(
            Float64, f"{topic_prefix}/gripper_command", 10
        )
        self._raw_input_pub = self.create_publisher(
            Joy, f"{topic_prefix}/vr/input_raw", qos_profile_sensor_data
        )
        self._human_command_pub = self.create_publisher(
            Joy,
            f"{topic_prefix}/shared_control/human_command",
            qos_profile_sensor_data,
        )
        self._timer = self.create_timer(1.0 / rate_hz, self._on_timer)
        self.get_logger().info(
            f"XRT teleoperation ready at {rate_hz:g} Hz; "
            f"control_mode={self._control_mode}; "
            f"orientation_control={self.get_parameter('control_orientation').value}; "
            f"waiting for {topic_prefix}/ee_pose"
        )

    def _on_ee_pose(self, message: PoseStamped) -> None:
        try:
            self._ee_pose = _pose_matrix(message)
        except ValueError as error:
            self.get_logger().error(f"Ignoring invalid end-effector pose: {error}")

    def _on_reset(self, _: Empty) -> None:
        self._ee_pose = None
        self._mapper.reset()
        self._target_filter.reset()
        self.get_logger().info("Simulation reset; waiting to realign VR control")

    def _on_timer(self) -> None:
        stamp = self.get_clock().now().to_msg()
        try:
            trigger = float(self._xrt.get_right_trigger())
            raw_pose = np.asarray(
                self._xrt.get_right_controller_pose(), dtype=np.float64
            )
            controller_pose = controller_pose_from_xrt(raw_pose)
            grip = float(self._xrt.get_right_grip())
        except Exception as error:
            self._mapper.reset()
            self._target_filter.reset()
            self._publish_raw_input(
                np.full(7, np.nan), np.nan, np.nan, False, stamp
            )
            self._publish_human_command(None, np.nan, False, False, stamp)
            self.get_logger().error(
                f"XRT read failed; waiting to realign: {error}",
                throttle_duration_sec=2.0,
            )
            return

        if not np.isfinite(trigger) or not np.isfinite(grip):
            self._mapper.reset()
            self._target_filter.reset()
            self._publish_raw_input(raw_pose, trigger, grip, False, stamp)
            self._publish_human_command(None, np.nan, False, False, stamp)
            self.get_logger().error("Ignoring non-finite XRT trigger or grip value")
            return

        gripper = Float64()
        gripper.data = GRIPPER_OPENING_METERS * (
            1.0 - np.clip(trigger, 0.0, 1.0)
        )
        if self._control_mode == "direct":
            self._gripper_pub.publish(gripper)

        if grip < self._grip_threshold:
            if self._mapper.aligned:
                self._mapper.reset()
                self._target_filter.reset()
                self.get_logger().info("VR teleoperation released; press grip to realign")
            self._publish_raw_input(raw_pose, trigger, grip, True, stamp)
            self._publish_human_command(
                self._ee_pose,
                gripper.data,
                False,
                self._ee_pose is not None,
                stamp,
            )
            return

        if self._ee_pose is None:
            self._publish_raw_input(raw_pose, trigger, grip, True, stamp)
            self._publish_human_command(None, gripper.data, True, False, stamp)
            self.get_logger().warning(
                "Waiting for the current MuJoCo end-effector pose before alignment",
                throttle_duration_sec=2.0,
            )
            return
        if not self._mapper.aligned:
            self._mapper.align(controller_pose, self._ee_pose)
            self._target_filter.reset(self._ee_pose)
            if self._control_mode == "direct":
                self._ee_pose_pub.publish(_pose_message(self._ee_pose, stamp))
            self._publish_raw_input(raw_pose, trigger, grip, True, stamp)
            self._publish_human_command(
                self._ee_pose, gripper.data, True, True, stamp
            )
            self.get_logger().info("VR controller and end-effector poses aligned")
            return

        target = self._target_filter.update(self._mapper.target(controller_pose))
        if self._control_mode == "direct":
            self._ee_pose_pub.publish(_pose_message(target, stamp))
        self._publish_raw_input(raw_pose, trigger, grip, True, stamp)
        self._publish_human_command(target, gripper.data, True, True, stamp)

    def _publish_human_command(
        self,
        target: np.ndarray | None,
        gripper: float,
        active: bool,
        valid: bool,
        stamp,
    ) -> None:
        """Publish the processed human intent independently of control mode.

        Joy axes are ``xyz, quaternion_wxyz, gripper, active, valid, aligned,
        control_orientation``.  In shared mode this is the only processed VR
        output; the policy node is solely responsible for robot commands.
        """
        if target is None:
            command = np.full(8, np.nan, dtype=np.float64)
            valid = False
        else:
            command = np.concatenate(
                (
                    np.asarray(target[:3, 3], dtype=np.float64),
                    matrix_to_quaternion(target[:3, :3]),
                    [float(gripper)],
                )
            )
            if command.shape != (8,) or not np.isfinite(command).all():
                command = np.full(8, np.nan, dtype=np.float64)
                valid = False
        message = Joy()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        message.axes = [
            *command.tolist(),
            float(bool(active)),
            float(bool(valid)),
            float(self._mapper.aligned),
            float(bool(self.get_parameter("control_orientation").value)),
        ]
        if len(message.axes) != HUMAN_COMMAND_AXES_SIZE:
            raise RuntimeError("Human command schema has an unexpected size")
        self._human_command_pub.publish(message)

    def _publish_raw_input(
        self,
        raw_pose: np.ndarray,
        trigger: float,
        grip: float,
        valid: bool,
        stamp,
    ) -> None:
        """Publish one atomic, stamped VR sample for the dataset recorder.

        Joy axes are ``xyz, quaternion_xyzw, trigger, grip, valid, aligned,
        control_orientation``.  A single message avoids cross-topic skew inside
        one 20 Hz dataset frame.
        """
        pose = np.asarray(raw_pose, dtype=np.float64)
        if pose.shape != (7,):
            pose = np.full(7, np.nan, dtype=np.float64)
            valid = False
        message = Joy()
        message.header.stamp = stamp
        message.header.frame_id = "xrt_tracking"
        message.axes = [
            *pose.tolist(),
            float(trigger),
            float(grip),
            float(bool(valid)),
            float(self._mapper.aligned),
            float(bool(self.get_parameter("control_orientation").value)),
        ]
        if len(message.axes) != VR_RAW_AXES_SIZE:
            raise RuntimeError("VR raw input schema has an unexpected size")
        self._raw_input_pub.publish(message)

    def destroy_node(self) -> bool:
        if self._sdk_initialized:
            self._xrt.close()
            self._sdk_initialized = False
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = XrtTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
