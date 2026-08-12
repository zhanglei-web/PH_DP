from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import struct
import time

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TwistStamped
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image as PilImage
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState, Joy
from std_msgs.msg import Bool, Empty, Float64, Header, String, UInt8

from mujoco_shared_control import PickPlaceEnv
from mujoco_shared_control.data.recording import (
    EpisodeRecorder,
    FramePayload,
    RenderedFrame,
    StageTracker,
    TeleopSnapshot,
    build_state_26,
)
from mujoco_shared_control.robots.franka import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES
from mujoco_shared_control.sensors.camera import CameraSensor
from mujoco_shared_control.utils.pose import matrix_to_quaternion, quaternion_to_matrix


VALID_UPDATE_RATES = (10, 20)
PHYSICS_CONTROL_TIMESTEP = 0.01
VR_RAW_AXES_SIZE = 12
HUMAN_COMMAND_AXES_SIZE = 12
POLICY_OUTPUT_AXES_SIZE = 13


def _time_message(seconds: float) -> Time:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec == 1_000_000_000:
        sec += 1
        nanosec = 0
    return Time(sec=sec, nanosec=nanosec)


def _message_time_seconds(stamp: Time) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _pose_message(pose: np.ndarray, stamp: Time, frame_id: str) -> PoseStamped:
    quaternion_wxyz = matrix_to_quaternion(pose[:3, :3])
    message = PoseStamped()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.pose.position.x = float(pose[0, 3])
    message.pose.position.y = float(pose[1, 3])
    message.pose.position.z = float(pose[2, 3])
    message.pose.orientation.w = float(quaternion_wxyz[0])
    message.pose.orientation.x = float(quaternion_wxyz[1])
    message.pose.orientation.y = float(quaternion_wxyz[2])
    message.pose.orientation.z = float(quaternion_wxyz[3])
    return message


def _pose_matrix(message: PoseStamped) -> np.ndarray:
    if message.header.frame_id not in {"", "world"}:
        raise ValueError("ee_pose_command must be expressed in the world frame")
    pose = message.pose
    position = np.array([pose.position.x, pose.position.y, pose.position.z])
    quaternion_xyzw = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    if not np.isfinite(position).all() or not np.isfinite(quaternion_xyzw).all():
        raise ValueError("ee_pose_command must contain finite values")
    norm = np.linalg.norm(quaternion_xyzw)
    if norm < 1e-8:
        raise ValueError("ee_pose_command orientation cannot be the zero quaternion")
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]] / norm
    target = np.eye(4, dtype=np.float64)
    target[:3, :3] = quaternion_to_matrix(quaternion_wxyz)
    target[:3, 3] = position
    return target


class MujocoPickPlaceBridge(Node):
    """Owns one MuJoCo environment and exposes commands and sensors over ROS 2."""

    def __init__(self) -> None:
        super().__init__("pick_place_bridge")
        self.declare_parameter("update_rate_hz", 20)
        self.declare_parameter("camera_name", "front")
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("publish_raw_images", False)
        self.declare_parameter("publish_compressed", True)
        self.declare_parameter("show_mujoco_viewer", True)
        self.declare_parameter("randomize_object", True)
        self.declare_parameter("randomize_goal", True)
        self.declare_parameter("dataset_dir", str(Path.cwd() / "datasets"))
        self.declare_parameter("task_name", "pick_place")

        self._camera_name = str(self.get_parameter("camera_name").value)
        self._camera_frame = f"{self._camera_name}_camera_optical_frame"
        self._publish_compressed = bool(
            self.get_parameter("publish_compressed").value
        )
        self._publish_raw_images = bool(
            self.get_parameter("publish_raw_images").value
        )
        if not self._publish_raw_images and not self._publish_compressed:
            raise ValueError(
                "At least one of publish_raw_images or publish_compressed must be true"
            )
        width = int(self.get_parameter("camera_width").value)
        height = int(self.get_parameter("camera_height").value)
        self._randomize_object = bool(
            self.get_parameter("randomize_object").value
        )
        self._randomize_goal = bool(self.get_parameter("randomize_goal").value)

        self.env = PickPlaceEnv(
            control_timestep=PHYSICS_CONTROL_TIMESTEP,
            max_episode_steps=2_000_000_000,
            camera_width=width,
            camera_height=height,
            enable_camera=False,
        )
        self.env.reset(
            seed=0,
            options={
                "randomize_object": self._randomize_object,
                "randomize_goal": self._randomize_goal,
            },
        )
        self._q_command = self.env.home_joint_positions
        self._gripper_command = 0.08
        self._pending_ee_target: np.ndarray | None = None
        self._latest_user_target: np.ndarray | None = None
        self._latest_user_target_source_time = float("nan")
        self._latest_user_target_receipt_ns = 0
        self._latest_gripper_receipt_ns = 0
        self._latest_human_command = np.full(8, np.nan, dtype=np.float64)
        self._latest_human_command_valid = False
        self._latest_human_command_source_time = float("nan")
        self._latest_human_command_receipt_ns = 0
        self._latest_policy_output = np.full(8, np.nan, dtype=np.float64)
        self._latest_policy_output_valid = False
        self._latest_policy_output_command_space = 0
        self._latest_policy_output_confidence = 0.0
        self._latest_policy_output_control_active = False
        self._latest_policy_output_fallback_used = False
        self._latest_policy_output_source_time = float("nan")
        self._latest_policy_output_receipt_ns = 0
        self._latest_vr_raw = np.full(9, np.nan, dtype=np.float64)
        self._latest_vr_valid = False
        self._latest_vr_aligned = False
        self._latest_vr_control_orientation = True
        self._latest_vr_source_time = float("nan")
        self._latest_vr_receipt_ns = 0
        self._policy_step_index = 0
        self._last_command_status = {
            "ik_success": True,
            "command_accepted": True,
            "action_clipped": False,
            "fallback_used": False,
            "rejection_reason": "",
        }
        self._task_name = str(self.get_parameter("task_name").value)
        self._recorder = EpisodeRecorder(
            str(self.get_parameter("dataset_dir").value)
        )
        self._stage_tracker = StageTracker()
        self._viewer = None
        self._camera_width = width
        self._camera_height = height
        self._camera_data: mujoco.MjData | None = None
        self._camera_sensor_instance: CameraSensor | None = None
        self._shutting_down = False
        self._camera_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rgbd_camera"
        )
        self._camera_futures: list[Future[None]] = []
        if bool(self.get_parameter("show_mujoco_viewer").value):
            self._viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)

        self._joint_state_pub = self.create_publisher(
            JointState, "joint_states", qos_profile_sensor_data
        )
        self._gripper_state_pub = self.create_publisher(
            JointState, "gripper/state", qos_profile_sensor_data
        )
        self._gripper_opening_pub = self.create_publisher(
            Float64, "gripper/opening", qos_profile_sensor_data
        )
        self._ee_pose_pub = self.create_publisher(
            PoseStamped, "ee_pose", qos_profile_sensor_data
        )
        self._object_pose_pub = self.create_publisher(
            PoseStamped, "object/pose", qos_profile_sensor_data
        )
        self._goal_pose_pub = self.create_publisher(
            PoseStamped, "goal/pose", qos_profile_sensor_data
        )
        self._object_twist_pub = self.create_publisher(
            TwistStamped, "object/twist", qos_profile_sensor_data
        )
        self._object_grasped_pub = self.create_publisher(
            Bool, "object/grasped", qos_profile_sensor_data
        )
        self._clock_pub = self.create_publisher(Clock, "/clock", 10)
        self._active_rate_pub = self.create_publisher(
            UInt8, "active_update_rate", 10
        )
        self._collection_status_pub = self.create_publisher(
            String, "collection/status", 10
        )
        self._reset_event_pub = self.create_publisher(Empty, "reset_event", 10)
        self._policy_state_pub = self.create_publisher(
            Joy, "shared_control/state_26", qos_profile_sensor_data
        )
        self._executed_action_pub = self.create_publisher(
            Joy, "shared_control/executed_action", qos_profile_sensor_data
        )

        camera_prefix = f"camera/{self._camera_name}"
        self._color_pub = None
        self._depth_pub = None
        if self._publish_raw_images:
            self._color_pub = self.create_publisher(
                Image, f"{camera_prefix}/color/image_raw", qos_profile_sensor_data
            )
            self._depth_pub = self.create_publisher(
                Image, f"{camera_prefix}/depth/image_raw", qos_profile_sensor_data
            )
        self._color_info_pub = self.create_publisher(
            CameraInfo, f"{camera_prefix}/color/camera_info", qos_profile_sensor_data
        )
        self._depth_info_pub = self.create_publisher(
            CameraInfo, f"{camera_prefix}/depth/camera_info", qos_profile_sensor_data
        )
        self._color_compressed_pub = None
        self._depth_compressed_pub = None
        if self._publish_compressed:
            self._color_compressed_pub = self.create_publisher(
                CompressedImage,
                f"{camera_prefix}/color/image_raw/compressed",
                10,
            )
            self._depth_compressed_pub = self.create_publisher(
                CompressedImage,
                f"{camera_prefix}/depth/image_raw/compressedDepth",
                10,
            )

        self.create_subscription(
            JointState, "joint_position_command", self._on_joint_command, 10
        )
        self.create_subscription(
            PoseStamped, "ee_pose_command", self._on_ee_pose_command, 10
        )
        self.create_subscription(
            Float64, "gripper_command", self._on_gripper_command, 10
        )
        self.create_subscription(UInt8, "update_rate_command", self._on_rate_command, 10)
        self.create_subscription(Empty, "reset", self._on_reset, 10)
        self.create_subscription(
            Joy, "vr/input_raw", self._on_vr_input, qos_profile_sensor_data
        )
        self.create_subscription(
            Joy,
            "shared_control/human_command",
            self._on_human_command,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Joy,
            "shared_control/policy_output",
            self._on_policy_output,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "collection/command", self._on_collection_command, 10
        )

        self._timer = None
        self._update_rate_hz = 0
        self._physics_steps_per_update = 0
        self.add_on_set_parameters_callback(self._on_parameter_change)
        initial_rate = int(self.get_parameter("update_rate_hz").value)
        self._set_update_rate(initial_rate)
        self.get_logger().info(
            f"Ready at {initial_rate} Hz; camera={self._camera_name} "
            f"{width}x{height}; raw={self._publish_raw_images}; "
            f"compressed={self._publish_compressed}"
        )

    def _set_update_rate(self, rate_hz: int) -> None:
        if rate_hz not in VALID_UPDATE_RATES:
            raise ValueError(f"update_rate_hz must be one of {VALID_UPDATE_RATES}")
        if self._timer is not None:
            self.destroy_timer(self._timer)
        self._update_rate_hz = rate_hz
        self._physics_steps_per_update = int(
            round((1.0 / rate_hz) / self.env.control_timestep)
        )
        self._timer = self.create_timer(1.0 / rate_hz, self._on_update)

    def _on_parameter_change(self, parameters: list[Parameter]) -> SetParametersResult:
        for parameter in parameters:
            if parameter.name == "update_rate_hz":
                try:
                    rate_hz = int(parameter.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False, reason="update_rate_hz must be 10 or 20"
                    )
                if rate_hz not in VALID_UPDATE_RATES:
                    return SetParametersResult(
                        successful=False, reason="update_rate_hz must be 10 or 20"
                    )
                self._set_update_rate(rate_hz)
                self.get_logger().info(f"Update rate changed to {rate_hz} Hz")
            elif parameter.name == "randomize_object":
                self._randomize_object = bool(parameter.value)
            elif parameter.name == "randomize_goal":
                self._randomize_goal = bool(parameter.value)
        return SetParametersResult(successful=True)

    def _on_rate_command(self, message: UInt8) -> None:
        rate_hz = int(message.data)
        if rate_hz not in VALID_UPDATE_RATES:
            self.get_logger().error(
                f"Rejected update rate {rate_hz}; expected 10 or 20"
            )
            return
        self.set_parameters([Parameter("update_rate_hz", value=rate_hz)])

    def _on_joint_command(self, message: JointState) -> None:
        if message.name:
            positions_by_name = dict(zip(message.name, message.position))
            missing = [name for name in ARM_JOINT_NAMES if name not in positions_by_name]
            if missing:
                self.get_logger().error(
                    f"Joint command is missing names: {', '.join(missing)}"
                )
                return
            command = np.asarray(
                [positions_by_name[name] for name in ARM_JOINT_NAMES],
                dtype=np.float64,
            )
        else:
            command = np.asarray(message.position, dtype=np.float64)
        if command.shape != (7,) or not np.isfinite(command).all():
            self.get_logger().error("Joint command must contain 7 finite positions")
            return
        clipped = np.clip(
            command,
            self.env.robot.arm_joint_limits[:, 0],
            self.env.robot.arm_joint_limits[:, 1],
        )
        self._q_command = clipped
        self._pending_ee_target = None
        self._last_command_status = {
            "ik_success": True,
            "command_accepted": True,
            "action_clipped": not np.array_equal(command, clipped),
            "fallback_used": False,
            "rejection_reason": "",
        }

    def _on_gripper_command(self, message: Float64) -> None:
        if not np.isfinite(message.data):
            self.get_logger().error("Gripper command must be finite")
            return
        clipped = float(np.clip(message.data, 0.0, 0.08))
        self._gripper_command = clipped
        self._latest_gripper_receipt_ns = time.monotonic_ns()
        if clipped != float(message.data):
            self._last_command_status["action_clipped"] = True

    def _on_ee_pose_command(self, message: PoseStamped) -> None:
        try:
            target = _pose_matrix(message)
            self._pending_ee_target = target
            self._latest_user_target = target.copy()
            self._latest_user_target_source_time = _message_time_seconds(
                message.header.stamp
            )
            self._latest_user_target_receipt_ns = time.monotonic_ns()
        except ValueError as error:
            self.get_logger().error(f"Rejected ee_pose_command: {error}")
            self._pending_ee_target = None

    def _on_vr_input(self, message: Joy) -> None:
        values = np.asarray(message.axes, dtype=np.float64)
        if values.shape != (VR_RAW_AXES_SIZE,):
            self.get_logger().error(
                f"VR raw input must contain {VR_RAW_AXES_SIZE} axes"
            )
            return
        valid = bool(values[9])
        if valid and not np.isfinite(values[:9]).all():
            self.get_logger().error("Valid VR raw input contains non-finite values")
            valid = False
        self._latest_vr_raw = values[:9].copy()
        self._latest_vr_valid = valid
        self._latest_vr_aligned = bool(values[10])
        self._latest_vr_control_orientation = bool(values[11])
        self._latest_vr_source_time = _message_time_seconds(message.header.stamp)
        self._latest_vr_receipt_ns = time.monotonic_ns()

    def _on_human_command(self, message: Joy) -> None:
        values = np.asarray(message.axes, dtype=np.float64)
        if values.shape != (HUMAN_COMMAND_AXES_SIZE,):
            self.get_logger().error(
                f"Human command must contain {HUMAN_COMMAND_AXES_SIZE} axes"
            )
            return
        valid = bool(values[9])
        if valid and not np.isfinite(values[:8]).all():
            self.get_logger().error("Valid human command contains non-finite values")
            valid = False
        self._latest_human_command = values[:8].copy()
        self._latest_human_command_valid = valid
        self._latest_human_command_source_time = _message_time_seconds(
            message.header.stamp
        )
        self._latest_human_command_receipt_ns = time.monotonic_ns()

    def _on_policy_output(self, message: Joy) -> None:
        values = np.asarray(message.axes, dtype=np.float64)
        if values.shape != (POLICY_OUTPUT_AXES_SIZE,):
            self.get_logger().error(
                f"Policy output must contain {POLICY_OUTPUT_AXES_SIZE} axes"
            )
            return
        valid = bool(values[8])
        if valid and not np.isfinite(values[:12]).all():
            self.get_logger().error("Valid policy output contains non-finite values")
            valid = False
        self._latest_policy_output = values[:8].copy()
        self._latest_policy_output_valid = valid
        self._latest_policy_output_confidence = float(values[9])
        self._latest_policy_output_control_active = bool(values[10])
        self._latest_policy_output_command_space = int(round(values[11]))
        self._latest_policy_output_fallback_used = bool(values[12])
        self._latest_policy_output_source_time = _message_time_seconds(
            message.header.stamp
        )
        self._latest_policy_output_receipt_ns = time.monotonic_ns()

    def _solve_latest_ee_target(self) -> None:
        if self._pending_ee_target is None:
            return
        target_pose = self._pending_ee_target
        self._pending_ee_target = None
        result = self.env.ik_controller.inverse_kinematics(
            target_pose, initial_guess=self._q_command
        )
        if not result.converged:
            self._last_command_status = {
                "ik_success": False,
                "command_accepted": False,
                "action_clipped": False,
                "fallback_used": True,
                "rejection_reason": "ik_not_converged",
            }
            self.get_logger().error(
                "Rejected ee_pose_command: Pinocchio IK did not converge "
                f"(position={result.position_error:.4f} m, "
                f"orientation={result.orientation_error:.4f} rad)",
                throttle_duration_sec=2.0,
            )
            return
        self._q_command = result.joint_positions
        self._last_command_status = {
            "ik_success": True,
            "command_accepted": True,
            "action_clipped": False,
            "fallback_used": False,
            "rejection_reason": "",
        }
        self.get_logger().debug(
            "Accepted ee_pose_command via Pinocchio IK "
            f"({result.iterations} iterations, "
            f"position={result.position_error:.5f} m, "
            f"orientation={result.orientation_error:.5f} rad)"
        )

    def _reset_environment(self) -> None:
        _, reset_info = self.env.reset(
            options={
                "randomize_object": self._randomize_object,
                "randomize_goal": self._randomize_goal,
            }
        )
        self._q_command = self.env.home_joint_positions
        self._gripper_command = 0.08
        self._pending_ee_target = None
        self._latest_user_target = None
        self._latest_user_target_receipt_ns = 0
        self._latest_gripper_receipt_ns = 0
        self._latest_human_command[:] = np.nan
        self._latest_human_command_valid = False
        self._latest_human_command_receipt_ns = 0
        self._latest_policy_output[:] = np.nan
        self._latest_policy_output_valid = False
        self._latest_policy_output_receipt_ns = 0
        self._policy_step_index = 0
        self._stage_tracker.reset()
        object_xy = reset_info["object_xy"]
        goal_xy = reset_info["goal_xy"]
        self.get_logger().info(
            "Environment reset: "
            f"object=({object_xy[0]:.3f}, {object_xy[1]:.3f}), "
            f"goal=({goal_xy[0]:.3f}, {goal_xy[1]:.3f})"
        )

    def _on_reset(self, _: Empty) -> None:
        if self._recorder.recording:
            self.get_logger().error(
                "Reset rejected while recording; use collection save or discard"
            )
            return
        self._reset_environment()

    def _publish_collection_status(self, text: str) -> None:
        self._collection_status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def _on_collection_command(self, message: String) -> None:
        command_line = message.data.strip()
        if not command_line:
            return
        command, _, argument = command_line.partition(" ")
        command = command.lower()
        try:
            if command == "task":
                if self._recorder.recording:
                    raise RuntimeError("cannot change task name while recording")
                if not argument.strip():
                    raise ValueError("usage: task <name>")
                self._task_name = argument.strip()
                self._publish_collection_status(f"task set to {self._task_name}")
            elif command == "start":
                if self._update_rate_hz != 20:
                    raise RuntimeError("dataset collection requires update_rate_hz=20")
                episode_id = self._recorder.start(self._task_name)
                self._stage_tracker.reset()
                self._publish_collection_status(f"recording started: {episode_id}")
            elif command in {"save", "discard"}:
                token = self._recorder.stop()
                discard = command == "discard"
                future = self._camera_executor.submit(
                    self._finalize_episode, token, discard
                )
                self._camera_futures.append(future)
                self._reset_environment()
                self._reset_event_pub.publish(Empty())
                verb = "discarding" if discard else "saving"
                self._publish_collection_status(
                    f"{verb} {token.episode_id}; environment reset"
                )
            elif command == "finish":
                if self._recorder.recording:
                    raise RuntimeError("save or discard the active episode first")
                future = self._camera_executor.submit(self._finish_collection_task)
                self._camera_futures.append(future)
                self._publish_collection_status(
                    f"finishing task {self._task_name}; waiting for pending writes"
                )
            else:
                raise ValueError(
                    "expected: task <name>, start, save, discard, or finish"
                )
        except (RuntimeError, ValueError) as error:
            self._publish_collection_status(f"collection command rejected: {error}")

    def _finalize_episode(self, token, discard: bool) -> None:
        result = self._recorder.finalize(token, discard=discard)
        if discard:
            self._publish_collection_status(f"episode discarded: {token.episode_id}")
        else:
            self._publish_collection_status(
                f"episode finalized: {result.get('path')}; valid={result['valid']}"
            )

    def _finish_collection_task(self) -> None:
        self._recorder.wait_idle()
        self._publish_collection_status(f"task complete: {self._task_name}")

    @staticmethod
    def _age_ms(sample_ns: int, receipt_ns: int) -> float:
        if receipt_ns <= 0:
            return float("inf")
        return max(0.0, (sample_ns - receipt_ns) / 1_000_000.0)

    def _teleop_snapshot(self, sample_ns: int) -> TeleopSnapshot:
        raw_age_ms = self._age_ms(sample_ns, self._latest_vr_receipt_ns)
        pose_age_ms = self._age_ms(
            sample_ns, self._latest_user_target_receipt_ns
        )
        gripper_age_ms = self._age_ms(
            sample_ns, self._latest_gripper_receipt_ns
        )
        user_age_ms = max(pose_age_ms, gripper_age_ms)
        human_age_ms = self._age_ms(
            sample_ns, self._latest_human_command_receipt_ns
        )
        policy_age_ms = self._age_ms(
            sample_ns, self._latest_policy_output_receipt_ns
        )
        if self._latest_human_command_receipt_ns > 0:
            user_command = self._latest_human_command.copy()
            user_valid = bool(
                self._latest_human_command_valid and human_age_ms <= 250.0
            )
            user_source_timestamp = self._latest_human_command_source_time
            user_age_ms = human_age_ms
        else:
            if self._latest_user_target is None:
                target_vector = np.full(7, np.nan, dtype=np.float64)
            else:
                target_vector = np.concatenate(
                    (
                        self._latest_user_target[:3, 3],
                        matrix_to_quaternion(self._latest_user_target[:3, :3]),
                    )
                )
            user_command = np.concatenate((target_vector, [self._gripper_command]))
            user_valid = bool(
                self._latest_vr_valid
                and self._latest_vr_aligned
                and self._latest_user_target is not None
                and user_age_ms <= 250.0
            )
            user_source_timestamp = self._latest_user_target_source_time
        return TeleopSnapshot(
            raw=self._latest_vr_raw.copy(),
            raw_valid=bool(self._latest_vr_valid and raw_age_ms <= 250.0),
            aligned=self._latest_vr_aligned,
            raw_source_timestamp=self._latest_vr_source_time,
            raw_age_ms=raw_age_ms,
            user_command=user_command,
            user_command_valid=user_valid,
            user_command_source_timestamp=user_source_timestamp,
            user_command_age_ms=user_age_ms,
            control_orientation=self._latest_vr_control_orientation,
            policy_output=self._latest_policy_output.copy(),
            policy_output_valid=bool(
                self._latest_policy_output_valid and policy_age_ms <= 250.0
            ),
            policy_output_command_space=self._latest_policy_output_command_space,
            policy_output_confidence=self._latest_policy_output_confidence,
            policy_output_control_active=self._latest_policy_output_control_active,
            policy_output_fallback_used=self._latest_policy_output_fallback_used,
            policy_output_source_timestamp=self._latest_policy_output_source_time,
            policy_output_age_ms=policy_age_ms,
        )

    def _on_update(self) -> None:
        started = time.perf_counter()
        sample_monotonic_ns = time.monotonic_ns()
        observation = self.env.get_observation()
        self._solve_latest_ee_target()
        action = np.concatenate((self._q_command, [self._gripper_command]))
        state_26 = build_state_26(observation)
        stamp = _time_message(float(observation["timestamp"][0]))
        self._publish_state(observation, stamp)
        self._publish_shared_control_input(state_26, action, stamp)

        completed_futures = [future for future in self._camera_futures if future.done()]
        self._camera_futures = [
            future for future in self._camera_futures if not future.done()
        ]
        for future in completed_futures:
            error = future.exception()
            if error is not None:
                self.get_logger().error(f"Camera worker failed: {error}")

        teleop = self._teleop_snapshot(sample_monotonic_ns)
        reward, success, _ = self.env.task.evaluate(observation)
        stage, events = self._stage_tracker.update(observation, teleop, success)
        identity = self._recorder.reserve_step()
        payload = None
        if identity is not None:
            status = self._last_command_status
            payload = FramePayload(
                identity=identity,
                simulation_timestamp=float(observation["timestamp"][0]),
                sample_monotonic_ns=sample_monotonic_ns,
                observation=observation,
                state_26=state_26,
                policy_state_42=self.env.get_policy_observation(observation),
                teleop=teleop,
                executed_action=action.copy(),
                mujoco_ctrl=np.concatenate(
                    (self._q_command, [self._gripper_command / 2.0])
                ),
                ik_success=bool(status["ik_success"]),
                command_accepted=bool(status["command_accepted"]),
                action_clipped=bool(status["action_clipped"]),
                fallback_used=bool(status["fallback_used"]),
                rejection_reason=str(status["rejection_reason"]),
                reward=float(reward),
                task_success=bool(success),
                stage=stage,
                events=events,
            )

        future = self._camera_executor.submit(
            self._render_and_publish_camera,
            self.env.data.qpos.copy(),
            self.env.data.mocap_pos.copy(),
            self.env.data.mocap_quat.copy(),
            float(observation["timestamp"][0]),
            payload,
        )
        self._camera_futures.append(future)

        for _ in range(self._physics_steps_per_update):
            self.env.step(action)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

        elapsed = time.perf_counter() - started
        if elapsed > 1.0 / self._update_rate_hz:
            self.get_logger().warning(
                f"Update took {elapsed * 1000.0:.1f} ms, slower than "
                f"{self._update_rate_hz} Hz target",
                throttle_duration_sec=2.0,
            )

    def _publish_shared_control_input(
        self, state_26: np.ndarray, action: np.ndarray, stamp: Time
    ) -> None:
        state_message = Joy()
        state_message.header = Header(stamp=stamp, frame_id="world")
        state_message.axes = state_26.astype(np.float32, copy=False).tolist()
        state_message.buttons = [self._policy_step_index]
        self._policy_state_pub.publish(state_message)

        action_message = Joy()
        action_message.header = state_message.header
        action_message.axes = action.astype(np.float32, copy=False).tolist()
        action_message.buttons = [self._policy_step_index]
        self._executed_action_pub.publish(action_message)
        self._policy_step_index += 1

    def _publish_state(self, observation: dict, stamp: Time) -> None:
        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.header.frame_id = "world"
        joint_state.name = [*ARM_JOINT_NAMES, *GRIPPER_JOINT_NAMES]
        joint_state.position = [
            *observation["q_obs"].tolist(),
            *observation["gripper_joint_positions"].tolist(),
        ]
        joint_state.velocity = [
            *observation["dq_obs"].tolist(),
            *observation["gripper_joint_velocities"].tolist(),
        ]
        self._joint_state_pub.publish(joint_state)

        gripper_state = JointState()
        gripper_state.header = joint_state.header
        gripper_state.name = list(GRIPPER_JOINT_NAMES)
        gripper_state.position = observation["gripper_joint_positions"].tolist()
        gripper_state.velocity = observation["gripper_joint_velocities"].tolist()
        self._gripper_state_pub.publish(gripper_state)
        self._gripper_opening_pub.publish(
            Float64(data=float(observation["gripper"][0]))
        )

        self._ee_pose_pub.publish(
            _pose_message(observation["ee_pose"], stamp, "world")
        )
        self._object_pose_pub.publish(
            _pose_message(observation["object_pose"], stamp, "world")
        )
        self._goal_pose_pub.publish(
            _pose_message(observation["goal_pose"], stamp, "world")
        )

        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = "world"
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = map(
            float, observation["object_linear_velocity"]
        )
        twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z = map(
            float, observation["object_angular_velocity"]
        )
        self._object_twist_pub.publish(twist)
        self._object_grasped_pub.publish(
            Bool(data=bool(observation["object_grasped"]))
        )
        self._clock_pub.publish(Clock(clock=stamp))
        self._active_rate_pub.publish(UInt8(data=self._update_rate_hz))

    def _camera_info(self, stamp: Time) -> CameraInfo:
        if self._camera_sensor_instance is None:
            raise RuntimeError("Camera sensor is not initialized")
        calibration = self._camera_sensor_instance.get_calibration(self._camera_name)
        intrinsic = calibration.intrinsic_matrix
        message = CameraInfo()
        message.header.stamp = stamp
        message.header.frame_id = self._camera_frame
        message.width = calibration.width
        message.height = calibration.height
        message.distortion_model = "plumb_bob"
        message.d = [0.0] * 5
        message.k = intrinsic.reshape(-1).tolist()
        message.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()
        message.p = [
            intrinsic[0, 0], 0.0, intrinsic[0, 2], 0.0,
            0.0, intrinsic[1, 1], intrinsic[1, 2], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return message

    def _publish_camera(self, rgb: np.ndarray, depth: np.ndarray, stamp: Time) -> None:
        header = Header(stamp=stamp, frame_id=self._camera_frame)
        if self._color_pub is not None:
            color_message = Image()
            color_message.header = header
            color_message.height, color_message.width = rgb.shape[:2]
            color_message.encoding = "rgb8"
            color_message.is_bigendian = 0
            color_message.step = color_message.width * 3
            color_message.data = np.ascontiguousarray(rgb).tobytes()
            self._color_pub.publish(color_message)

        if self._depth_pub is not None:
            depth_le = np.ascontiguousarray(depth.astype("<f4", copy=False))
            depth_message = Image()
            depth_message.header = header
            depth_message.height, depth_message.width = depth_le.shape
            depth_message.encoding = "32FC1"
            depth_message.is_bigendian = 0
            depth_message.step = depth_message.width * 4
            depth_message.data = depth_le.tobytes()
            self._depth_pub.publish(depth_message)

        camera_info = self._camera_info(stamp)
        self._color_info_pub.publish(camera_info)
        self._depth_info_pub.publish(camera_info)

        if self._color_compressed_pub is not None:
            self._compress_and_publish(rgb, depth, header)

    def _render_and_publish_camera(
        self,
        qpos_snapshot: np.ndarray,
        mocap_pos_snapshot: np.ndarray,
        mocap_quat_snapshot: np.ndarray,
        simulation_time: float,
        payload: FramePayload | None,
    ) -> None:
        render_start_ns = time.monotonic_ns()
        try:
            if self._camera_sensor_instance is None:
                self._camera_data = mujoco.MjData(self.env.model)
                self._camera_sensor_instance = CameraSensor(
                    self.env.model,
                    self._camera_data,
                    width=self._camera_width,
                    height=self._camera_height,
                )
            assert self._camera_data is not None
            self._camera_data.qpos[:] = qpos_snapshot
            self._camera_data.mocap_pos[:] = mocap_pos_snapshot
            self._camera_data.mocap_quat[:] = mocap_quat_snapshot
            self._camera_data.time = simulation_time
            mujoco.mj_forward(self.env.model, self._camera_data)
            rgb, depth = self._camera_sensor_instance.render_rgbd(self._camera_name)
            calibration = self._camera_sensor_instance.get_calibration(
                self._camera_name
            )
            calibration_dict = {
                "name": calibration.name,
                "width": calibration.width,
                "height": calibration.height,
                "fovy_degrees": calibration.fovy_degrees,
                "intrinsic_matrix": calibration.intrinsic_matrix.tolist(),
                "position_world": calibration.position_world.tolist(),
                "rotation_camera_to_world": (
                    calibration.rotation_camera_to_world.tolist()
                ),
                "near": float(
                    self.env.model.vis.map.znear * self.env.model.stat.extent
                ),
                "far": float(
                    self.env.model.vis.map.zfar * self.env.model.stat.extent
                ),
                "mujoco_camera_axes": "+x right, +y up, view along -z",
                "ros_optical_axes": "+x right, +y down, +z forward",
            }
            render_end_ns = time.monotonic_ns()
            if not self._shutting_down and rclpy.ok():
                self._publish_camera(rgb, depth, _time_message(simulation_time))
            if payload is not None:
                self._recorder.submit_frame(
                    RenderedFrame(
                        payload=payload,
                        rgb=rgb,
                        depth=depth,
                        image_valid=True,
                        drop_reason="",
                        render_start_monotonic_ns=render_start_ns,
                        render_end_monotonic_ns=render_end_ns,
                        camera_calibration=calibration_dict,
                    )
                )
        except Exception as error:
            render_end_ns = time.monotonic_ns()
            if payload is not None:
                self._recorder.submit_frame(
                    RenderedFrame(
                        payload=payload,
                        rgb=np.zeros(
                            (self._camera_height, self._camera_width, 3),
                            dtype=np.uint8,
                        ),
                        depth=np.full(
                            (self._camera_height, self._camera_width),
                            np.nan,
                            dtype=np.float32,
                        ),
                        image_valid=False,
                        drop_reason=f"render_error:{type(error).__name__}",
                        render_start_monotonic_ns=render_start_ns,
                        render_end_monotonic_ns=render_end_ns,
                        camera_calibration={
                            "name": self._camera_name,
                            "width": self._camera_width,
                            "height": self._camera_height,
                            "error": str(error),
                        },
                    )
                )
            raise

    def _compress_and_publish(
        self, rgb: np.ndarray, depth: np.ndarray, header: Header
    ) -> None:
        color_buffer = BytesIO()
        PilImage.fromarray(rgb, mode="RGB").save(
            color_buffer, format="JPEG", quality=90
        )
        self._color_compressed_pub.publish(
            CompressedImage(
                header=header,
                format="rgb8; jpeg compressed rgb8",
                data=color_buffer.getvalue(),
            )
        )

        depth_mm = np.clip(
            np.rint(depth * 1000.0), 0, np.iinfo(np.uint16).max
        ).astype(np.uint16)
        depth_buffer = BytesIO()
        PilImage.fromarray(depth_mm, mode="I;16").save(
            depth_buffer, format="PNG", compress_level=1
        )
        self._depth_compressed_pub.publish(
            CompressedImage(
                header=header,
                format="16UC1; compressedDepth png",
                data=struct.pack("<iff", 0, 0.0, 0.0) + depth_buffer.getvalue(),
            )
        )

    def destroy_node(self) -> bool:
        self._shutting_down = True
        if self._recorder.recording:
            token = self._recorder.stop()
            self._camera_futures.append(
                self._camera_executor.submit(
                    self._finalize_episode, token, True
                )
            )

        def close_camera() -> None:
            if self._camera_sensor_instance is not None:
                self._camera_sensor_instance.close()
                self._camera_sensor_instance = None

        self._camera_futures.append(self._camera_executor.submit(close_camera))
        self._camera_executor.shutdown(wait=True, cancel_futures=False)
        for future in self._camera_futures:
            try:
                future.result()
            except Exception:
                if rclpy.ok():
                    raise
        self._recorder.close()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.env.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MujocoPickPlaceBridge()
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
