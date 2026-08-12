"""ROS adapter for pluggable, ROS-independent shared-control policies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import BytesIO
import json
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time
from typing import Any

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
import numpy as np
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState, Joy
from std_msgs.msg import Empty as EmptyMessage
from std_msgs.msg import Float64, String

from mujoco_shared_control.shared_control import (
    COMMAND_DIM,
    STATE_DIM,
    CommandSpace,
    HumanPassthroughPolicy,
    ModelInput,
    ModelInputSpec,
    ModelOutput,
    load_policy,
)


HUMAN_COMMAND_AXES_SIZE = 12
POLICY_OUTPUT_AXES_SIZE = 13


def _stamp_key(stamp: Time) -> tuple[int, int]:
    return int(stamp.sec), int(stamp.nanosec)


def _stamp_seconds(stamp: Time) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _time_message(seconds: float) -> Time:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec == 1_000_000_000:
        sec += 1
        nanosec = 0
    return Time(sec=sec, nanosec=nanosec)


@dataclass
class _HumanCommand:
    command: np.ndarray
    valid: bool
    active: bool
    source_timestamp: float
    receipt_ns: int


@dataclass
class _Sample:
    timestamp: float
    step_index: int
    state: np.ndarray
    human_command: np.ndarray
    human_valid: bool
    human_active: bool
    human_source_timestamp: float
    human_age_ms: float
    executed_action: np.ndarray
    rgb: dict[str, np.ndarray]
    depth: dict[str, np.ndarray]


@dataclass(frozen=True)
class _InferenceResult:
    model_input: ModelInput
    output: ModelOutput | None
    duration_ms: float
    error: str


@dataclass(frozen=True)
class _ResetPolicy:
    task_name: str


class SharedControlInferenceNode(Node):
    """Synchronize model inputs and adapt policy results to robot commands."""

    def __init__(self) -> None:
        super().__init__("shared_control_inference")
        self.declare_parameter("policy_plugin", "human_passthrough")
        self.declare_parameter("policy_config_json", "{}")
        self.declare_parameter("task_name", "pick_place")
        self.declare_parameter("inference_timeout_ms", 250.0)
        self.declare_parameter("human_command_timeout_ms", 250.0)

        self._task_name = str(self.get_parameter("task_name").value)
        self._timeout_ms = float(self.get_parameter("inference_timeout_ms").value)
        self._human_timeout_ms = float(
            self.get_parameter("human_command_timeout_ms").value
        )
        if not np.isfinite(self._timeout_ms) or self._timeout_ms <= 0.0:
            raise ValueError("inference_timeout_ms must be positive and finite")
        if not np.isfinite(self._human_timeout_ms) or self._human_timeout_ms <= 0.0:
            raise ValueError("human_command_timeout_ms must be positive and finite")

        plugin_name = str(self.get_parameter("policy_plugin").value)
        try:
            plugin_config = json.loads(
                str(self.get_parameter("policy_config_json").value)
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"policy_config_json is invalid: {error}") from error
        if not isinstance(plugin_config, dict):
            raise ValueError("policy_config_json must contain a JSON object")
        self._policy = load_policy(plugin_name, plugin_config)
        self._spec = self._policy.input_spec
        if not isinstance(self._spec, ModelInputSpec):
            raise TypeError("policy.input_spec must be a ModelInputSpec")
        self._policy.reset(self._task_name)
        self._fallback_policy = HumanPassthroughPolicy()

        self._states: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
        self._actions: dict[tuple[int, int], np.ndarray] = {}
        self._rgb: dict[str, dict[tuple[int, int], np.ndarray]] = {
            camera: {} for camera in self._spec.cameras
        }
        self._depth: dict[str, dict[tuple[int, int], np.ndarray]] = {
            camera: {} for camera in self._spec.cameras
        }
        self._processed_keys: deque[tuple[int, int]] = deque(maxlen=128)
        self._processed_key_set: set[tuple[int, int]] = set()
        self._samples: deque[_Sample] = deque(
            maxlen=max(self._spec.history_length, self._spec.image_history_length)
        )
        self._human = _HumanCommand(
            command=np.full(COMMAND_DIM, np.nan, dtype=np.float32),
            valid=False,
            active=False,
            source_timestamp=float("nan"),
            receipt_ns=0,
        )

        self._work_queue: Queue[ModelInput | _ResetPolicy | None] = Queue(maxsize=1)
        self._result_queue: Queue[_InferenceResult] = Queue()
        self._stop_event = Event()
        self._worker_state_lock = Lock()
        self._active_inference: tuple[int, int, float] | None = None
        self._timeout_reported_for_step: int | None = None
        self._last_valid_output: ModelOutput | None = None
        self._last_error = ""
        self._worker = Thread(
            target=self._inference_worker,
            name="shared_control_policy",
            daemon=True,
        )
        self._worker.start()

        self._pose_pub = self.create_publisher(PoseStamped, "ee_pose_command", 10)
        self._joint_pub = self.create_publisher(
            JointState, "joint_position_command", 10
        )
        self._gripper_pub = self.create_publisher(
            Float64, "gripper_command", 10
        )
        self._output_pub = self.create_publisher(
            Joy, "shared_control/policy_output", qos_profile_sensor_data
        )
        self._status_pub = self.create_publisher(
            String, "shared_control/status", 10
        )

        self.create_subscription(
            Joy,
            "shared_control/state_26",
            self._on_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Joy,
            "shared_control/executed_action",
            self._on_executed_action,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Joy,
            "shared_control/human_command",
            self._on_human_command,
            qos_profile_sensor_data,
        )
        for camera in self._spec.cameras:
            if self._spec.use_rgb:
                self.create_subscription(
                    CompressedImage,
                    f"camera/{camera}/color/image_raw/compressed",
                    lambda message, name=camera: self._on_rgb(name, message),
                    10,
                )
            if self._spec.use_depth:
                self.create_subscription(
                    CompressedImage,
                    f"camera/{camera}/depth/image_raw/compressedDepth",
                    lambda message, name=camera: self._on_depth(name, message),
                    10,
                )
        self.create_subscription(
            EmptyMessage, "reset_event", self._on_reset, 10
        )
        self.create_subscription(
            String, "collection/status", self._on_collection_status, 10
        )
        self._result_timer = self.create_timer(0.01, self._on_result_timer)
        self._publish_status(
            "ready",
            policy=plugin_name,
            input_spec=self._spec.__dict__,
            behavior="invalid_or_timeout_holds_last_valid_command",
        )
        self.get_logger().info(
            f"Shared-control policy ready: {plugin_name}; input_spec={self._spec}"
        )

    def _on_human_command(self, message: Joy) -> None:
        values = np.asarray(message.axes, dtype=np.float32)
        if values.shape != (HUMAN_COMMAND_AXES_SIZE,):
            self._report_error(
                f"human command has {values.size} axes, expected {HUMAN_COMMAND_AXES_SIZE}",
                -1,
            )
            return
        valid = bool(values[9]) and bool(np.isfinite(values[:8]).all())
        self._human = _HumanCommand(
            command=values[:8].copy(),
            valid=valid,
            active=bool(values[8]),
            source_timestamp=_stamp_seconds(message.header.stamp),
            receipt_ns=time.monotonic_ns(),
        )

    def _on_state(self, message: Joy) -> None:
        values = np.asarray(message.axes, dtype=np.float32)
        if values.shape != (STATE_DIM,) or not np.isfinite(values).all():
            self._report_error("state_26 is invalid", -1)
            return
        key = _stamp_key(message.header.stamp)
        step = int(message.buttons[0]) if message.buttons else 0
        self._states[key] = (step, values.copy())
        self._try_assemble(key)

    def _on_executed_action(self, message: Joy) -> None:
        values = np.asarray(message.axes, dtype=np.float32)
        if values.shape != (COMMAND_DIM,) or not np.isfinite(values).all():
            self._report_error("executed action is invalid", -1)
            return
        key = _stamp_key(message.header.stamp)
        self._actions[key] = values.copy()
        self._try_assemble(key)

    def _on_rgb(self, camera: str, message: CompressedImage) -> None:
        try:
            with PilImage.open(BytesIO(bytes(message.data))) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        except Exception as error:
            self._report_error(f"failed to decode {camera} RGB: {error}", -1)
            return
        key = _stamp_key(message.header.stamp)
        self._rgb[camera][key] = rgb
        self._try_assemble(key)

    def _on_depth(self, camera: str, message: CompressedImage) -> None:
        payload = bytes(message.data)
        if len(payload) <= 12:
            self._report_error(f"{camera} depth payload is truncated", -1)
            return
        try:
            with PilImage.open(BytesIO(payload[12:])) as image:
                depth = np.asarray(image, dtype=np.uint16).astype(np.float32) * 0.001
        except Exception as error:
            self._report_error(f"failed to decode {camera} depth: {error}", -1)
            return
        key = _stamp_key(message.header.stamp)
        self._depth[camera][key] = depth
        self._try_assemble(key)

    def _ready(self, key: tuple[int, int]) -> bool:
        if key not in self._states:
            return False
        if self._spec.use_executed_action and key not in self._actions:
            return False
        if self._spec.use_rgb and any(key not in self._rgb[c] for c in self._spec.cameras):
            return False
        if self._spec.use_depth and any(
            key not in self._depth[c] for c in self._spec.cameras
        ):
            return False
        return True

    def _try_assemble(self, key: tuple[int, int]) -> None:
        if key in self._processed_key_set or not self._ready(key):
            return
        step, state = self._states[key]
        now_ns = time.monotonic_ns()
        human_age_ms = (
            float("inf")
            if self._human.receipt_ns <= 0
            else (now_ns - self._human.receipt_ns) / 1_000_000.0
        )
        human_valid = bool(
            self._human.valid and human_age_ms <= self._human_timeout_ms
        )
        sample = _Sample(
            timestamp=float(key[0]) + float(key[1]) * 1e-9,
            step_index=step,
            state=state,
            human_command=self._human.command.copy(),
            human_valid=human_valid,
            human_active=bool(human_valid and self._human.active),
            human_source_timestamp=self._human.source_timestamp,
            human_age_ms=human_age_ms,
            executed_action=self._actions.get(
                key, np.zeros(COMMAND_DIM, dtype=np.float32)
            ),
            rgb={camera: self._rgb[camera][key] for camera in self._spec.cameras}
            if self._spec.use_rgb
            else {},
            depth={camera: self._depth[camera][key] for camera in self._spec.cameras}
            if self._spec.use_depth
            else {},
        )
        self._samples.append(sample)
        try:
            model_input = self._build_model_input()
        except ValueError as error:
            self._report_error(f"failed to assemble model input: {error}", step)
        else:
            self._replace_work(model_input)
        self._remember_processed(key)
        self._remove_key(key)

    def _build_model_input(self) -> ModelInput:
        samples = list(self._samples)[-self._spec.history_length:]
        padding = self._spec.history_length - len(samples)

        def history_array(attribute: str, width: int, fill: float = 0.0) -> np.ndarray:
            values = [getattr(sample, attribute) for sample in samples]
            prefix = [np.full(width, fill, dtype=np.float32) for _ in range(padding)]
            return np.stack([*prefix, *values]).astype(np.float32, copy=False)

        state_history = (
            history_array("state", STATE_DIM)
            if self._spec.use_state_26
            else np.empty((self._spec.history_length, 0), dtype=np.float32)
        )
        human_history = (
            history_array("human_command", COMMAND_DIM, np.nan)
            if self._spec.use_human_action
            else np.empty((self._spec.history_length, 0), dtype=np.float32)
        )
        action_history = (
            history_array("executed_action", COMMAND_DIM)
            if self._spec.use_executed_action
            else np.empty((self._spec.history_length, 0), dtype=np.float32)
        )
        timestamps = np.array(
            [0.0] * padding + [sample.timestamp for sample in samples],
            dtype=np.float64,
        )
        human_timestamps = np.array(
            [np.nan] * padding
            + [sample.human_source_timestamp for sample in samples],
            dtype=np.float64,
        )
        human_age_ms = np.array(
            [np.inf] * padding + [sample.human_age_ms for sample in samples],
            dtype=np.float32,
        )
        history_valid = np.array(
            [False] * padding + [True] * len(samples), dtype=np.bool_
        )
        human_valid = np.array(
            [False] * padding + [sample.human_valid for sample in samples],
            dtype=np.bool_,
        )
        human_active = np.array(
            [False] * padding + [sample.human_active for sample in samples],
            dtype=np.bool_,
        )

        image_samples = list(self._samples)[-self._spec.image_history_length:]
        image_padding = self._spec.image_history_length - len(image_samples)
        rgb: dict[str, np.ndarray] = {}
        depth: dict[str, np.ndarray] = {}
        image_timestamps: dict[str, np.ndarray] = {}
        image_valid: dict[str, np.ndarray] = {}
        for camera in self._spec.cameras:
            reference = image_samples[-1]
            if self._spec.use_rgb:
                shape = reference.rgb[camera].shape
                rgb[camera] = np.stack(
                    [np.zeros(shape, dtype=np.uint8) for _ in range(image_padding)]
                    + [sample.rgb[camera] for sample in image_samples]
                )
            if self._spec.use_depth:
                shape = reference.depth[camera].shape
                depth[camera] = np.stack(
                    [np.zeros(shape, dtype=np.float32) for _ in range(image_padding)]
                    + [sample.depth[camera] for sample in image_samples]
                )
            if self._spec.use_rgb or self._spec.use_depth:
                image_timestamps[camera] = np.array(
                    [0.0] * image_padding
                    + [sample.timestamp for sample in image_samples],
                    dtype=np.float64,
                )
                image_valid[camera] = np.array(
                    [False] * image_padding + [True] * len(image_samples),
                    dtype=np.bool_,
                )

        latest = samples[-1]
        return ModelInput(
            timestamp=latest.timestamp,
            step_index=latest.step_index,
            task_name=self._task_name,
            spec=self._spec,
            state_history=state_history,
            human_action_history=human_history,
            executed_action_history=action_history,
            history_timestamps=timestamps,
            human_action_timestamps=human_timestamps,
            human_action_age_ms=human_age_ms,
            history_valid=history_valid,
            human_action_valid=human_valid,
            human_control_active=human_active,
            rgb=rgb,
            depth=depth,
            image_timestamps=image_timestamps,
            image_valid=image_valid,
        )

    def _replace_work(self, item: ModelInput | _ResetPolicy | None) -> None:
        try:
            self._work_queue.put_nowait(item)
        except Full:
            try:
                self._work_queue.get_nowait()
            except Empty:
                pass
            self._work_queue.put_nowait(item)

    def _inference_worker(self) -> None:
        while not self._stop_event.is_set():
            item = self._work_queue.get()
            if item is None:
                return
            if isinstance(item, _ResetPolicy):
                try:
                    self._policy.reset(item.task_name)
                except Exception as error:
                    self._result_queue.put(
                        _InferenceResult(
                            model_input=self._empty_input_for_error(),
                            output=None,
                            duration_ms=0.0,
                            error=f"policy reset failed: {type(error).__name__}: {error}",
                        )
                    )
                continue
            started = time.monotonic()
            with self._worker_state_lock:
                self._active_inference = (item.step_index, time.monotonic_ns(), item.timestamp)
            output = None
            error_text = ""
            try:
                output = self._policy.predict(item)
                if not isinstance(output, ModelOutput):
                    raise TypeError("predict() must return ModelOutput")
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
            duration_ms = (time.monotonic() - started) * 1000.0
            with self._worker_state_lock:
                self._active_inference = None
            self._result_queue.put(
                _InferenceResult(item, output, duration_ms, error_text)
            )

    def _empty_input_for_error(self) -> ModelInput:
        # Only used to attach a frame identifier to reset errors.
        spec = ModelInputSpec(
            history_length=1,
            use_state_26=False,
            use_human_action=False,
            use_executed_action=False,
        )
        return ModelInput(
            timestamp=0.0,
            step_index=0,
            task_name=self._task_name,
            spec=spec,
            state_history=np.empty((1, 0), dtype=np.float32),
            human_action_history=np.empty((1, 0), dtype=np.float32),
            executed_action_history=np.empty((1, 0), dtype=np.float32),
            history_timestamps=np.zeros(1),
            human_action_timestamps=np.full(1, np.nan),
            human_action_age_ms=np.full(1, np.inf, dtype=np.float32),
            history_valid=np.zeros(1, dtype=np.bool_),
            human_action_valid=np.zeros(1, dtype=np.bool_),
            human_control_active=np.zeros(1, dtype=np.bool_),
        )

    def _on_result_timer(self) -> None:
        self._check_active_timeout()
        while True:
            try:
                result = self._result_queue.get_nowait()
            except Empty:
                return
            step = result.model_input.step_index
            if result.error:
                self._report_error(f"policy inference failed: {result.error}", step)
                continue
            if result.duration_ms > self._timeout_ms:
                self._report_error(
                    f"policy inference timeout: {result.duration_ms:.1f} ms > "
                    f"{self._timeout_ms:.1f} ms",
                    step,
                )
                continue
            output = result.output
            if output is None or not output.valid:
                self._report_error("policy returned an invalid output", step)
                continue
            if abs(output.timestamp - result.model_input.timestamp) > 1e-6:
                self._report_error("policy output timestamp does not match its input", step)
                continue
            if not 0.0 <= float(output.command[7]) <= 0.08:
                self._report_error("policy gripper output is outside [0.0, 0.08] m", step)
                continue
            self._last_valid_output = output
            self._last_error = ""
            self._timeout_reported_for_step = None
            self._publish_output(output, step, result.duration_ms)

    def _check_active_timeout(self) -> None:
        with self._worker_state_lock:
            active = self._active_inference
        if active is None:
            return
        step, started_ns, _ = active
        elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
        if elapsed_ms > self._timeout_ms and self._timeout_reported_for_step != step:
            self._timeout_reported_for_step = step
            self._report_error(
                f"policy inference is still running after {elapsed_ms:.1f} ms "
                f"(limit {self._timeout_ms:.1f} ms)",
                step,
            )

    def _publish_output(
        self, output: ModelOutput, step_index: int, duration_ms: float
    ) -> None:
        stamp = _time_message(output.timestamp)
        if output.control_active:
            if output.command_space == CommandSpace.CARTESIAN_POSE:
                message = PoseStamped()
                message.header.stamp = stamp
                message.header.frame_id = "world"
                message.pose.position.x = float(output.command[0])
                message.pose.position.y = float(output.command[1])
                message.pose.position.z = float(output.command[2])
                message.pose.orientation.w = float(output.command[3])
                message.pose.orientation.x = float(output.command[4])
                message.pose.orientation.y = float(output.command[5])
                message.pose.orientation.z = float(output.command[6])
                self._pose_pub.publish(message)
            else:
                message = JointState()
                message.header.stamp = stamp
                message.position = output.command[:7].tolist()
                self._joint_pub.publish(message)
        self._gripper_pub.publish(Float64(data=float(output.command[7])))

        output_message = Joy()
        output_message.header.stamp = stamp
        output_message.header.frame_id = "world"
        space_code = 0.0 if output.command_space == CommandSpace.CARTESIAN_POSE else 1.0
        output_message.axes = [
            *output.command.tolist(),
            1.0,
            float(output.confidence),
            float(output.control_active),
            space_code,
            float(output.fallback_used),
        ]
        output_message.buttons = [step_index]
        if len(output_message.axes) != POLICY_OUTPUT_AXES_SIZE:
            raise RuntimeError("Policy output ROS schema has an unexpected size")
        self._output_pub.publish(output_message)
        self._publish_status(
            "ok",
            step_index=step_index,
            policy=output.policy_name,
            command_space=output.command_space.value,
            confidence=output.confidence,
            inference_ms=duration_ms,
            holding_last_valid=False,
            diagnostics=output.diagnostics,
        )

    def _report_error(self, reason: str, step_index: int) -> None:
        self._last_error = reason
        holding = self._last_valid_output is not None
        self.get_logger().error(
            "SHARED CONTROL ERROR: "
            f"{reason}; step={step_index}; holding_last_valid_command={holding}",
            throttle_duration_sec=1.0,
        )
        self._publish_status(
            "error",
            reason=reason,
            step_index=step_index,
            holding_last_valid=holding,
            last_policy=(
                self._last_valid_output.policy_name
                if self._last_valid_output is not None
                else None
            ),
        )

    def _publish_status(self, state: str, **fields: Any) -> None:
        payload = {"state": state, "timestamp_monotonic": time.monotonic(), **fields}
        self._status_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, default=str))
        )

    def _on_reset(self, _: EmptyMessage) -> None:
        self._states.clear()
        self._actions.clear()
        for cache in (*self._rgb.values(), *self._depth.values()):
            cache.clear()
        self._samples.clear()
        self._processed_keys.clear()
        self._processed_key_set.clear()
        self._human = _HumanCommand(
            command=np.full(COMMAND_DIM, np.nan, dtype=np.float32),
            valid=False,
            active=False,
            source_timestamp=float("nan"),
            receipt_ns=0,
        )
        self._last_valid_output = None
        self._last_error = ""
        self._replace_work(_ResetPolicy(self._task_name))
        self._publish_status("reset", policy=type(self._policy).__name__)

    def _on_collection_status(self, message: String) -> None:
        prefix = "task set to "
        if not message.data.startswith(prefix):
            return
        task_name = message.data[len(prefix):].strip()
        if not task_name or task_name == self._task_name:
            return
        self._task_name = task_name
        self._on_reset(EmptyMessage())
        self.get_logger().info(f"Shared-control task changed to {task_name}")

    def _remember_processed(self, key: tuple[int, int]) -> None:
        if len(self._processed_keys) == self._processed_keys.maxlen:
            oldest = self._processed_keys[0]
            self._processed_key_set.discard(oldest)
        self._processed_keys.append(key)
        self._processed_key_set.add(key)

    def _remove_key(self, key: tuple[int, int]) -> None:
        self._states.pop(key, None)
        self._actions.pop(key, None)
        for cache in (*self._rgb.values(), *self._depth.values()):
            cache.pop(key, None)
            while len(cache) > 32:
                cache.pop(min(cache))

    def destroy_node(self) -> bool:
        self._stop_event.set()
        self._replace_work(None)
        self._worker.join(timeout=2.0)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SharedControlInferenceNode()
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
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
