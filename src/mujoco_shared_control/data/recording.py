"""Synchronized RGB-D demonstration recording in per-episode HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, IntFlag
import json
from pathlib import Path
from queue import Queue
import re
from threading import Event, Lock, Thread
from typing import Any
import uuid
import zlib

import h5py
import numpy as np
from numpy.typing import NDArray

from mujoco_shared_control.utils.pose import matrix_to_quaternion


SCHEMA_VERSION = "1.1.0"
SAMPLE_RATE_HZ = 20.0
STATE_26_DIM = 26
ACTION_DIM = 8
STATE_26_NAMES = (
    *(f"joint_position_{index}" for index in range(1, 8)),
    *(f"joint_velocity_{index}" for index in range(1, 8)),
    "ee_x",
    "ee_y",
    "ee_z",
    "gripper_opening",
    "object_x",
    "object_y",
    "object_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "object_grasped",
    "object_goal_distance",
)


class Stage(IntEnum):
    APPROACH = 0
    GRASP = 1
    TRANSPORT = 2
    PLACE = 3
    COMPLETE = 4


class FrameEvent(IntFlag):
    NONE = 0
    GRIP_PRESSED = 1 << 0
    GRIP_RELEASED = 1 << 1
    GRASP_ACQUIRED = 1 << 2
    GRASP_LOST = 1 << 3
    ENTERED_GOAL = 1 << 4
    TASK_SUCCESS = 1 << 5


def _pose_vector(pose: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate(
        (pose[:3, 3], matrix_to_quaternion(pose[:3, :3]))
    ).astype(np.float64, copy=False)


def build_state_26(observation: dict[str, Any]) -> NDArray[np.float32]:
    """Build the documented 26-D policy state from a raw environment observation."""
    object_position = np.asarray(observation["object_pose"][:3, 3], dtype=np.float64)
    goal_position = np.asarray(observation["goal_pose"][:3, 3], dtype=np.float64)
    state = np.concatenate(
        (
            observation["q_obs"],
            observation["dq_obs"],
            observation["ee_pose"][:3, 3],
            observation["gripper"],
            object_position,
            goal_position,
            [float(bool(observation["object_grasped"]))],
            [float(np.linalg.norm(object_position - goal_position))],
        )
    ).astype(np.float32)
    if state.shape != (STATE_26_DIM,) or not np.isfinite(state).all():
        raise ValueError("26-D policy state must contain 26 finite values")
    return state


@dataclass(frozen=True)
class TeleopSnapshot:
    """Latest VR input and processed user command latched at a sample boundary."""

    raw: NDArray[np.float64]  # xyz + quaternion_xyzw + trigger + grip
    raw_valid: bool
    aligned: bool
    raw_source_timestamp: float
    raw_age_ms: float
    user_command: NDArray[np.float64]  # ee xyz + quaternion_wxyz + gripper
    user_command_valid: bool
    user_command_source_timestamp: float
    user_command_age_ms: float
    control_orientation: bool
    policy_output: NDArray[np.float64] = field(
        default_factory=lambda: np.full(8, np.nan, dtype=np.float64)
    )
    policy_output_valid: bool = False
    policy_output_command_space: int = 0
    policy_output_confidence: float = 0.0
    policy_output_control_active: bool = False
    policy_output_fallback_used: bool = False
    policy_output_source_timestamp: float = float("nan")
    policy_output_age_ms: float = float("inf")


@dataclass(frozen=True)
class FrameIdentity:
    episode_id: str
    task_name: str
    step_index: int


@dataclass(frozen=True)
class EpisodeToken:
    episode_id: str
    task_name: str
    inprogress_path: Path
    scheduled_frames: int


@dataclass(frozen=True)
class FramePayload:
    """All non-image values captured atomically before executing an action."""

    identity: FrameIdentity
    simulation_timestamp: float
    sample_monotonic_ns: int
    observation: dict[str, Any]
    state_26: NDArray[np.float32]
    policy_state_42: NDArray[np.float32]
    teleop: TeleopSnapshot
    executed_action: NDArray[np.float64]
    mujoco_ctrl: NDArray[np.float64]
    ik_success: bool
    command_accepted: bool
    action_clipped: bool
    fallback_used: bool
    rejection_reason: str
    reward: float
    task_success: bool
    stage: int
    events: int


@dataclass(frozen=True)
class RenderedFrame:
    payload: FramePayload
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    image_valid: bool
    drop_reason: str
    render_start_monotonic_ns: int
    render_end_monotonic_ns: int
    camera_calibration: dict[str, Any]


class StageTracker:
    """Derive reproducible pick-place stages and transition events."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous_grip_pressed = False
        self._previous_grasped = False
        self._previous_inside_goal = False
        self._previous_success = False

    def update(
        self,
        observation: dict[str, Any],
        teleop: TeleopSnapshot,
        task_success: bool,
    ) -> tuple[int, int]:
        object_position = observation["object_pose"][:3, 3]
        goal_position = observation["goal_pose"][:3, 3]
        object_goal_distance = float(np.linalg.norm(object_position - goal_position))
        grasped = bool(observation["object_grasped"])
        grip_pressed = bool(teleop.raw_valid and teleop.raw[8] >= 0.5)
        inside_goal = object_goal_distance < 0.08

        if task_success and not grasped:
            stage = Stage.COMPLETE
        elif grasped and inside_goal:
            stage = Stage.PLACE
        elif grasped:
            stage = Stage.TRANSPORT
        elif grip_pressed or bool(observation["contact"]["count"][0]):
            stage = Stage.GRASP
        else:
            stage = Stage.APPROACH

        events = FrameEvent.NONE
        if grip_pressed and not self._previous_grip_pressed:
            events |= FrameEvent.GRIP_PRESSED
        if not grip_pressed and self._previous_grip_pressed:
            events |= FrameEvent.GRIP_RELEASED
        if grasped and not self._previous_grasped:
            events |= FrameEvent.GRASP_ACQUIRED
        if not grasped and self._previous_grasped:
            events |= FrameEvent.GRASP_LOST
        if inside_goal and not self._previous_inside_goal:
            events |= FrameEvent.ENTERED_GOAL
        if task_success and not self._previous_success:
            events |= FrameEvent.TASK_SUCCESS

        self._previous_grip_pressed = grip_pressed
        self._previous_grasped = grasped
        self._previous_inside_goal = inside_goal
        self._previous_success = task_success
        return int(stage), int(events)


def _safe_task_name(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("task name cannot be empty")
    safe = re.sub(r"[^\w.-]+", "_", stripped, flags=re.UNICODE).strip("._-")
    if not safe:
        raise ValueError("task name must contain at least one letter or number")
    return safe


class _EpisodeFile:
    def __init__(self, token: EpisodeToken, first_frame: RenderedFrame) -> None:
        token.inprogress_path.parent.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.file = h5py.File(token.inprogress_path, "w")
        self.length = 0
        self.file.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_id": token.episode_id,
                "task_name": token.task_name,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "timing_semantics": (
                    "observation and RGB-D are captured before executed_action is "
                    "applied over [t, t+0.05s)"
                ),
                "state_26_names_json": json.dumps(STATE_26_NAMES),
                "camera_calibration_json": json.dumps(
                    first_frame.camera_calibration, separators=(",", ":")
                ),
            }
        )
        height, width = first_frame.depth.shape
        self.file.attrs["camera_name"] = first_frame.camera_calibration["name"]
        self.file.attrs["camera_width"] = width
        self.file.attrs["camera_height"] = height
        self.file.attrs["rgb_encoding"] = "rgb8"
        self.file.attrs["rgb_channel_order"] = "RGB"
        self.file.attrs["depth_encoding"] = "32FC1"
        self.file.attrs["depth_unit"] = "meter"
        self._create_datasets(height, width)

    def _dataset(
        self,
        name: str,
        shape_tail: tuple[int, ...],
        dtype: Any,
        *,
        compression: str | None = None,
    ) -> None:
        group_name, dataset_name = name.rsplit("/", 1)
        group = self.file.require_group(group_name)
        chunk = (1, *shape_tail) if shape_tail else (256,)
        group.create_dataset(
            dataset_name,
            shape=(0, *shape_tail),
            maxshape=(None, *shape_tail),
            chunks=chunk,
            dtype=dtype,
            compression=compression,
            shuffle=bool(compression),
        )

    def _create_datasets(self, height: int, width: int) -> None:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        specifications = {
            "identity/step_index": ((), np.int64),
            "timestamps/simulation": ((), np.float64),
            "timestamps/sample_monotonic_ns": ((), np.uint64),
            "observations/state_26": ((26,), np.float32),
            "observations/policy_state_42": ((42,), np.float32),
            "observations/joint_position": ((7,), np.float64),
            "observations/joint_velocity": ((7,), np.float64),
            "observations/ee_pose_xyz_wxyz": ((7,), np.float64),
            "observations/gripper_opening": ((), np.float64),
            "observations/gripper_joint_position": ((2,), np.float64),
            "observations/gripper_joint_velocity": ((2,), np.float64),
            "observations/object_pose_xyz_wxyz": ((7,), np.float64),
            "observations/goal_pose_xyz_wxyz": ((7,), np.float64),
            "observations/object_linear_velocity": ((3,), np.float64),
            "observations/object_angular_velocity": ((3,), np.float64),
            "observations/contact": ((5,), np.float32),
            "observations/object_grasped": ((), np.uint8),
            "actions/vr_raw": ((9,), np.float64),
            "actions/vr_raw_valid": ((), np.uint8),
            "actions/vr_aligned": ((), np.uint8),
            "actions/vr_source_timestamp": ((), np.float64),
            "actions/vr_age_ms": ((), np.float64),
            "actions/user_command": ((8,), np.float64),
            "actions/user_command_valid": ((), np.uint8),
            "actions/user_command_source_timestamp": ((), np.float64),
            "actions/user_command_age_ms": ((), np.float64),
            "actions/control_orientation": ((), np.uint8),
            "actions/policy_output": ((8,), np.float64),
            "actions/policy_output_valid": ((), np.uint8),
            "actions/policy_output_command_space": ((), np.uint8),
            "actions/policy_output_confidence": ((), np.float32),
            "actions/policy_output_control_active": ((), np.uint8),
            "actions/policy_output_fallback_used": ((), np.uint8),
            "actions/policy_output_source_timestamp": ((), np.float64),
            "actions/policy_output_age_ms": ((), np.float64),
            "actions/executed": ((8,), np.float64),
            "actions/mujoco_ctrl": ((8,), np.float64),
            "actions/status": ((4,), np.uint8),
            "actions/rejection_reason": ((), string_dtype),
            "labels/stage": ((), np.uint8),
            "labels/events": ((), np.uint32),
            "labels/reward": ((), np.float64),
            "labels/task_success": ((), np.uint8),
            "camera/front/frame_index": ((), np.int64),
            "camera/front/name": ((), string_dtype),
            "camera/front/width": ((), np.uint16),
            "camera/front/height": ((), np.uint16),
            "camera/front/rgb_encoding": ((), string_dtype),
            "camera/front/depth_encoding": ((), string_dtype),
            "camera/front/rgb_channel_order": ((), string_dtype),
            "camera/front/image_valid": ((), np.uint8),
            "camera/front/image_scene_timestamp": ((), np.float64),
            "camera/front/image_age_ms": ((), np.float64),
            "camera/front/render_start_monotonic_ns": ((), np.uint64),
            "camera/front/render_end_monotonic_ns": ((), np.uint64),
            "camera/front/render_latency_ms": ((), np.float64),
            "camera/front/state_image_sync_error_ms": ((), np.float64),
            "camera/front/rgb_crc32": ((), np.uint32),
            "camera/front/drop_reason": ((), string_dtype),
        }
        for name, (shape_tail, dtype) in specifications.items():
            self._dataset(name, shape_tail, dtype)
        self._dataset(
            "observations/images/front/rgb",
            (height, width, 3),
            np.uint8,
            compression="lzf",
        )
        self._dataset(
            "observations/images/front/depth",
            (height, width),
            np.float32,
            compression="lzf",
        )

    def _append(self, name: str, value: Any) -> None:
        dataset = self.file[name]
        dataset.resize(self.length + 1, axis=0)
        dataset[self.length] = value

    def append(self, frame: RenderedFrame) -> None:
        payload = frame.payload
        obs = payload.observation
        contact = obs["contact"]
        render_latency_ms = (
            frame.render_end_monotonic_ns - frame.render_start_monotonic_ns
        ) / 1_000_000.0
        values = {
            "identity/step_index": payload.identity.step_index,
            "timestamps/simulation": payload.simulation_timestamp,
            "timestamps/sample_monotonic_ns": payload.sample_monotonic_ns,
            "observations/state_26": payload.state_26,
            "observations/policy_state_42": payload.policy_state_42,
            "observations/joint_position": obs["q_obs"],
            "observations/joint_velocity": obs["dq_obs"],
            "observations/ee_pose_xyz_wxyz": _pose_vector(obs["ee_pose"]),
            "observations/gripper_opening": float(obs["gripper"][0]),
            "observations/gripper_joint_position": obs["gripper_joint_positions"],
            "observations/gripper_joint_velocity": obs["gripper_joint_velocities"],
            "observations/object_pose_xyz_wxyz": _pose_vector(obs["object_pose"]),
            "observations/goal_pose_xyz_wxyz": _pose_vector(obs["goal_pose"]),
            "observations/object_linear_velocity": obs["object_linear_velocity"],
            "observations/object_angular_velocity": obs["object_angular_velocity"],
            "observations/contact": np.array(
                [
                    contact["left"][0],
                    contact["right"][0],
                    contact["left_force"][0],
                    contact["right_force"][0],
                    contact["count"][0],
                ],
                dtype=np.float32,
            ),
            "observations/object_grasped": int(bool(obs["object_grasped"])),
            "actions/vr_raw": payload.teleop.raw,
            "actions/vr_raw_valid": int(payload.teleop.raw_valid),
            "actions/vr_aligned": int(payload.teleop.aligned),
            "actions/vr_source_timestamp": payload.teleop.raw_source_timestamp,
            "actions/vr_age_ms": payload.teleop.raw_age_ms,
            "actions/user_command": payload.teleop.user_command,
            "actions/user_command_valid": int(payload.teleop.user_command_valid),
            "actions/user_command_source_timestamp": (
                payload.teleop.user_command_source_timestamp
            ),
            "actions/user_command_age_ms": payload.teleop.user_command_age_ms,
            "actions/control_orientation": int(payload.teleop.control_orientation),
            "actions/policy_output": payload.teleop.policy_output,
            "actions/policy_output_valid": int(
                payload.teleop.policy_output_valid
            ),
            "actions/policy_output_command_space": (
                payload.teleop.policy_output_command_space
            ),
            "actions/policy_output_confidence": (
                payload.teleop.policy_output_confidence
            ),
            "actions/policy_output_control_active": int(
                payload.teleop.policy_output_control_active
            ),
            "actions/policy_output_fallback_used": int(
                payload.teleop.policy_output_fallback_used
            ),
            "actions/policy_output_source_timestamp": (
                payload.teleop.policy_output_source_timestamp
            ),
            "actions/policy_output_age_ms": payload.teleop.policy_output_age_ms,
            "actions/executed": payload.executed_action,
            "actions/mujoco_ctrl": payload.mujoco_ctrl,
            "actions/status": np.array(
                [
                    payload.ik_success,
                    payload.command_accepted,
                    payload.action_clipped,
                    payload.fallback_used,
                ],
                dtype=np.uint8,
            ),
            "actions/rejection_reason": payload.rejection_reason,
            "labels/stage": payload.stage,
            "labels/events": payload.events,
            "labels/reward": payload.reward,
            "labels/task_success": int(payload.task_success),
            "camera/front/frame_index": payload.identity.step_index,
            "camera/front/name": frame.camera_calibration["name"],
            "camera/front/width": frame.rgb.shape[1],
            "camera/front/height": frame.rgb.shape[0],
            "camera/front/rgb_encoding": "rgb8",
            "camera/front/depth_encoding": "32FC1",
            "camera/front/rgb_channel_order": "RGB",
            "camera/front/image_valid": int(frame.image_valid),
            "camera/front/image_scene_timestamp": payload.simulation_timestamp,
            "camera/front/image_age_ms": 0.0,
            "camera/front/render_start_monotonic_ns": frame.render_start_monotonic_ns,
            "camera/front/render_end_monotonic_ns": frame.render_end_monotonic_ns,
            "camera/front/render_latency_ms": render_latency_ms,
            "camera/front/state_image_sync_error_ms": 0.0,
            "camera/front/rgb_crc32": zlib.crc32(frame.rgb.tobytes()),
            "camera/front/drop_reason": frame.drop_reason,
            "observations/images/front/rgb": frame.rgb,
            "observations/images/front/depth": frame.depth,
        }
        for name, value in values.items():
            self._append(name, value)
        self.length += 1
        if self.length % 20 == 0:
            self.file.flush()

    def close(self, scheduled_frames: int) -> None:
        self.file.attrs["scheduled_frames"] = scheduled_frames
        self.file.attrs["written_frames"] = self.length
        self.file.flush()
        self.file.close()


@dataclass
class _FinalizeRequest:
    token: EpisodeToken
    discard: bool
    completed: Event
    result: dict[str, Any] | None = None
    error: BaseException | None = None


class EpisodeRecorder:
    """Reserve frame indices on the control thread and write frames asynchronously."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self._lock = Lock()
        self._active: EpisodeToken | None = None
        self._next_step = 0
        self._queue: Queue[RenderedFrame | _FinalizeRequest | None] = Queue()
        self._worker_error: BaseException | None = None
        self._worker = Thread(target=self._run_writer, name="hdf5_writer", daemon=True)
        self._worker.start()

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._active is not None

    def start(self, task_name: str) -> str:
        safe_task = _safe_task_name(task_name)
        with self._lock:
            self._raise_worker_error()
            if self._active is not None:
                raise RuntimeError("an episode is already recording")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            episode_id = f"{safe_task}_{timestamp}_{uuid.uuid4().hex[:8]}"
            path = self.dataset_root / safe_task / f"episode_{episode_id}.inprogress.h5"
            self._active = EpisodeToken(episode_id, safe_task, path, 0)
            self._next_step = 0
            return episode_id

    def reserve_step(self) -> FrameIdentity | None:
        with self._lock:
            self._raise_worker_error()
            if self._active is None:
                return None
            identity = FrameIdentity(
                self._active.episode_id,
                self._active.task_name,
                self._next_step,
            )
            self._next_step += 1
            return identity

    def stop(self) -> EpisodeToken:
        with self._lock:
            self._raise_worker_error()
            if self._active is None:
                raise RuntimeError("no episode is currently recording")
            token = EpisodeToken(
                self._active.episode_id,
                self._active.task_name,
                self._active.inprogress_path,
                self._next_step,
            )
            self._active = None
            self._next_step = 0
            return token

    def submit_frame(self, frame: RenderedFrame) -> None:
        self._raise_worker_error()
        self._queue.put(frame)

    def finalize(self, token: EpisodeToken, *, discard: bool = False) -> dict[str, Any]:
        self._raise_worker_error()
        request = _FinalizeRequest(token=token, discard=discard, completed=Event())
        self._queue.put(request)
        if not request.completed.wait(timeout=300.0):
            raise TimeoutError(f"timed out finalizing episode {token.episode_id}")
        if request.error is not None:
            raise RuntimeError(f"failed to finalize {token.episode_id}") from request.error
        assert request.result is not None
        return request.result

    def wait_idle(self) -> None:
        self._queue.join()
        self._raise_worker_error()

    def close(self) -> None:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("cannot close recorder while an episode is recording")
        self._queue.put(None)
        self._worker.join(timeout=300.0)
        if self._worker.is_alive():
            raise TimeoutError("HDF5 writer did not shut down")
        self._raise_worker_error()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("HDF5 writer failed") from self._worker_error

    def _run_writer(self) -> None:
        files: dict[str, _EpisodeFile] = {}
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    if isinstance(item, RenderedFrame):
                        episode_id = item.payload.identity.episode_id
                        episode_file = files.get(episode_id)
                        if episode_file is None:
                            token = EpisodeToken(
                                episode_id=episode_id,
                                task_name=item.payload.identity.task_name,
                                inprogress_path=(
                                    self.dataset_root
                                    / item.payload.identity.task_name
                                    / f"episode_{episode_id}.inprogress.h5"
                                ),
                                scheduled_frames=0,
                            )
                            episode_file = _EpisodeFile(token, item)
                            files[episode_id] = episode_file
                        episode_file.append(item)
                    else:
                        self._finalize_request(item, files)
                finally:
                    self._queue.task_done()
        except BaseException as error:
            self._worker_error = error
            for episode_file in files.values():
                try:
                    episode_file.file.close()
                except Exception:
                    pass

    def _finalize_request(
        self,
        request: _FinalizeRequest,
        files: dict[str, _EpisodeFile],
    ) -> None:
        try:
            episode_file = files.pop(request.token.episode_id, None)
            if episode_file is None:
                request.token.inprogress_path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(request.token.inprogress_path, "w") as empty_file:
                    empty_file.attrs.update(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "episode_id": request.token.episode_id,
                            "task_name": request.token.task_name,
                            "sample_rate_hz": SAMPLE_RATE_HZ,
                            "scheduled_frames": request.token.scheduled_frames,
                            "written_frames": 0,
                        }
                    )
            else:
                episode_file.close(request.token.scheduled_frames)

            if request.discard:
                request.token.inprogress_path.unlink(missing_ok=True)
                request.result = {
                    "episode_id": request.token.episode_id,
                    "discarded": True,
                }
                return

            report = validate_episode(request.token.inprogress_path)
            destination_dir = request.token.inprogress_path.parent
            if not report["valid"]:
                destination_dir = destination_dir / "invalid"
                destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / request.token.inprogress_path.name.replace(
                ".inprogress", ""
            )
            request.token.inprogress_path.replace(destination)
            report_path = destination.with_suffix(".validation.json")
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            request.result = {
                **report,
                "episode_id": request.token.episode_id,
                "path": str(destination),
            }
        except BaseException as error:
            request.error = error
        finally:
            request.completed.set()


def validate_episode(path: str | Path) -> dict[str, Any]:
    """Dispatch validation without changing the established schema-1 contract."""
    episode_path = Path(path)
    try:
        with h5py.File(episode_path, "r") as episode:
            version = str(episode.attrs.get("schema_version", ""))
    except OSError as error:
        return {"path": str(episode_path), "valid": False, "errors": [str(error)]}
    if version.startswith("2."):
        return _validate_transition_episode(episode_path)
    return _validate_rgbd_episode(episode_path)


def _validate_rgbd_episode(path: str | Path) -> dict[str, Any]:
    """Validate timing, image completeness, shapes, and synchronization."""
    episode_path = Path(path)
    report: dict[str, Any] = {
        "path": str(episode_path),
        "valid": False,
        "errors": [],
    }
    try:
        with h5py.File(episode_path, "r") as episode:
            required = (
                "timestamps/simulation",
                "identity/step_index",
                "camera/front/image_valid",
                "camera/front/image_age_ms",
                "camera/front/state_image_sync_error_ms",
                "camera/front/rgb_crc32",
                "observations/images/front/rgb",
                "observations/images/front/depth",
                "observations/state_26",
                "actions/executed",
            )
            missing = [name for name in required if name not in episode]
            if missing:
                report["errors"].append(f"missing datasets: {missing}")
                return report

            timestamps = episode["timestamps/simulation"][:]
            steps = episode["identity/step_index"][:]
            image_valid = episode["camera/front/image_valid"][:].astype(bool)
            sync_error = episode["camera/front/state_image_sync_error_ms"][:]
            image_age = episode["camera/front/image_age_ms"][:]
            hashes = episode["camera/front/rgb_crc32"][:]
            rgb_shape = episode["observations/images/front/rgb"].shape
            depth_shape = episode["observations/images/front/depth"].shape
            scheduled = int(episode.attrs.get("scheduled_frames", len(timestamps)))
            count = len(timestamps)
            intervals = np.diff(timestamps)
            strictly_increasing = bool(count > 1 and np.all(intervals > 0.0))
            contiguous_steps = bool(np.array_equal(steps, np.arange(count)))
            average_hz = (
                float(1.0 / np.mean(intervals)) if len(intervals) else float("nan")
            )
            duplicate_count = int(np.count_nonzero(np.diff(hashes) == 0))
            missing_images = int(np.count_nonzero(~image_valid))
            stale_images = int(np.count_nonzero(image_age > 25.0))
            valid_images = int(np.count_nonzero(image_valid))
            report.update(
                {
                    "scheduled_frames": scheduled,
                    "written_frames": count,
                    "average_sample_hz": average_hz,
                    "mean_interval_ms": (
                        float(np.mean(intervals) * 1000.0) if len(intervals) else None
                    ),
                    "max_interval_ms": (
                        float(np.max(intervals) * 1000.0) if len(intervals) else None
                    ),
                    "p95_interval_ms": (
                        float(np.percentile(intervals, 95) * 1000.0)
                        if len(intervals)
                        else None
                    ),
                    "strictly_increasing_timestamps": strictly_increasing,
                    "contiguous_step_indices": contiguous_steps,
                    "rgb_frames": int(rgb_shape[0]),
                    "depth_frames": int(depth_shape[0]),
                    "rgb_shape": list(rgb_shape[1:]),
                    "depth_shape": list(depth_shape[1:]),
                    "missing_image_frames": missing_images,
                    "image_drop_rate": float(missing_images / max(count, 1)),
                    "stale_image_frames": stale_images,
                    "consecutive_duplicate_rgb_frames": duplicate_count,
                    "mean_state_image_sync_error_ms": (
                        float(np.mean(np.abs(sync_error))) if count else None
                    ),
                    "max_state_image_sync_error_ms": (
                        float(np.max(np.abs(sync_error))) if count else None
                    ),
                    "per_camera": {
                        "front": {
                            "total_frames": count,
                            "valid_frames": valid_images,
                            "missing_frames": missing_images,
                            "stale_frames": stale_images,
                        }
                    },
                }
            )
            if count < 2:
                report["errors"].append("episode must contain at least two frames")
            if scheduled != count:
                report["errors"].append(
                    f"scheduled {scheduled} frames but wrote {count}"
                )
            if not strictly_increasing:
                report["errors"].append("simulation timestamps are not strictly increasing")
            if not contiguous_steps:
                report["errors"].append("step indices are not contiguous from zero")
            if missing_images:
                report["errors"].append(f"{missing_images} RGB-D frames are invalid")
            if stale_images:
                report["errors"].append(f"{stale_images} RGB-D frames are stale")
            if count and float(np.max(np.abs(sync_error))) > 1e-6:
                report["errors"].append("RGB-D and state timestamps are not synchronized")
            if len(intervals) and not np.allclose(intervals, 0.05, atol=1e-6):
                report["errors"].append("sample intervals are not exactly 50 ms")
            if rgb_shape[0] != count or depth_shape[0] != count:
                report["errors"].append("RGB-D frame counts do not match state count")
            report["valid"] = not report["errors"]
            return report
    except OSError as error:
        report["errors"].append(str(error))
        return report


def _validate_transition_episode(path: str | Path) -> dict[str, Any]:
    episode_path = Path(path)
    report: dict[str, Any] = {"path": str(episode_path), "valid": False, "errors": []}
    required_attrs = (
        "schema_version", "episode_id", "run_id", "worker_id",
        "worker_episode_index", "environment_seed", "policy_seed",
        "perturbation_seed", "expert_type", "expert_code_version",
        "config_version", "config_hash", "collection_variant", "image_enabled",
    )
    required = (
        "identity/step_index", "timestamps/simulation_before",
        "timestamps/simulation_after", "observations/state_26",
        "next_observations/state_26", "actions/expert_nominal",
        "actions/policy_command", "actions/executed_joint_target",
        "labels/reward", "labels/terminated", "labels/truncated",
        "labels/task_success", "labels/termination_reason", "labels/stage",
        "labels/next_stage", "perturbations/active",
    )
    try:
        with h5py.File(episode_path, "r") as episode:
            missing_attrs = [name for name in required_attrs if name not in episode.attrs]
            missing = [name for name in required if name not in episode]
            if missing_attrs:
                report["errors"].append(f"missing attributes: {missing_attrs}")
            if missing:
                report["errors"].append(f"missing datasets: {missing}")
                return report
            if bool(episode.attrs["image_enabled"]):
                report["errors"].append("state-only schema must set image_enabled=false")
            if "camera" in episode or "observations/images" in episode:
                report["errors"].append("state-only episode unexpectedly contains camera data")
            steps = episode["identity/step_index"][:]
            count = len(steps)
            before = episode["timestamps/simulation_before"][:]
            after = episode["timestamps/simulation_after"][:]
            lengths = {name: len(episode[name]) for name in required}
            bad_lengths = {name: length for name, length in lengths.items() if length != count}
            if bad_lengths:
                report["errors"].append(f"dataset length mismatch: {bad_lengths}")
            if count < 1:
                report["errors"].append("episode must contain at least one transition")
            if not np.array_equal(steps, np.arange(count)):
                report["errors"].append("step indices are not contiguous from zero")
            if count and (not np.all(after > before) or
                          (count > 1 and not np.allclose(before[1:], after[:-1], atol=1e-9))):
                report["errors"].append("simulation transition timestamps are not aligned")
            if count > 1:
                current = episode["observations/state_26"][1:]
                previous_next = episode["next_observations/state_26"][:-1]
                if not np.allclose(current, previous_next, atol=1e-6):
                    report["errors"].append("obs[t+1] does not match next_obs[t]")
            terminal = np.logical_or(episode["labels/terminated"][:].astype(bool),
                                     episode["labels/truncated"][:].astype(bool))
            if count and (not terminal[-1] or np.any(terminal[:-1])):
                report["errors"].append("terminal flag must appear exactly on the final row")
            tracking = ("labels/entered_settling", "labels/settling_step",
                        "labels/expert_failed_step", "labels/task_milestones")
            present_tracking = [name in episode for name in tracking]
            if any(present_tracking) and not all(present_tracking):
                report["errors"].append("settling tracking fields are only partially present")
            if all(present_tracking):
                milestones = episode["labels/task_milestones"][:]
                if milestones.shape != (count, 5):
                    report["errors"].append("task_milestones must have shape (T, 5)")
                if count > 1 and np.any(np.diff(milestones.astype(np.int8), axis=0) < 0):
                    report["errors"].append("task milestones must be monotonic")
                failed_steps = episode["labels/expert_failed_step"][:]
                nonnegative = failed_steps[failed_steps >= 0]
                if len(nonnegative) and np.any(nonnegative != nonnegative[0]):
                    report["errors"].append("expert_failed_step changes within an episode")
            report.update({
                "written_transitions": count,
                "contiguous_step_indices": bool(np.array_equal(steps, np.arange(count))),
                "transition_aligned": not any("aligned" in error or "next_obs" in error
                                               for error in report["errors"]),
                "image_enabled": False,
                "schema_version": str(episode.attrs["schema_version"]),
            })
            report["valid"] = not report["errors"]
            return report
    except OSError as error:
        report["errors"].append(str(error))
        return report
