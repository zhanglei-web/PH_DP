"""Stable NumPy contracts at the model inference boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


MODEL_INTERFACE_VERSION = "1.0.0"
STATE_DIM = 26
COMMAND_DIM = 8


class CommandSpace(str, Enum):
    """Command coordinates understood by the ROS output adapter."""

    CARTESIAN_POSE = "cartesian_pose"
    JOINT_POSITION = "joint_position"


@dataclass(frozen=True)
class ModelInputSpec:
    """Inputs and history sizes requested by one policy plugin."""

    history_length: int = 8
    use_state_26: bool = True
    use_human_action: bool = True
    use_executed_action: bool = True
    use_rgb: bool = False
    use_depth: bool = False
    image_history_length: int = 1
    cameras: tuple[str, ...] = ("front",)

    def __post_init__(self) -> None:
        if self.history_length < 1:
            raise ValueError("history_length must be at least 1")
        if self.image_history_length < 1:
            raise ValueError("image_history_length must be at least 1")
        cameras = tuple(str(name).strip() for name in self.cameras)
        if (self.use_rgb or self.use_depth) and (
            not cameras or any(not name for name in cameras)
        ):
            raise ValueError("image policies must declare at least one camera")
        object.__setattr__(self, "cameras", cameras)


def _array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    finite: bool = True,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class ModelInput:
    """One inference request assembled from synchronized history and RGB-D.

    History is oldest-to-newest.  ``history_valid`` marks real samples when the
    beginning of an episode is left-padded.  Human commands use
    ``xyz + quaternion_wxyz + gripper``.  Executed actions use
    ``joint1..joint7 + gripper``.
    """

    timestamp: float
    step_index: int
    task_name: str
    spec: ModelInputSpec
    state_history: NDArray[np.float32]
    human_action_history: NDArray[np.float32]
    executed_action_history: NDArray[np.float32]
    history_timestamps: NDArray[np.float64]
    human_action_timestamps: NDArray[np.float64]
    human_action_age_ms: NDArray[np.float32]
    history_valid: NDArray[np.bool_]
    human_action_valid: NDArray[np.bool_]
    human_control_active: NDArray[np.bool_]
    rgb: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
    depth: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    image_timestamps: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    image_valid: dict[str, NDArray[np.bool_]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        history_length = self.spec.history_length
        normalized = {
            "state_history": _array(
                self.state_history,
                name="state_history",
                shape=(history_length, STATE_DIM if self.spec.use_state_26 else 0),
                dtype=np.dtype(np.float32),
            ),
            "human_action_history": _array(
                self.human_action_history,
                name="human_action_history",
                shape=(history_length, COMMAND_DIM if self.spec.use_human_action else 0),
                dtype=np.dtype(np.float32),
                finite=False,
            ),
            "executed_action_history": _array(
                self.executed_action_history,
                name="executed_action_history",
                shape=(
                    history_length,
                    COMMAND_DIM if self.spec.use_executed_action else 0,
                ),
                dtype=np.dtype(np.float32),
            ),
            "history_timestamps": _array(
                self.history_timestamps,
                name="history_timestamps",
                shape=(history_length,),
                dtype=np.dtype(np.float64),
            ),
            "human_action_timestamps": _array(
                self.human_action_timestamps,
                name="human_action_timestamps",
                shape=(history_length,),
                dtype=np.dtype(np.float64),
                finite=False,
            ),
            "human_action_age_ms": _array(
                self.human_action_age_ms,
                name="human_action_age_ms",
                shape=(history_length,),
                dtype=np.dtype(np.float32),
                finite=False,
            ),
            "history_valid": _array(
                self.history_valid,
                name="history_valid",
                shape=(history_length,),
                dtype=np.dtype(np.bool_),
                finite=False,
            ),
            "human_action_valid": _array(
                self.human_action_valid,
                name="human_action_valid",
                shape=(history_length,),
                dtype=np.dtype(np.bool_),
                finite=False,
            ),
            "human_control_active": _array(
                self.human_control_active,
                name="human_control_active",
                shape=(history_length,),
                dtype=np.dtype(np.bool_),
                finite=False,
            ),
        }
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

        expected_rgb = set(self.spec.cameras) if self.spec.use_rgb else set()
        expected_depth = set(self.spec.cameras) if self.spec.use_depth else set()
        if set(self.rgb) != expected_rgb:
            raise ValueError(f"rgb cameras must be {sorted(expected_rgb)}")
        if set(self.depth) != expected_depth:
            raise ValueError(f"depth cameras must be {sorted(expected_depth)}")
        normalized_rgb: dict[str, NDArray[np.uint8]] = {}
        normalized_depth: dict[str, NDArray[np.float32]] = {}
        for camera, rgb in self.rgb.items():
            color = np.asarray(rgb, dtype=np.uint8)
            if color.ndim == 3:
                color = color[None, ...]
            if (
                color.ndim != 4
                or color.shape[0] != self.spec.image_history_length
                or color.shape[3] != 3
            ):
                raise ValueError(
                    f"rgb[{camera!r}] must have shape "
                    f"({self.spec.image_history_length}, H, W, 3)"
                )
            normalized_rgb[camera] = np.ascontiguousarray(color)
        for camera, raw_depth in self.depth.items():
            depth = np.asarray(raw_depth, dtype=np.float32)
            if depth.ndim == 2:
                depth = depth[None, ...]
            if depth.ndim != 3 or depth.shape[0] != self.spec.image_history_length:
                raise ValueError(
                    f"depth[{camera!r}] must have shape "
                    f"({self.spec.image_history_length}, H, W)"
                )
            if camera in normalized_rgb and depth.shape != normalized_rgb[camera].shape[:3]:
                raise ValueError(f"depth[{camera!r}] must match the RGB image size")
            normalized_depth[camera] = np.ascontiguousarray(depth)
        object.__setattr__(self, "rgb", normalized_rgb)
        object.__setattr__(self, "depth", normalized_depth)

        expected_image_cameras = expected_rgb | expected_depth
        normalized_image_times: dict[str, NDArray[np.float64]] = {}
        normalized_image_valid: dict[str, NDArray[np.bool_]] = {}
        for camera in expected_image_cameras:
            normalized_image_times[camera] = _array(
                self.image_timestamps.get(camera, []),
                name=f"image_timestamps[{camera!r}]",
                shape=(self.spec.image_history_length,),
                dtype=np.dtype(np.float64),
            )
            normalized_image_valid[camera] = _array(
                self.image_valid.get(camera, []),
                name=f"image_valid[{camera!r}]",
                shape=(self.spec.image_history_length,),
                dtype=np.dtype(np.bool_),
                finite=False,
            )
        object.__setattr__(self, "image_timestamps", normalized_image_times)
        object.__setattr__(self, "image_valid", normalized_image_valid)

    @property
    def latest_state(self) -> NDArray[np.float32]:
        if not self.spec.use_state_26:
            raise RuntimeError("this policy did not request state_26")
        return self.state_history[-1]

    @property
    def latest_human_action(self) -> NDArray[np.float32]:
        if not self.spec.use_human_action:
            raise RuntimeError("this policy did not request human actions")
        return self.human_action_history[-1]

    @property
    def latest_executed_action(self) -> NDArray[np.float32]:
        if not self.spec.use_executed_action:
            raise RuntimeError("this policy did not request executed actions")
        return self.executed_action_history[-1]

    @property
    def latest_rgb(self) -> dict[str, NDArray[np.uint8]]:
        return {camera: frames[-1] for camera, frames in self.rgb.items()}

    @property
    def latest_depth(self) -> dict[str, NDArray[np.float32]]:
        return {camera: frames[-1] for camera, frames in self.depth.items()}


@dataclass(frozen=True)
class ModelOutput:
    """Canonical result returned by every shared-control policy plugin."""

    timestamp: float
    command: NDArray[np.float64]
    command_space: CommandSpace = CommandSpace.CARTESIAN_POSE
    valid: bool = True
    control_active: bool = True
    confidence: float = 1.0
    policy_name: str = "unknown"
    fallback_used: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            command_space = CommandSpace(self.command_space)
        except ValueError as error:
            raise ValueError(f"unsupported command_space: {self.command_space}") from error
        object.__setattr__(self, "command_space", command_space)
        command = _array(
            self.command,
            name="command",
            shape=(COMMAND_DIM,),
            dtype=np.dtype(np.float64),
            finite=bool(self.valid),
        )
        object.__setattr__(self, "command", command)
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if self.valid and command_space == CommandSpace.CARTESIAN_POSE:
            quaternion_norm = float(np.linalg.norm(command[3:7]))
            if quaternion_norm < 1e-8:
                raise ValueError("Cartesian output quaternion must be non-zero")
            normalized = command.copy()
            normalized[3:7] /= quaternion_norm
            object.__setattr__(self, "command", normalized)


@runtime_checkable
class SharedPolicy(Protocol):
    """The only API a model implementation needs to provide."""

    input_spec: ModelInputSpec

    def reset(self, task_name: str) -> None:
        ...

    def predict(self, model_input: ModelInput) -> ModelOutput:
        ...
