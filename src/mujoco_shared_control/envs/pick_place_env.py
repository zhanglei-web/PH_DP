from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from mujoco_shared_control.control.ik_controller import IKController
from mujoco_shared_control.control.joint_controller import JointPositionController
from mujoco_shared_control.robots.franka import FrankaRobot
from mujoco_shared_control.sensors.camera import CameraCalibration, CameraSensor
from mujoco_shared_control.sensors.observation import ObservationReader
from mujoco_shared_control.tasks.pick_place import PickPlaceTask


SCENE_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "menagerie"
    / "franka_emika_panda"
    / "pick_place_scene.xml"
)

DEFAULT_OBJECT_XY = np.array([0.50, 0.0], dtype=np.float64)
DEFAULT_GOAL_XY = np.array([0.55, -0.22], dtype=np.float64)
OBJECT_XY_LOW = np.array([0.46, -0.06], dtype=np.float64)
OBJECT_XY_HIGH = np.array([0.54, 0.06], dtype=np.float64)
GOAL_XY_LOW = np.array([0.50, -0.27], dtype=np.float64)
GOAL_XY_HIGH = np.array([0.60, -0.17], dtype=np.float64)
MIN_OBJECT_GOAL_DISTANCE = 0.16
OBJECT_RESET_HEIGHT = 0.25
GOAL_HEIGHT = 0.245
MAX_RESET_SAMPLING_ATTEMPTS = 1_000
DEFAULT_ARM_JOINT_NOISE_RAD = np.array(
    [0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06], dtype=np.float64
)


class PickPlaceEnv(gym.Env[dict[str, Any], NDArray[np.float64]]):
    """Standalone Franka pick-and-place environment with stable adapter boundaries."""

    metadata = {"render_modes": ["rgb_array", "depth_array", "human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        control_timestep: float = 0.02,
        max_episode_steps: int = 500,
        camera_width: int = 640,
        camera_height: int = 480,
        enable_camera: bool = True,
    ) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        ratio = control_timestep / self.model.opt.timestep
        self.frame_skip = int(round(ratio))
        if self.frame_skip < 1 or not np.isclose(
            self.frame_skip * self.model.opt.timestep, control_timestep
        ):
            raise ValueError("control_timestep must be an integer multiple of model timestep")
        self.control_timestep = self.frame_skip * self.model.opt.timestep

        self.robot = FrankaRobot(self.model, self.data)
        self.joint_controller = JointPositionController(self.robot)
        self.ik_controller = IKController(self.robot)
        self.observation_reader = ObservationReader(self.model, self.data, self.robot)
        self.task = PickPlaceTask()
        self.camera = (
            CameraSensor(
                self.model, self.data, width=camera_width, height=camera_height
            )
            if enable_camera
            else None
        )

        self.action_space = self.joint_controller.action_space
        self.observation_space = self._build_observation_space()
        self.policy_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )

        self._home_keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        self._object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint"
        )
        self._goal_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )
        if (
            self._home_keyframe_id < 0
            or self._object_joint_id < 0
            or self._goal_body_id < 0
        ):
            raise ValueError(
                "Scene must define home keyframe, object_freejoint, and goal body"
            )
        self._goal_mocap_id = int(self.model.body_mocapid[self._goal_body_id])
        if self._goal_mocap_id < 0:
            raise ValueError("Scene goal body must be a mocap body")
        self._object_qpos_address = int(self.model.jnt_qposadr[self._object_joint_id])
        self._object_dof_address = int(self.model.jnt_dofadr[self._object_joint_id])
        self._episode_steps = 0
        self._viewer = None

    def _build_observation_space(self) -> spaces.Dict:
        pose_space = spaces.Box(-np.inf, np.inf, shape=(4, 4), dtype=np.float64)
        vector3 = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
        return spaces.Dict(
            {
                "q_obs": spaces.Box(
                    self.robot.arm_joint_limits[:, 0],
                    self.robot.arm_joint_limits[:, 1],
                    dtype=np.float64,
                ),
                "dq_obs": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "ee_pose": pose_space,
                "gripper": spaces.Box(0.0, 0.08, shape=(1,), dtype=np.float64),
                "gripper_joint_positions": spaces.Box(
                    self.robot.gripper_joint_limits[:, 0],
                    self.robot.gripper_joint_limits[:, 1],
                    dtype=np.float64,
                ),
                "gripper_joint_velocities": spaces.Box(
                    -np.inf, np.inf, shape=(2,), dtype=np.float64
                ),
                "object_pose": pose_space,
                "goal_pose": pose_space,
                "object_linear_velocity": vector3,
                "object_angular_velocity": vector3,
                "contact": spaces.Dict(
                    {
                        "left": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                        "right": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                        "left_force": spaces.Box(
                            0.0, np.inf, shape=(1,), dtype=np.float32
                        ),
                        "right_force": spaces.Box(
                            0.0, np.inf, shape=(1,), dtype=np.float32
                        ),
                        "count": spaces.Box(
                            0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
                        ),
                    }
                ),
                "object_grasped": spaces.Discrete(2),
                "timestamp": spaces.Box(
                    0.0, np.inf, shape=(1,), dtype=np.float64
                ),
            }
        )

    @property
    def home_joint_positions(self) -> NDArray[np.float64]:
        return self.model.key_qpos[
            self._home_keyframe_id, self.robot.arm_qpos_indices
        ].copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else options
        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self._home_keyframe_id
        )

        arm_joint_position = self._reset_arm_position(options)
        self.data.qpos[self.robot.arm_qpos_indices] = arm_joint_position
        self.data.qvel[self.robot.arm_dof_indices] = 0.0

        object_xy, goal_xy = self._reset_positions(options)
        object_qpos = self.data.qpos[
            self._object_qpos_address : self._object_qpos_address + 7
        ]
        object_qpos[:3] = [object_xy[0], object_xy[1], OBJECT_RESET_HEIGHT]
        object_qpos[3:] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[
            self._object_dof_address : self._object_dof_address + 6
        ] = 0.0
        self.data.mocap_pos[self._goal_mocap_id] = [
            goal_xy[0],
            goal_xy[1],
            GOAL_HEIGHT,
        ]
        self.data.mocap_quat[self._goal_mocap_id] = [1.0, 0.0, 0.0, 0.0]

        self.robot.set_joint_position_target(arm_joint_position)
        self.robot.set_gripper_command(0.08)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._episode_steps = 0

        obs = self.get_observation()
        return obs, {
            "policy_obs": self.get_policy_observation(obs),
            "joint_metadata": self.robot.metadata,
            "object_xy": object_xy.copy(),
            "goal_xy": goal_xy.copy(),
            "arm_joint_position": arm_joint_position.copy(),
        }

    def _reset_arm_position(self, options: dict[str, Any]) -> NDArray[np.float64]:
        if "arm_joint_position" in options:
            position = np.asarray(options["arm_joint_position"], dtype=np.float64)
            if position.shape != (7,) or not np.isfinite(position).all():
                raise ValueError("options['arm_joint_position'] must be 7 finite values")
        elif bool(options.get("randomize_arm", False)):
            scale = float(options.get("arm_joint_noise_scale", 1.0))
            if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
                raise ValueError("arm_joint_noise_scale must be finite and in [0, 1]")
            noise = self.np_random.uniform(-1.0, 1.0, size=7)
            position = self.home_joint_positions + scale * DEFAULT_ARM_JOINT_NOISE_RAD * noise
        else:
            position = self.home_joint_positions
        return np.clip(
            position,
            self.robot.arm_joint_limits[:, 0],
            self.robot.arm_joint_limits[:, 1],
        ).copy()

    def _reset_positions(
        self, options: dict[str, Any]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        object_xy = self._xy_option(options, "object_xy", DEFAULT_OBJECT_XY)
        goal_xy = self._xy_option(options, "goal_xy", DEFAULT_GOAL_XY)
        randomize_object = bool(options.get("randomize_object", True))
        randomize_goal = bool(options.get("randomize_goal", randomize_object))

        if not randomize_object and not randomize_goal:
            return object_xy, goal_xy

        for _ in range(MAX_RESET_SAMPLING_ATTEMPTS):
            candidate_object_xy = (
                self.np_random.uniform(OBJECT_XY_LOW, OBJECT_XY_HIGH)
                if randomize_object
                else object_xy
            )
            candidate_goal_xy = (
                self.np_random.uniform(GOAL_XY_LOW, GOAL_XY_HIGH)
                if randomize_goal
                else goal_xy
            )
            if (
                np.linalg.norm(candidate_object_xy - candidate_goal_xy)
                >= MIN_OBJECT_GOAL_DISTANCE
            ):
                return candidate_object_xy, candidate_goal_xy

        raise RuntimeError(
            "Unable to sample safe object and goal positions; check the reset bounds"
        )

    @staticmethod
    def _xy_option(
        options: dict[str, Any], name: str, default: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        value = np.asarray(options.get(name, default), dtype=np.float64).copy()
        if value.shape != (2,):
            raise ValueError(f"options['{name}'] must have shape (2,)")
        if not np.isfinite(value).all():
            raise ValueError(f"options['{name}'] must contain finite values")
        return value

    def step(
        self, action: ArrayLike
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        applied_action = self.joint_controller.apply(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._episode_steps += 1

        obs = self.get_observation()
        reward, success, task_info = self.task.evaluate(obs)
        truncated = self._episode_steps >= self.max_episode_steps
        info = {
            **task_info,
            "applied_action": applied_action,
            "policy_obs": self.get_policy_observation(obs),
            "contact_details": self.observation_reader.get_contact_details(),
        }
        if self.render_mode == "human":
            self.render()
        return obs, reward, success, truncated, info

    def set_joint_position_target(self, q_cmd: ArrayLike) -> NDArray[np.float64]:
        return self.robot.set_joint_position_target(q_cmd)

    def set_gripper_command(self, g_cmd: float | ArrayLike) -> float:
        return self.robot.set_gripper_command(g_cmd)

    def set_ee_target(self, target_pose: ArrayLike) -> NDArray[np.float64]:
        q_cmd = self.ik_controller.solve(target_pose)
        return self.set_joint_position_target(q_cmd)

    def forward_kinematics(self, q_cmd: ArrayLike) -> NDArray[np.float64]:
        return self.ik_controller.forward_kinematics(q_cmd)

    def get_observation(self) -> dict[str, Any]:
        return self.observation_reader.get_observation()

    def get_policy_observation(
        self, raw_obs: dict[str, Any] | None = None
    ) -> NDArray[np.float32]:
        return self.observation_reader.get_policy_observation(raw_obs)

    def render_rgb(self, camera_name: str = "front") -> NDArray[np.uint8]:
        return self._camera_sensor().render_rgb(camera_name)

    def render_depth(self, camera_name: str = "front") -> NDArray[np.float32]:
        return self._camera_sensor().render_depth(camera_name)

    def render_rgbd(
        self, camera_name: str = "front"
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        return self._camera_sensor().render_rgbd(camera_name)

    def get_camera_calibration(
        self, camera_name: str = "front"
    ) -> CameraCalibration:
        return self._camera_sensor().get_calibration(camera_name)

    def _camera_sensor(self) -> CameraSensor:
        if self.camera is None:
            raise RuntimeError("This environment was created with enable_camera=False")
        return self.camera

    def render(self) -> NDArray[np.uint8] | NDArray[np.float32] | None:
        if self.render_mode in {None, "rgb_array"}:
            return self.render_rgb("front")
        if self.render_mode == "depth_array":
            return self.render_depth("front")
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        raise RuntimeError(f"Invalid render mode {self.render_mode}")

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
