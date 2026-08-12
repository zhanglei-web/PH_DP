from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    width: int
    height: int
    fovy_degrees: float
    intrinsic_matrix: NDArray[np.float64]
    position_world: NDArray[np.float64]
    rotation_camera_to_world: NDArray[np.float64]


class CameraSensor:
    """Reusable named-camera RGB and metric-depth renderer."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def _camera_id(self, camera_name: str) -> int:
        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if camera_id < 0:
            available = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(self.model.ncam)
            ]
            raise ValueError(f"Unknown camera '{camera_name}'. Available: {available}")
        return camera_id

    def render_rgb(self, camera_name: str = "front") -> NDArray[np.uint8]:
        self._camera_id(camera_name)
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render().copy()

    def render_depth(self, camera_name: str = "front") -> NDArray[np.float32]:
        self._camera_id(camera_name)
        self._renderer.enable_depth_rendering()
        try:
            self._renderer.update_scene(self.data, camera=camera_name)
            return self._renderer.render().astype(np.float32, copy=True)
        finally:
            self._renderer.disable_depth_rendering()

    def render_rgbd(
        self, camera_name: str = "front"
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        """Render registered RGB and metric depth with one framebuffer pass."""
        self._camera_id(camera_name)
        renderer = self._renderer
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=camera_name)
        if renderer._mjr_context is None:
            raise RuntimeError("Renderer is closed")
        if renderer._gl_context:
            renderer._gl_context.make_current()

        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        depth = np.empty((self.height, self.width), dtype=np.float32)
        mujoco.mjr_render(renderer._rect, renderer._scene, renderer._mjr_context)
        mujoco.mjr_readPixels(
            rgb, depth, renderer._rect, renderer._mjr_context
        )

        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        zfar = np.float32(far)
        znear = np.float32(near)
        c_coef = -(zfar + znear) / (zfar - znear)
        d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)
        c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
        d_coef = np.float32(-0.5) * d_coef
        depth[:] = (d_coef / (depth.astype(np.float64) + c_coef)).astype(
            np.float32
        )

        if renderer._gl_context:
            rgb[:] = np.flipud(rgb)
            depth[:] = np.flipud(depth)
        return rgb, depth

    def get_calibration(self, camera_name: str = "front") -> CameraCalibration:
        camera_id = self._camera_id(camera_name)
        fovy = float(self.model.cam_fovy[camera_id])
        focal = 0.5 * self.height / np.tan(np.deg2rad(fovy) / 2.0)
        intrinsic = np.array(
            [
                [focal, 0.0, (self.width - 1) / 2.0],
                [0.0, focal, (self.height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return CameraCalibration(
            name=camera_name,
            width=self.width,
            height=self.height,
            fovy_degrees=fovy,
            intrinsic_matrix=intrinsic,
            position_world=self.data.cam_xpos[camera_id].copy(),
            rotation_camera_to_world=self.data.cam_xmat[camera_id].reshape(3, 3).copy(),
        )

    def close(self) -> None:
        self._renderer.close()
