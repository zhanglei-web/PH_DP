from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mujoco_shared_control import PickPlaceEnv


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = PickPlaceEnv(camera_width=640, camera_height=480)
    try:
        env.reset(seed=5, options={"randomize_object": False})
        rgb = env.render_rgb(camera_name="front")
        depth = env.render_depth(camera_name="front")
        calibration = env.get_camera_calibration("front")

        rgb_path = output_dir / "front_rgb.png"
        depth_npy_path = output_dir / "front_depth.npy"
        depth_png_path = output_dir / "front_depth.png"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_npy_path, depth)

        finite = np.isfinite(depth)
        if not finite.any():
            raise RuntimeError("Depth renderer returned no finite pixels")
        near, far = np.percentile(depth[finite], [1, 99])
        normalized = np.clip((depth - near) / max(far - near, 1e-6), 0.0, 1.0)
        depth_visual = ((1.0 - normalized) * 255.0).astype(np.uint8)
        Image.fromarray(depth_visual).save(depth_png_path)

        print("camera_name:", calibration.name)
        print("resolution:", calibration.width, "x", calibration.height)
        print("fovy_degrees:", calibration.fovy_degrees)
        print("intrinsic_matrix:\n", calibration.intrinsic_matrix)
        print("position_world:", calibration.position_world)
        print("rotation_camera_to_world:\n", calibration.rotation_camera_to_world)
        print("rgb:", rgb.shape, rgb.dtype, rgb_path)
        print("depth:", depth.shape, depth.dtype, float(depth.min()), float(depth.max()))
        print("depth_outputs:", depth_npy_path, depth_png_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()

