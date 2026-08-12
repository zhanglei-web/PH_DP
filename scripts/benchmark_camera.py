from __future__ import annotations

from io import BytesIO
import time

import numpy as np
from PIL import Image

from mujoco_shared_control import PickPlaceEnv


def milliseconds_per_call(function, iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    return 1000.0 * (time.perf_counter() - started) / iterations


def main() -> None:
    env = PickPlaceEnv(camera_width=640, camera_height=480)
    try:
        env.reset(seed=0, options={"randomize_object": False})
        for _ in range(5):
            rgb, depth = env.render_rgbd("front")

        render_ms = milliseconds_per_call(lambda: env.render_rgbd("front"), 30)

        def encode_color() -> None:
            buffer = BytesIO()
            Image.fromarray(rgb, mode="RGB").save(
                buffer, format="JPEG", quality=90
            )

        def encode_depth() -> None:
            depth_mm = np.clip(
                np.rint(depth * 1000.0), 0, np.iinfo(np.uint16).max
            ).astype(np.uint16)
            buffer = BytesIO()
            Image.fromarray(depth_mm, mode="I;16").save(
                buffer, format="PNG", compress_level=1
            )

        jpeg_ms = milliseconds_per_call(encode_color, 30)
        depth_png_ms = milliseconds_per_call(encode_depth, 30)
        print(f"rgbd_render_ms: {render_ms:.2f}")
        print(f"jpeg_encode_ms: {jpeg_ms:.2f}")
        print(f"depth_png_encode_ms: {depth_png_ms:.2f}")
        print(f"serial_pipeline_hz: {1000.0 / (render_ms + jpeg_ms + depth_png_ms):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
