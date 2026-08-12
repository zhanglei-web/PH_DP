from __future__ import annotations

import time

import mujoco.viewer
import numpy as np

from mujoco_shared_control import PickPlaceEnv


def main() -> None:
    env = PickPlaceEnv()
    env.reset(seed=0, options={"randomize_object": False})
    hold_action = np.concatenate((env.home_joint_positions, [0.08]))

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                started = time.perf_counter()
                env.step(hold_action)
                viewer.sync()
                elapsed = time.perf_counter() - started
                time.sleep(max(env.control_timestep - elapsed, 0.0))
    finally:
        env.close()


if __name__ == "__main__":
    main()

