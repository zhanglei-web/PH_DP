from glob import glob
from setuptools import find_packages, setup


package_name = "mujoco_pick_place_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["h5py", "setuptools"],
    zip_safe=True,
    maintainer="ray",
    maintainer_email="ray@example.com",
    description="ROS 2 bridge for MuJoCo Franka pick-and-place",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bridge_node = mujoco_pick_place_ros.bridge_node:main",
            "collection_control_node = mujoco_pick_place_ros.collection_control_node:main",
            "rgbd_viewer_node = mujoco_pick_place_ros.rgbd_viewer_node:main",
            "shared_control_node = mujoco_pick_place_ros.shared_control_node:main",
            "viewer_node = mujoco_pick_place_ros.viewer_node:main",
            "xrt_teleop_node = mujoco_pick_place_ros.vr_teleop_node:main",
        ],
    },
)
