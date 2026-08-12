import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    project_root = Path(__file__).resolve().parents[4]
    venv_site_packages = (
        project_root / ".venv" / "lib" / "python3.10" / "site-packages"
    )
    project_python_path = os.pathsep.join(
        (
            str(venv_site_packages),
            str(venv_site_packages / "cmeel.prefix" / "lib" / "python3.10" / "site-packages"),
            str(project_root / "src"),
            os.environ.get("PYTHONPATH", ""),
        )
    )
    update_rate_hz = LaunchConfiguration("update_rate_hz")
    camera_name = LaunchConfiguration("camera_name")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    publish_raw_images = LaunchConfiguration("publish_raw_images")
    publish_compressed = LaunchConfiguration("publish_compressed")
    show_mujoco_viewer = LaunchConfiguration("show_mujoco_viewer")
    show_camera_views = LaunchConfiguration("show_camera_views")
    randomize_object = LaunchConfiguration("randomize_object")
    randomize_goal = LaunchConfiguration("randomize_goal")
    mujoco_gl = LaunchConfiguration("mujoco_gl")
    dataset_dir = LaunchConfiguration("dataset_dir")
    task_name = LaunchConfiguration("task_name")

    return LaunchDescription(
        [
            DeclareLaunchArgument("update_rate_hz", default_value="20"),
            DeclareLaunchArgument("camera_name", default_value="front"),
            DeclareLaunchArgument("camera_width", default_value="640"),
            DeclareLaunchArgument("camera_height", default_value="480"),
            DeclareLaunchArgument("publish_raw_images", default_value="false"),
            DeclareLaunchArgument("publish_compressed", default_value="true"),
            DeclareLaunchArgument("show_mujoco_viewer", default_value="true"),
            DeclareLaunchArgument("show_camera_views", default_value="true"),
            DeclareLaunchArgument("randomize_object", default_value="true"),
            DeclareLaunchArgument("randomize_goal", default_value="true"),
            DeclareLaunchArgument("mujoco_gl", default_value="egl"),
            DeclareLaunchArgument(
                "dataset_dir", default_value=str(project_root / "datasets")
            ),
            DeclareLaunchArgument("task_name", default_value="pick_place"),
            Node(
                package="mujoco_pick_place_ros",
                executable="bridge_node",
                namespace="mujoco",
                output="screen",
                parameters=[
                    {
                        "update_rate_hz": ParameterValue(update_rate_hz, value_type=int),
                        "camera_name": camera_name,
                        "camera_width": ParameterValue(camera_width, value_type=int),
                        "camera_height": ParameterValue(camera_height, value_type=int),
                        "publish_raw_images": ParameterValue(
                            publish_raw_images, value_type=bool
                        ),
                        "publish_compressed": ParameterValue(
                            publish_compressed, value_type=bool
                        ),
                        "show_mujoco_viewer": ParameterValue(
                            False, value_type=bool
                        ),
                        "randomize_object": ParameterValue(
                            randomize_object, value_type=bool
                        ),
                        "randomize_goal": ParameterValue(
                            randomize_goal, value_type=bool
                        ),
                        "dataset_dir": dataset_dir,
                        "task_name": task_name,
                    }
                ],
                additional_env={
                    "MUJOCO_GL": mujoco_gl,
                    "PYTHONPATH": project_python_path,
                },
            ),
            Node(
                package="mujoco_pick_place_ros",
                executable="viewer_node",
                namespace="mujoco",
                output="screen",
                condition=IfCondition(show_mujoco_viewer),
                additional_env={
                    "MUJOCO_GL": "glfw",
                    "PYTHONPATH": project_python_path,
                },
            ),
            Node(
                package="mujoco_pick_place_ros",
                executable="rgbd_viewer_node",
                namespace="mujoco",
                output="screen",
                condition=IfCondition(show_camera_views),
                additional_env={"PYTHONPATH": project_python_path},
            ),
        ]
    )
