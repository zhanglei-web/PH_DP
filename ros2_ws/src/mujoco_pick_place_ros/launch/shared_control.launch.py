import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
            str(
                venv_site_packages
                / "cmeel.prefix"
                / "lib"
                / "python3.10"
                / "site-packages"
            ),
            str(project_root / "src"),
            os.environ.get("PYTHONPATH", ""),
        )
    )
    policy_plugin = LaunchConfiguration("policy_plugin")
    policy_config_json = LaunchConfiguration("policy_config_json")
    task_name = LaunchConfiguration("task_name")
    inference_timeout_ms = LaunchConfiguration("inference_timeout_ms")
    human_command_timeout_ms = LaunchConfiguration("human_command_timeout_ms")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "policy_plugin", default_value="human_passthrough"
            ),
            DeclareLaunchArgument("policy_config_json", default_value="{}"),
            DeclareLaunchArgument("task_name", default_value="pick_place"),
            DeclareLaunchArgument("inference_timeout_ms", default_value="250.0"),
            DeclareLaunchArgument(
                "human_command_timeout_ms", default_value="250.0"
            ),
            Node(
                package="mujoco_pick_place_ros",
                executable="shared_control_node",
                namespace="mujoco",
                output="screen",
                parameters=[
                    {
                        "policy_plugin": policy_plugin,
                        "policy_config_json": ParameterValue(
                            policy_config_json, value_type=str
                        ),
                        "task_name": task_name,
                        "inference_timeout_ms": ParameterValue(
                            inference_timeout_ms, value_type=float
                        ),
                        "human_command_timeout_ms": ParameterValue(
                            human_command_timeout_ms, value_type=float
                        ),
                    }
                ],
                additional_env={"PYTHONPATH": project_python_path},
            ),
        ]
    )
