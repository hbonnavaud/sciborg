import importlib
import os
import pathlib
from launch import LaunchService, LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def launch_gazebo(world_file_path: str, headless=False):

    pkg_gazebo_ros = FindPackageShare(package='gazebo_ros').find('gazebo_ros')
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_file_path}.items())

    # Start Gazebo client
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(str(not headless).lower()))

    # Build the launch description
    launch_description = LaunchDescription()
    launch_description.add_action(start_gazebo_server_cmd)
    launch_description.add_action(start_gazebo_client_cmd)

    # Launch the launch description
    launch_service = LaunchService()
    launch_service.include_launch_description(launch_description)
    launch_service.run()
