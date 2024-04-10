import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare


from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():

    # args that can be set from the command line or a default will be used
    #color_file_launch_arg = DeclareLaunchArgument(
    #    "color_file", default_value=TextSubstitution(text="0")
    #)
    declared_arguments = []
    
    joyps4_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('teleop_twist_joy'),
                'launch',
                'teleop-launch.py'
            ])
        ]),
        launch_arguments={
                'joy_vel' : '/irobot01/cmd_vel',
                'config_filepath' : os.path.join( get_package_share_directory('isae_simulations_package'), 'config', 'isae_parameters.yaml'),
            }.items()
    )
    
    image_2_scan = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='image_line_scan',
        remappings=[
            ('/depth','/ms_kinect_depth/depth/image_raw'),
            ('/depth_camera_info', '/ms_kinect_depth/depth/camera_info'),
            ],
        parameters=[{
        'scan_height': 14,
         'range_max': 7.0,
         }],
        output='screen',
    )

#TODO:
#ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 -1.5707 base_footprint kinect_depth_frame
#ros2 run tf2_ros static_transform_publisher 0 0 0 1.5707 0 0 base_footprint camera_depth_frame

    return LaunchDescription(declared_arguments+[GroupAction(
     actions=[
            joyps4_launch,
            image_2_scan,
    ])])
