"""
    Author: Jason Hughes
    Date: November 2024

    About: launch depth_pro ros2 node
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    share_dir = get_package_share_directory("depth_pro_ros")

    intrinsics_path = os.path.join(share_dir, 'blackfly-deimos.yaml')
    model_path = os.path.join(share_dir, 'depth_pro.pt')

    return LaunchDescription([
        Node(
            package='depth_pro_ros',
            executable='depth_pro',
            name='depth_pro',
            output='screen',
            remappings=[
                ("/image/compressed", "/flir_camera/image_raw/compressed")
            ],
            parameters=[
                {'intrinsics_path': intrinsics_path},
                {'model_path': model_path},
            ]
        )
    ])
