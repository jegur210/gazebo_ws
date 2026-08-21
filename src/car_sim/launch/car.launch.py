import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 1. 사용할 패키지 이름 정의
    package_name = "car_sim"

    # 2. 선언: "use_sim_time" 설정
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    # 3. 로봇 설계도(Xacro) 준비
    pkg_path = os.path.join(get_package_share_directory(package_name))
    xacro_file = os.path.join(pkg_path, "urdf", "car.xacro")
    robot_description = xacro.process_file(xacro_file)
    param_dir = os.path.join(pkg_path, 'param', 'bev_param.yaml')
    
    # 4. 로봇 상태 발행기(Robot State Publisher)
    params = {"robot_description": robot_description.toxml(), "use_sim_time": use_sim_time}
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[params],
    )

    world_file = os.path.join(pkg_path, "worlds", "control_world.sdf")

    # 5. 가상 세계(Gazebo Sim) 실행 설정
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")]
        ),
        # 빈 월드 대신 city_world.sdf를 열도록 수정
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # 6. 로봇 소환 (Gazebo 내에 모델 스폰)
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", "robot_description", "-name", "with_robot"],
        output="screen",
    )

    # ROS 2의 Twist 메시지와 Gazebo의 Twist 메시지를 매핑하여 조종 명령을 전달합니다.
    # 7. ROS 2 - Gazebo 양방향/단방향 통신 브리지 설정
    # 7-1. 일반 제어/상태 토픽 브리지
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
        ],
        output='screen'
    )

    # 7-2. 카메라 이미지 전용 브리지 (안정적인 이미지 변환)
    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera/image_raw'],
        output='screen'
    )

    lane_tracker_node = Node(
        package=package_name,                          # 패키지 이름 (car_sim)
        executable="camera_test",    # 실행파일명 (또는 setup.py의 entry_points)
        name="camera_test",
        output="screen",
        parameters=[
            param_dir,                                 # 불러올 YAML 파일 경로
            {"use_sim_time": use_sim_time}             # 추가 파라미터를 딕셔너리로 병합 가능
        ]
    )

    # LaunchDescription return에 image_bridge 추가
    return LaunchDescription([
        declare_use_sim_time,
        node_robot_state_publisher,
        gazebo,
        spawn_entity,
        bridge,
        image_bridge, # 추가
        lane_tracker_node
    ])