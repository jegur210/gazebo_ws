import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CameraTestNode(Node):
    def __init__(self):
        super().__init__('camera_test') # UI에서 찾을 노드 이름 ('/camera_test')

        # 1. 파라미터 선언 및 초기값 설정 (요청해주신 파라미터들)
        self.declare_parameter('top_x', 0.15)
        self.declare_parameter('top_y', 0.45)
        self.declare_parameter('bottom_x', 0.00)
        self.declare_parameter('bottom_y', 0.95)

        self.declare_parameter('left_search_min', 0.00)
        self.declare_parameter('left_search_max', 0.45)
        self.declare_parameter('right_search_min', 0.55)
        self.declare_parameter('right_search_max', 1.00)

        self.declare_parameter('kp', 0.003)
        self.declare_parameter('kd', 0.0005)

        # 초기 값 변수에 로드
        self.load_parameters()

        # 2. 파라미터 변경 실시간 감지(콜백) 등록
        self.add_on_set_parameters_callback(self.parameter_callback)

        # 3. 차량 속도 명령 수신을 위한 구독자(Subscriber) 생성
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.get_logger().info("🚗 Camera Test Node & Parameter Server initialized successfully.")

    def load_parameters(self):
        # 파라미터 서버에서 현재 값들을 가져와 멤버 변수에 저장
        self.top_x = self.get_parameter('top_x').value
        self.top_y = self.get_parameter('top_y').value
        self.bottom_x = self.get_parameter('bottom_x').value
        self.bottom_y = self.get_parameter('bottom_y').value

        self.left_search_min = self.get_parameter('left_search_min').value
        self.left_search_max = self.get_parameter('left_search_max').value
        self.right_search_min = self.get_parameter('right_search_min').value
        self.right_search_max = self.get_parameter('right_search_max').value

        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value

    def parameter_callback(self, params):
        # UI에서 파라미터 변경 요청이 들어올 때 자동으로 호출됨
        for param in params:
            self.get_logger().info(f"[Parameter Updated] {param.name} -> {param.value}")
        
        # 내부 변수 갱신
        self.load_parameters()
        
        from rcl_interfaces.msg import SetParametersResult
        return SetParametersResult(successful=True)

    def cmd_vel_callback(self, msg: Twist):
        # UI 슬라이더에서 전송되는 /cmd_vel 속도 데이터 수신
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        
        # 여기에 모터 제어 또는 로그 출력 로직 작성
        # self.get_logger().info(f"Received Cmd -> Linear: {linear_x:.2f}, Angular: {angular_z:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = CameraTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()