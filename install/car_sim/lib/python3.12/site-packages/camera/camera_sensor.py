import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist  # 로봇 속도 제어 메시지 추가
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraSensorTrackingControl(Node):
    def __init__(self):
        super().__init__('camera_sensor_tracking_control')
        
        # 1. 카메라 구독 설정
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # 2. 로봇 제어 명령(cmd_vel) 발행 설정
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.br = CvBridge()

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 노란색 중앙선 마스크
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 흰색 차선 마스크
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([179, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # 관심 영역(ROI) 설정
        mask_yellow[0:int(h * 0.6), 0:w] = 0
        mask_white[0:int(h * 0.6), 0:w] = 0
        
        dot_y = int(h * 0.8)
        screen_center_x = int(w / 2)
        
        # 화면 중앙에 빨간 점 표시
        cv2.circle(frame, (screen_center_x, dot_y), 5, (0, 0, 255), -1)

        M_yellow = cv2.moments(mask_yellow)
        M_white = cv2.moments(mask_white)
        
        cx_yellow = None
        if M_yellow['m00'] > 0:
            cx_yellow = int(M_yellow['m10'] / M_yellow['m00'])
            
        cx_white = None
        if M_white['m00'] > 0:
            cx_white = int(M_white['m10'] / M_white['m00'])
            
        # 3. 로봇 이동 명령 초기화 (기본값: 정지)
        twist = Twist()
        
        # 두 차선이 모두 감지되었을 때 제어 수행
        if cx_yellow is not None and cx_white is not None:
            # 목표 지점 (초록점)
            lane_center_x = int((cx_yellow + cx_white) / 2)
            cv2.circle(frame, (lane_center_x, dot_y), 5, (0, 255, 0), -1)

            # 4. P 제어 (비례 제어) 연산
            # 중심 오차 계산 (ROS 표준에 맞추어 좌회전이 +z, 우회전이 -z 가 되도록 방향 설정)
            error = screen_center_x - lane_center_x
            
            # Kp: 비례 제어 상수 (이 값을 조절하여 핸들을 꺾는 민감도를 설정)
            Kp = 0.005 
            
            # 로봇의 전진 속도 설정
            twist.linear.x = 0.15 
            
            # 오차에 비례하여 회전 속도 대입
            twist.angular.z = float(error * Kp)
            
            # (선택) 시각적으로 오차를 보기 위해 빨간점에서 초록점까지 선 긋기
            cv2.line(frame, (screen_center_x, dot_y), (lane_center_x, dot_y), (255, 255, 0), 2)
            
        else:
            # 차선이 둘 중 하나라도 안 보이면 안전을 위해 정지
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        # 5. 최종 계산된 속도 명령을 로봇에게 전달
        self.publisher_.publish(twist)

        # 화면 출력
        mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
        cv2.imshow("Camera Tracking Control", frame)
        cv2.imshow("Mask View", mask_combined)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CameraSensorTrackingControl()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()