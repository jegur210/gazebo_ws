import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraSensorTrackingControl(Node):
    def __init__(self):
        super().__init__('camera_sensor_tracking_control')
        
        # 카메라 영상 구독 (Topic: /camera/image_raw)
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # 로봇 제어 명령 발행 (Topic: /cmd_vel)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # ROS Image 메시지 <-> OpenCV 이미지 변환 객체
        self.br = CvBridge()

        # [Gazebo 환경 설정]
        # CLAHE 사용 여부 플래그 (시뮬레이션에서는 False 권장, 필요시 켜기)
        self.use_clahe = False
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def preprocess_image(self, frame):
        """
        Gazebo 시뮬레이션 최적화 전처리 파이프라인
        1. Gaussian Blur (선택적 가벼운 노이즈 제거)
        2. 조명 보정 (CLAHE - 옵션)
        """
        # 1. 가우시안 블러 (선 경계 손실을 줄이기 위해 커널 크기를 3x3으로 축소)
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)

        # 2. CLAHE (시뮬레이션에서는 대개의 경우 불필요하므로 플래그로 분기)
        if self.use_clahe:
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            cl_channel = self.clahe.apply(l_channel)
            merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
            preprocessed_frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        else:
            preprocessed_frame = blurred

        return preprocessed_frame

    def extract_lane_masks(self, bgr_frame):
        """
        Gazebo SDF 맵 재질(Material) 스펙트럼 기준 HSV 마스킹
        - White Lane: SDF RGB(1.0, 1.0, 1.0)
        - Yellow Line: SDF RGB(1.0, 0.8, 0.0)
        """
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        # 1. 노란색 중앙선 마스크 (SDF: 1.0, 0.8, 0.0)
        lower_yellow = np.array([15, 120, 150])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 2. 흰색 차선 마스크 (SDF: 1.0, 1.0, 1.0)
        # S(채도)는 낮고, V(명도)는 높은 영역
        lower_white = np.array([0, 0, 190])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 노란색과 흰색 마스크 합성
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

        return combined_mask, yellow_mask, white_mask

    def image_callback(self, msg):
        # 1. ROS Image 메시지를 OpenCV 이미지(BGR8)로 변환
        frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. 영상 전처리 수행
        preprocessed_frame = self.preprocess_image(frame)
        
        # 3. 차선 마스킹 (Gazebo SDF 색상 기반 추출)
        combined_mask, yellow_mask, white_mask = self.extract_lane_masks(preprocessed_frame)

        # ===================================================
        # TODO: 다음 단계 (ROI 추출 & Bird's Eye View 변환 및 Sliding Window)
        # ===================================================
        
        # 디버깅용 모니터링 출력
        cv2.imshow("1. Original Camera View", frame)
        cv2.imshow("2. Combined Lane Mask", combined_mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CameraSensorTrackingControl()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()