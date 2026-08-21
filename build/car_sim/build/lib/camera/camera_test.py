import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraSensorTrackingControl(Node):
    def __init__(self):
        super().__init__('camera_sensor_tracking_control')
        
        # 1. ROS 2 파라미터 선언 (ROI 튜닝용 비율 값: 0.0 ~ 1.0)
        self.declare_parameter('top_x', 0.20)      # 상단 좌우 여백 비율 (0.20 -> 좌: 20%, 우: 80%)
        self.declare_parameter('top_y', 0.50)      # 상단 Y 높이 비율 (0.50 -> 화면 중앙)
        self.declare_parameter('bottom_x', 0.02)   # 하단 좌우 여백 비율 (0.02 -> 좌: 2%, 우: 98%)
        self.declare_parameter('bottom_y', 0.95)   # 하단 Y 높이 비율 (0.95 -> 화면 맨 아래쪽)

        # 파라미터 변수 읽기
        self.top_x = self.get_parameter('top_x').value
        self.top_y = self.get_parameter('top_y').value
        self.bottom_x = self.get_parameter('bottom_x').value
        self.bottom_y = self.get_parameter('bottom_y').value

        # 실시간 파라미터 변경 감지 콜백 등록
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # 카메라 영상 구독 (/camera/image_raw)
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # 제어 명령 발행 (/cmd_vel)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.br = CvBridge()

        # [변환 후 출력할 탑뷰 이미지 크기 설정]
        self.bev_w = 640
        self.bev_h = 480

    def parameter_callback(self, params):
        """터미널이나 GUI에서 파라미터 변경 시 즉시 반영되는 콜백"""
        for param in params:
            if param.name == 'top_x':
                self.top_x = param.value
            elif param.name == 'top_y':
                self.top_y = param.value
            elif param.name == 'bottom_x':
                self.bottom_x = param.value
            elif param.name == 'bottom_y':
                self.bottom_y = param.value
        
        self.get_logger().info(
            f"ROI 파라미터 업데이트 -> top_x: {self.top_x:.2f}, top_y: {self.top_y:.2f}, "
            f"bottom_x: {self.bottom_x:.2f}, bottom_y: {self.bottom_y:.2f}"
        )
        return SetParametersResult(successful=True)

    def preprocess_image(self, frame):
        """가우시안 블러 전처리"""
        return cv2.GaussianBlur(frame, (3, 3), 0)

    def extract_lane_masks(self, bgr_frame):
        """Gazebo SDF 색상 스펙트럼 기준 HSV 차선 마스킹"""
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        # 1. 노란색 중앙선 (SDF: 1.0, 0.8, 0.0)
        lower_yellow = np.array([15, 120, 150])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 2. 흰색 차선 (SDF: 1.0, 1.0, 1.0)
        lower_white = np.array([0, 0, 190])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 합성 마스크
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
        return combined_mask

    def bird_eye_view(self, img):
        h, w = img.shape[:2]

        # =========================================================
        # [파라미터 기반 src_pts] 
        # 좌우 대칭을 보장하면서 파라미터로 실시간 좌표 변경
        # =========================================================
        src_pts = np.float32([
            [w * self.top_x, h * self.top_y],                # Top-Left
            [w * (1.0 - self.top_x), h * self.top_y],        # Top-Right
            [w * (1.0 - self.bottom_x), h * self.bottom_y],   # Bottom-Right
            [w * self.bottom_x, h * self.bottom_y]           # Bottom-Left
        ])

        dst_pts = np.float32([
            [0, 0],
            [self.bev_w, 0],
            [self.bev_w, self.bev_h],
            [0, self.bev_h]
        ])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, matrix, (self.bev_w, self.bev_h))

        return warped, src_pts
    
    def draw_roi_polyline(self, frame, src_pts):
        """원본 카메라이미지에 ROI 사다리꼴 다각형을 시각화 (디버깅용)"""
        copied_frame = frame.copy()
        pts = src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(copied_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return copied_frame

    def image_callback(self, msg):
        # 1. ROS Image -> OpenCV BGR 변환
        frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. 전처리
        preprocessed_frame = self.preprocess_image(frame)
        
        # 3. HSV 마스크 생성
        combined_mask = self.extract_lane_masks(preprocessed_frame)

        # 4. Bird's Eye View 변환 (마스크 영상 및 BGR 원본 영상 모두 변환 가능)
        warped_mask, src_pts = self.bird_eye_view(combined_mask)
        warped_bgr, _ = self.bird_eye_view(frame)

        # 5. 디버깅용 ROI 사다리꼴 가시화
        roi_visualized = self.draw_roi_polyline(frame, src_pts)

        # 디버깅 출력
        cv2.imshow("1. ROI Polyline View", roi_visualized)
        cv2.imshow("2. BEV Masked View", warped_mask)
        cv2.imshow("3. BEV Color View", warped_bgr)
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