import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, FloatingPointRange
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraSensorTrackingControl(Node):
    def __init__(self):
        super().__init__('camera_sensor_tracking_control')
        
        # 1. ROS 2 파라미터 선언 (YAML 파일 및 rqt_reconfigure 호환)
        range_desc = ParameterDescriptor(
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)]
        )
        self.declare_parameter('top_x', 0.15, range_desc)
        self.declare_parameter('top_y', 0.45, range_desc)
        self.declare_parameter('bottom_x', 0.00, range_desc)
        self.declare_parameter('bottom_y', 0.95, range_desc)

        # YAML 또는 Launch에서 로드된 초기 파라미터 가져오기
        self.top_x = self.get_parameter('top_x').value
        self.top_y = self.get_parameter('top_y').value
        self.bottom_x = self.get_parameter('bottom_x').value
        self.bottom_y = self.get_parameter('bottom_y').value

        # 실시간 파라미터 변경 콜백 등록
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # ROS 2 Pub/Sub
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.br = CvBridge()

        # BEV 출력 해상도
        self.bev_w = 640
        self.bev_h = 480

    def parameter_callback(self, params):
        """rqt_reconfigure 또는 CLI 명령어로 파라미터 변경 시 실시간 반영"""
        for param in params:
            if param.name == 'top_x': self.top_x = param.value
            elif param.name == 'top_y': self.top_y = param.value
            elif param.name == 'bottom_x': self.bottom_x = param.value
            elif param.name == 'bottom_y': self.bottom_y = param.value
        return SetParametersResult(successful=True)

    def bird_eye_view(self, img):
        """시점 변환 (Bird's Eye View)"""
        h, w = img.shape[:2]
        src_pts = np.float32([
            [w * self.top_x, h * self.top_y],
            [w * (1.0 - self.top_x), h * self.top_y],
            [w * (1.0 - self.bottom_x), h * self.bottom_y],
            [w * self.bottom_x, h * self.bottom_y]
        ])
        dst_pts = np.float32([
            [0, 0], [self.bev_w, 0],
            [self.bev_w, self.bev_h], [0, self.bev_h]
        ])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, matrix, (self.bev_w, self.bev_h))
        return warped, src_pts

    def binarize_lane(self, bgr_frame):
        """
        차선 이진화 파이프라인 (Color Thresholding + Sobel X Gradient + Morphology)
        """
        # 1. 노이즈 제거
        blurred = cv2.GaussianBlur(bgr_frame, (5, 5), 0)

        # 2. 색상 마스킹 (HSV)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 노란색 중앙선 추출
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 흰색 차선 추출
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 45, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        color_binary = cv2.bitwise_or(yellow_mask, white_mask)

        # 3. Sobel X 기울기(Gradient) 마스킹 (수직 차선 경계선 강조)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        
        # 정규화 (0~255)
        max_sobel = np.max(abs_sobelx)
        if max_sobel > 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel)
        else:
            scaled_sobel = np.uint8(abs_sobelx)
        
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 255

        # 4. Color 마스크와 Sobel 마스크 결합 (OR 연산)
        combined_binary = cv2.bitwise_or(color_binary, sobel_binary)

        # 5. 모폴로지 연산 (차선 내 미세한 구멍 채우기)
        kernel = np.ones((3, 3), np.uint8)
        cleaned_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)

        return cleaned_binary

    def draw_roi_polyline(self, frame, src_pts):
        """디버깅용 ROI 사다리꼴 가시화"""
        copied_frame = frame.copy()
        pts = src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(copied_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return copied_frame

    def image_callback(self, msg):
        # 1. ROS Image -> OpenCV BGR 변환
        frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. BEV 시점 변환 먼저 적용 (계산 효율화)
        warped_bgr, src_pts = self.bird_eye_view(frame)
        
        # 3. BEV 영상 상에서 이진화 수행
        binary_bev = self.binarize_lane(warped_bgr)

        # 4. 원본 비주얼용 ROI 폴리라인
        roi_visualized = self.draw_roi_polyline(frame, src_pts)

        # =========================================================
        # TODO: 다음 단계 (Sliding Window & Polynomial Fitting)
        # binary_bev 영상을 입력값으로 사용하게 됩니다.
        # =========================================================

        # 디버깅 창 출력
        cv2.imshow("1. ROI Polyline View", roi_visualized)
        cv2.imshow("2. BEV Color View", warped_bgr)
        cv2.imshow("3. Final Binary BEV", binary_bev)
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