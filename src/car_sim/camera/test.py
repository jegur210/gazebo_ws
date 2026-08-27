import rclpy  # ROS 2 Python 클라이언트 라이브러리 임포트
from rclpy.node import Node  # ROS 2 노드 클래스 임포트
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, FloatingPointRange  # 동적 파라미터 설정을 위한 메시지 타입
from sensor_msgs.msg import Image  # ROS 2 카메라 이미지 메시지 타입
from geometry_msgs.msg import Twist  # 로봇 속도 제어 메시지 타입 (cmd_vel)
from cv_bridge import CvBridge  # ROS 이미지 메시지를 OpenCV 이미지(numpy array)로 변환해주는 브릿지
import cv2  # OpenCV 영상처리 라이브러리
import numpy as np  # 행렬 및 수치 계산을 위한 NumPy 라이브러리

# ROS 2 노드 클래스 정의
class CameraSensorTrackingControl(Node):
    def __init__(self):
        # 부모 클래스(Node) 초기화 및 노드 이름 설정
        super().__init__('camera_sensor_tracking_control')
        
        # 1. ROS 2 파라미터 선언 (BEV 원근 변환용 파라미터)
        # 파라미터 입력 범위를 0.0 ~ 1.0으로 제한하는 속성 정의
        range_desc = ParameterDescriptor(
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)]
        )
        # BEV 변환에 사용될 사각형 영역 비율 파라미터 등록 (기본값 설정)
        self.declare_parameter('top_x', 0.15, range_desc)      # 상단 X축 여백 비율
        self.declare_parameter('top_y', 0.45, range_desc)      # 상단 Y축 위치 비율 (자르는 높이)
        self.declare_parameter('bottom_x', 0.00, range_desc)   # 하단 X축 여백 비율
        self.declare_parameter('bottom_y', 0.95, range_desc)   # 하단 Y축 위치 비율

        # 슬라이딩 윈도우 시작점(차선 바닥 위치) 탐색 제한 영역 파라미터 선언
        self.declare_parameter('left_search_min', 0.10, range_desc)   # 왼쪽 차선 탐색 시작 X비율
        self.declare_parameter('left_search_max', 0.45, range_desc)   # 왼쪽 차선 탐색 종료 X비율
        self.declare_parameter('right_search_min', 0.55, range_desc)  # 오른쪽 차선 탐색 시작 X비율
        self.declare_parameter('right_search_max', 0.90, range_desc)  # 오른쪽 차선 탐색 종료 X비율

        # 초기 등록된 파라미터 값을 읽어서 멤버 변수에 저장
        self.top_x = self.get_parameter('top_x').value
        self.top_y = self.get_parameter('top_y').value
        self.bottom_x = self.get_parameter('bottom_x').value
        self.bottom_y = self.get_parameter('bottom_y').value

        self.left_search_min = self.get_parameter('left_search_min').value
        self.left_search_max = self.get_parameter('left_search_max').value
        self.right_search_min = self.get_parameter('right_search_min').value
        self.right_search_max = self.get_parameter('right_search_max').value

        # 외부에서 rqt 등으로 파라미터 값을 변경했을 때 반응할 콜백 함수 등록
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # ROS 2 Subscription / Publisher 설정
        # 카메라 이미지 메시지를 구독 (/camera/image_raw 토픽)
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        # 속도 제어 명령을 발행할 Publisher 생성 (/cmd_vel 토픽)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # ROS 이미지 <-> OpenCV 변환 객체 생성
        self.br = CvBridge()

        # BEV 출력 해상도 및 슬라이딩 윈도우 알고리즘 관련 파라미터
        self.bev_w = 640  # 조감도 변환 이미지 너비 (픽셀)
        self.bev_h = 480  # 조감도 변환 이미지 높이 (픽셀)
        
        self.nwindows = 9         # Y축 방향으로 쌓을 윈도우 개수
        self.margin = 50          # 탐색 윈도우의 좌우 반절 너비 (픽셀 단위)
        self.minpix = 30          # 윈도우 중심을 재설정하기 위해 필요한 최소 차선 픽셀 수

        # 이전 프레임에서 계산된 2차 곡선 피팅 계수(a, b, c)를 저장하는 변수
        self.left_fit = None   # 왼쪽 차선 곡선 계수 [a, b, c]
        self.right_fit = None  # 오른쪽 차선 곡선 계수 [a, b, c]
        self.lane_width = 380.0   # 기본 차선 폭 (픽셀 단위, 양쪽 다 검출될 때 자동 갱신됨)

    def parameter_callback(self, params):
        """동적으로 변경된 ROS 파라미터를 실시간 변수에 반영하는 콜백 함수"""
        for param in params:
            if param.name == 'top_x': self.top_x = param.value
            elif param.name == 'top_y': self.top_y = param.value
            elif param.name == 'bottom_x': self.bottom_x = param.value
            elif param.name == 'bottom_y': self.bottom_y = param.value
            elif param.name == 'left_search_min': self.left_search_min = param.value
            elif param.name == 'left_search_max': self.left_search_max = param.value
            elif param.name == 'right_search_min': self.right_search_min = param.value
            elif param.name == 'right_search_max': self.right_search_max = param.value
        # 파라미터 변경 성공 응답 반환
        return SetParametersResult(successful=True)

    def bird_eye_view(self, img):
        """입력 이미지를 위에서 내려다보는 탑뷰(Bird's Eye View)로 원근 변환"""
        h, w = img.shape[:2]  # 입력 이미지의 높이(h)와 너비(w) 획득
        
        # 원본 이미지에서 가져올 사다리꼴 형태의 4개 지점 좌표 지정
        src_pts = np.float32([
            [w * self.top_x, h * self.top_y],                 # 좌상단
            [w * (1.0 - self.top_x), h * self.top_y],         # 우상단
            [w * (1.0 - self.bottom_x), h * self.bottom_y],   # 우하단
            [w * self.bottom_x, h * self.bottom_y]            # 좌하단
        ])
        # 변환 후 매핑될 직사각형 형태의 4개 지점 좌표 지정 (640x480)
        dst_pts = np.float32([
            [0, 0], [self.bev_w, 0],
            [self.bev_w, self.bev_h], [0, self.bev_h]
        ])
        
        # 원근 변환 행렬 계산
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # 역원근 변환 행렬 계산 (나중에 원본 시점으로 되돌리기 위함)
        inv_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
        # 이미지 변환 적용
        warped = cv2.warpPerspective(img, matrix, (self.bev_w, self.bev_h))
        
        return warped, src_pts, inv_matrix

    def binarize_lane(self, bgr_frame):
        """노란색/흰색 필터 및 Sobel 에지 필터를 조합하여 차선 마스크(이진화) 생성"""
        # 노이즈 제거를 위한 가우시안 블러 적용
        blurred = cv2.GaussianBlur(bgr_frame, (5, 5), 0)
        # BGR 색상 공간을 HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 노란색 차선 HSV 임계값 설정 및 마스크 생성
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 흰색 차선 HSV 임계값 설정 및 마스크 생성
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 45, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 노란색 마스크와 흰색 마스크를 비트 OR 연산으로 합침
        color_binary = cv2.bitwise_or(yellow_mask, white_mask)

        # 수직 에지(차선 경계) 추출을 위한 Sobel X 필터 적용
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # 흑백 변환
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X방향 미분
        abs_sobelx = np.absolute(sobelx)  # 절댓값 취함
        
        # Sobel 에지 응답값을 0~255로 스케일링
        max_sobel = np.max(abs_sobelx)
        if max_sobel > 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel)
        else:
            scaled_sobel = np.uint8(abs_sobelx)
        
        # 에지 강도 임계값 적용 (20~100 사이 필터링)
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 255

        # 색상 마스크와 에지 마스크를 OR 연산으로 결합
        combined_binary = cv2.bitwise_or(color_binary, sobel_binary)
        
        # 형태학적 모폴로지(Closing) 연산으로 차선 내부 빈틈/노이즈 मे꿔줌
        kernel = np.ones((3, 3), np.uint8)
        cleaned_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)

        return cleaned_binary

    def recover_missing_lane(self, left_fit, right_fit):
            """[핵심] 한쪽 차선이 안 보일 때 다른 쪽 차선을 오프셋하여 복원"""
            if left_fit is not None and right_fit is not None:
                # 양쪽 다 보일 때는 실제 차선 폭을 측정하여 갱신 (지속 학습)
                self.lane_width = right_fit[2] - left_fit[2]
                return left_fit, right_fit
    
            if left_fit is not None and right_fit is None:
                # 오른쪽 차선만 안 보일 때: 왼쪽 차선을 오른쪽으로 평행이동
                right_fit = left_fit.copy()
                right_fit[2] += self.lane_width
                
            elif left_fit is None and right_fit is not None:
                # 왼쪽 차선만 안 보일 때: 오른쪽 차선을 왼쪽으로 평행이동
                left_fit = right_fit.copy()
                left_fit[2] -= self.lane_width
    
            return left_fit, right_fit

    def fit_polynomial_sliding_window(self, binary_warped):
        """슬라이딩 윈도우 알고리즘: 처음 차선을 찾거나 차선을 잃어버렸을 때 사용"""
        w = binary_warped.shape[1]
        h = binary_warped.shape[0]

        # 이미지 하단 절반(h//2:) 영역의 X축 방향 픽셀 합계를 구해 히스토그램 생성
        histogram = np.sum(binary_warped[h//2:, :], axis=0)

        # 좌/우 차선 탐색 범위를 픽셀 인덱스 단위로 전환
        l_min_idx = int(w * self.left_search_min)
        l_max_idx = int(w * self.left_search_max)
        r_min_idx = int(w * self.right_search_min)
        r_max_idx = int(w * self.right_search_max)

        # 인덱스 범위가 이미지 영역을 벗어나지 않도록 안전하게 클램핑
        l_min_idx = max(0, min(l_min_idx, w))
        l_max_idx = max(l_min_idx, min(l_max_idx, w))
        r_min_idx = max(0, min(r_min_idx, w))
        r_max_idx = max(r_min_idx, min(r_max_idx, w))

        # 설정된 탐색 영역 내에서 히스토그램 슬라이싱
        left_slice = histogram[l_min_idx:l_max_idx]
        right_slice = histogram[r_min_idx:r_max_idx]

        # 히스토그램 피크(가장 흰 픽셀이 많은 X 좌표) 위치를 슬라이딩 윈도우의 시작점으로 지정
        if len(left_slice) > 0 and np.max(left_slice) > 0:
            leftx_base = np.argmax(left_slice) + l_min_idx
        else:
            leftx_base = int((l_min_idx + l_max_idx) / 2)

        if len(right_slice) > 0 and np.max(right_slice) > 0:
            rightx_base = np.argmax(right_slice) + r_min_idx
        else:
            rightx_base = int((r_min_idx + r_max_idx) / 2)

        # 윈도우 하나의 높이 계산
        window_height = int(h // self.nwindows)
        
        # 이진화 이미지에서 0이 아닌(흰색) 픽셀들의 Y, X 좌표 추출
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # 현재 윈도우의 중심 X 좌표 초기화
        leftx_current = leftx_base
        rightx_current = rightx_base

        # 윈도우 내부에서 발견된 차선 픽셀 인덱스들을 모을 리스트
        left_lane_inds = []
        right_lane_inds = []

        # 시각화를 위한 3채널 RGB 이미지 생성
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # 아래에서부터 위로 윈도우를 올려가며 차선 탐색
        for window in range(self.nwindows):
            # 현재 윈도우의 상단/하단 Y 범위
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            # 현재 윈도우의 좌/우 차선 X 범위 (중심 +/- margin)
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # 시각화를 위해 초록색 윈도우 사각형 그리기
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # 사각형 영역 안에 들어오는 흰색 픽셀의 인덱스 추출
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # 픽셀 수가 minpix(30개)보다 많으면 다음 윈도우의 중심 X를 픽셀들의 평균 위치로 이동
            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # 리스트 합치기
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # 좌/우 차선 픽셀 좌표 최종 추출
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit, right_fit = None, None
        # 추출된 픽셀이 일정 개수(200개) 이상일 때만 2차 다항식 피팅(polyfit) 진행: x = a*y^2 + b*y + c
        if len(leftx) > 200:
            left_fit = np.polyfit(lefty, leftx, 2)
            out_img[lefty, leftx] = [255, 0, 0]  # 왼쪽 차선 픽셀: 빨간색으로 표시
        if len(rightx) > 200:
            right_fit = np.polyfit(righty, rightx, 2)
            out_img[righty, rightx] = [0, 0, 255]  # 오른쪽 차선 픽셀: 파란색으로 표시

        left_fit, right_fit = self.recover_missing_lane(left_fit, right_fit)
        # 탐색 범위를 빨간색/초록색 마스크로 표시하는 디버그 이미지 생성
        range_visual_img = self.generate_range_visual_img(out_img, l_min_idx, l_max_idx, r_min_idx, r_max_idx)

        return out_img, range_visual_img, left_fit, right_fit

    def search_around_poly(self, binary_warped):
        """Margin Search: 이전 프레임 곡선 방정식을 바탕으로 주변 영역만 빠르게 탐색"""
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.margin

        # 이전 계수(left_fit/right_fit)로 계산된 X 곡선 위치를 기준으로 margin 범위 내부 픽셀만 추출
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & 
                          (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & 
                           (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

        # 추출된 차선 픽셀 좌표
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # 픽셀 수가 충분하면(200개 초과) 신규 2차 다항식 피팅 적용, 부족하면 None 반환하여 슬라이딩 윈도우로 복귀 유도
        left_fit, right_fit = None, None
        if len(leftx) > 200:
            left_fit = np.polyfit(lefty, leftx, 2)
            out_img[lefty, leftx] = [255, 0, 0]  # 빨간색
        if len(rightx) > 200:
            right_fit = np.polyfit(righty, rightx, 2)
            out_img[righty, rightx] = [0, 0, 255]  # 파란색

        # 추적 중인 차선 곡선 중심선을 노란색 선으로 시각화
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        if left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            pts_left_line = np.dstack((left_fitx, ploty)).astype(np.int32)
            cv2.polylines(out_img, [pts_left_line], False, (0, 255, 255), 2)
        if right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            pts_right_line = np.dstack((right_fitx, ploty)).astype(np.int32)
            cv2.polylines(out_img, [pts_right_line], False, (0, 255, 255), 2)

        left_fit, right_fit = self.recover_missing_lane(left_fit, right_fit)
        
        # 디버그용 안내 텍스트 추가
        range_visual_img = out_img.copy()
        cv2.putText(range_visual_img, "MARGIN SEARCH MODE (Active)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return out_img, range_visual_img, left_fit, right_fit

    def generate_range_visual_img(self, out_img, l_min_idx, l_max_idx, r_min_idx, r_max_idx):
        """슬라이딩 윈도우 모드 시 탐색 구역(초록색)과 제외 구역(빨간색)을 오버레이 표시"""
        h, w = out_img.shape[:2]
        range_visual_img = out_img.copy()
        overlay = range_visual_img.copy()

        # 차선 탐색 제외 영역(빨간색 영역)
        cv2.rectangle(overlay, (0, h//2), (l_min_idx, h), (0, 0, 255), -1)
        cv2.rectangle(overlay, (l_max_idx, h//2), (r_min_idx, h), (0, 0, 255), -1)
        cv2.rectangle(overlay, (r_max_idx, h//2), (w, h), (0, 0, 255), -1)

        # 실제 차선을 탐색하는 허용 영역(초록색 영역)
        cv2.rectangle(overlay, (l_min_idx, h//2), (l_max_idx, h), (0, 255, 0), -1)
        cv2.rectangle(overlay, (r_min_idx, h//2), (r_max_idx, h), (0, 255, 0), -1)

        # 투명도 35%로 오버레이 합성
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, range_visual_img, 1 - alpha, 0, range_visual_img)

        # 영역 경계 세로선 그리기 (노란색)
        for x_line in [l_min_idx, l_max_idx, r_min_idx, r_max_idx]:
            cv2.line(range_visual_img, (x_line, 0), (x_line, h), (0, 255, 255), 2)

        # 모드 안내 문구 표시
        cv2.putText(range_visual_img, "SLIDING WINDOW MODE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return range_visual_img

    def draw_lane_area(self, original_img, binary_warped, left_fit, right_fit, inv_matrix):
        """추적된 차선 곡선 사이의 주행 영역을 원본 카메라 시점으로 복원하여 합성"""
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # 좌/우 차선 피팅 결과가 모두 존재하는 경우에만 주행 영역을 채움
        if left_fit is not None and right_fit is not None:
            # 2차 방정식 기반 X 좌표 복원
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            # 다각형 생성을 위한 좌표 정렬
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # 주행 영역 내부를 초록색(0, 255, 0) 다각형으로 채우기
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # BEV 시점의 초록색 영성을 역원근 변환 행렬(inv_matrix)을 사용해 원본 카메라 시점으로 역변환
        new_warp = cv2.warpPerspective(color_warp, inv_matrix, (original_img.shape[1], original_img.shape[0]))
        # 원본 영상과 40% 투명도로 합성
        result = cv2.addWeighted(original_img, 1, new_warp, 0.4, 0)
        
        return result

    def image_callback(self, msg):
        """카메라 프레임이 수신될 때마다 호출되는 메인 처리 파이프라인"""
        # 1. ROS Image 메시지를 OpenCV BGR 이미지 포맷으로 변환
        frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. BEV 탑뷰 변환
        warped_bgr, src_pts, inv_matrix = self.bird_eye_view(frame)
        
        # 3. 이진화 (차선 추출)
        binary_bev = self.binarize_lane(warped_bgr)

        # 4. 차선 추적 파이프라인 (Margin Search vs Sliding Window 제어)
        if self.left_fit is not None and self.right_fit is not None:
            # 이전 프레임 정보가 남아있으면 Margin Search 우선 수행 (연산량 절감)
            sliding_window_img, range_visual_img, current_left_fit, current_right_fit = self.search_around_poly(binary_bev)
        else:
            current_left_fit, current_right_fit = None, None

        # Margin Search에 실패했거나(None), 초기 프레임인 경우 -> 슬라이딩 윈도우로 차선 재탐색
        if current_left_fit is None or current_right_fit is None:
            sliding_window_img, range_visual_img, current_left_fit, current_right_fit = self.fit_polynomial_sliding_window(binary_bev)

        # 최신 계수로 멤버 변수 업데이트 (다음 프레임 판단 기준이 됨)
        self.left_fit = current_left_fit
        self.right_fit = current_right_fit

        # 5. 원본 화면상에 주행 영역을 시각화 오버레이
        result_lane_img = self.draw_lane_area(frame, binary_bev, self.left_fit, self.right_fit, inv_matrix)

        # 디버그용 디스플레이 출력 (실시간 3개 창)
        cv2.imshow("1. Sliding Window / Margin Search", sliding_window_img)  # 윈도우/마진 추적 창
        cv2.imshow("2. Final Lane Tracking", result_lane_img)                  # 원본 오버레이 결과 창
        cv2.imshow("3. Search Window Range Mode", range_visual_img)           # 현재 탐색 모드 표시 창
        cv2.waitKey(1)  # GUI 이벤트를 처리하기 위한 1ms 대기

def main(args=None):
    """메인 실행 함수"""
    rclpy.init(args=args)  # ROS 2 Python 클라이언트 초기화
    node = CameraSensorTrackingControl()  # 노드 객체 생성
    
    try:
        rclpy.spin(node)  # 노드가 종료될 때까지 노드 실행 (콜백 대기)
    except KeyboardInterrupt:
        pass  # Ctrl+C 종료 예외 처리
    finally:
        # 노드 종료 및 모든 OpenCV 창 닫기
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()