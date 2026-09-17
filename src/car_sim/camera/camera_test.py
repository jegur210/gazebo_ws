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
        
        # ROS 2 파라미터 선언
        range_desc = ParameterDescriptor(
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)]
        )
        self.declare_parameter('top_x', 0.15, range_desc)
        self.declare_parameter('top_y', 0.45, range_desc)
        self.declare_parameter('bottom_x', 0.00, range_desc)
        self.declare_parameter('bottom_y', 0.95, range_desc)

        self.declare_parameter('left_search_min', 0.00, range_desc)
        self.declare_parameter('left_search_max', 0.45, range_desc)
        self.declare_parameter('right_search_min', 0.55, range_desc)
        self.declare_parameter('right_search_max', 1.00, range_desc)

        self.top_x = self.get_parameter('top_x').value
        self.top_y = self.get_parameter('top_y').value
        self.bottom_x = self.get_parameter('bottom_x').value
        self.bottom_y = self.get_parameter('bottom_y').value

        self.left_search_min = self.get_parameter('left_search_min').value
        self.left_search_max = self.get_parameter('left_search_max').value
        self.right_search_min = self.get_parameter('right_search_min').value
        self.right_search_max = self.get_parameter('right_search_max').value

        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.br = CvBridge()

        # BEV 및 알고리즘 파라미터
        self.bev_w = 640
        self.bev_h = 480
        self.nwindows = 9         
        self.margin = 80          
        self.minpix = 20          

        self.left_fit = None   
        self.right_fit = None  
        self.lane_width = 380.0   

        # [최적화 1] 디버그 모드 플래그 (실제 주행 시 False로 바꾸면 처리 속도 2~3배 향상)
        self.debug_mode = True  

        # [최적화 2] 메모리 연산용 사전 할당
        self.ploty = np.linspace(0, self.bev_h - 1, self.bev_h)
        self.morph_kernel = np.ones((3, 3), np.uint8)

        self.fail_count = 0
        self.max_fail_count = 3  # 연속 3프레임 실패 시 완전 reset
        
    def parameter_callback(self, params):
        for param in params:
            if param.name == 'top_x': self.top_x = param.value
            elif param.name == 'top_y': self.top_y = param.value
            elif param.name == 'bottom_x': self.bottom_x = param.value
            elif param.name == 'bottom_y': self.bottom_y = param.value
            elif param.name == 'left_search_min': self.left_search_min = param.value
            elif param.name == 'left_search_max': self.left_search_max = param.value
            elif param.name == 'right_search_min': self.right_search_min = param.value
            elif param.name == 'right_search_max': self.right_search_max = param.value
        return SetParametersResult(successful=True)

    def bird_eye_view(self, img):
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
        inv_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
        warped = cv2.warpPerspective(img, matrix, (self.bev_w, self.bev_h))
        
        return warped, inv_matrix

    def binarize_lane(self, bgr_frame):
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        
        yellow_mask = cv2.inRange(hsv, np.array([10, 60, 60]), np.array([40, 255, 255]))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
        color_binary = cv2.bitwise_or(yellow_mask, white_mask)

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        _, gray_binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, sobel_binary = cv2.threshold(sobelx, 30, 255, cv2.THRESH_BINARY)

        combined_binary = cv2.bitwise_or(color_binary, gray_binary)
        combined_binary = cv2.bitwise_or(combined_binary, sobel_binary)
        
        return cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, self.morph_kernel)

    def sanity_check(self, left_fit, right_fit):
        """다항식 추정 결과의 물리적 타당성 검증"""
        if left_fit is None or right_fit is None:
            return False

        # 1. 곡률 유사성 검증 (0.0030 허용)
        A_diff = abs(left_fit[0] - right_fit[0])
        if A_diff > 0.0030:
            return False

        # 2. 도로 폭 연산
        y_bottom = self.bev_h - 1
        y_top = 0

        left_x_bottom = left_fit[0]*(y_bottom**2) + left_fit[1]*y_bottom + left_fit[2]
        right_x_bottom = right_fit[0]*(y_bottom**2) + right_fit[1]*y_bottom + right_fit[2]
        bottom_width = right_x_bottom - left_x_bottom

        left_x_top = left_fit[0]*(y_top**2) + left_fit[1]*y_top + left_fit[2]
        right_x_top = right_fit[0]*(y_top**2) + right_fit[1]*y_top + right_fit[2]
        top_width = right_x_top - left_x_top

        # 3. 화면 양 끝 오검출 방지 (BEV 너비 640px 기준)
        # 하단 폭이 600px 이상이면 양 끝 윈도우 오검출로 판단하여 Reject
        if bottom_width > 600.0 or bottom_width < 150.0:
            return False

        # 4. 상대적 폭 비율 검증
        min_w_bottom = self.lane_width * 0.60
        max_w_bottom = self.lane_width * 1.50
        
        min_w_top = self.lane_width * 0.40
        max_w_top = self.lane_width * 1.60

        if not (min_w_bottom < bottom_width < max_w_bottom):
            return False

        if not (min_w_top < top_width < max_w_top):
            return False
        
        #self.get_logger().info()
        return True
    
    def process_lane_fitting(self, left_fit, right_fit):
        """Fitting -> Sanity Check -> Advanced Recovery -> Fallback 파이프라인"""
        
        # [Step 1] 두 차선 모두 검출된 경우 Sanity Check 진행
        if left_fit is not None and right_fit is not None:
            if self.sanity_check(left_fit, right_fit):
                y_bottom = self.bev_h - 1
                x_left = left_fit[0]*(y_bottom**2) + left_fit[1]*y_bottom + left_fit[2]
                x_right = right_fit[0]*(y_bottom**2) + right_fit[1]*y_bottom + right_fit[2]
                self.lane_width = abs(x_right - x_left)
                return left_fit, right_fit
            else:
                # 검증 실패 시 한쪽 차선을 버리고 복원 로직으로 이관
                left_fit, right_fit = None, None

        # [Step 2] 한쪽 차선만 검출되었거나 검증 실패 시: 접선/법선 기울기 기반 곡률 복원
        if left_fit is not None and right_fit is None:
            eval_y = np.linspace(0, self.bev_h - 1, 10)
            left_x = left_fit[0]*(eval_y**2) + left_fit[1]*eval_y + left_fit[2]
            dx_dy = 2 * left_fit[0] * eval_y + left_fit[1]
            right_x = left_x + self.lane_width * np.sqrt(1 + dx_dy**2)
            right_fit = np.polyfit(eval_y, right_x, 2)

        elif left_fit is None and right_fit is not None:
            eval_y = np.linspace(0, self.bev_h - 1, 10)
            right_x = right_fit[0]*(eval_y**2) + right_fit[1]*eval_y + right_fit[2]
            dx_dy = 2 * right_fit[0] * eval_y + right_fit[1]
            left_x = right_x - self.lane_width * np.sqrt(1 + dx_dy**2)
            left_fit = np.polyfit(eval_y, left_x, 2)

        # [Step 3] 양쪽 모두 놓치거나 복원도 실패한 경우: 직전 프레임 계수 재사용 (Fallback)
        elif left_fit is None and right_fit is None:
            left_fit = self.left_fit
            right_fit = self.right_fit

        return left_fit, right_fit

    def fit_polynomial_sliding_window(self, binary_warped):
        w, h = binary_warped.shape[1], binary_warped.shape[0]
        histogram = np.sum(binary_warped[h//2:, :], axis=0)

        l_min_idx = max(0, min(int(w * self.left_search_min), w))
        l_max_idx = max(l_min_idx, min(int(w * self.left_search_max), w))
        r_min_idx = max(0, min(int(w * self.right_search_min), w))
        r_max_idx = max(r_min_idx, min(int(w * self.right_search_max), w))

        left_slice = histogram[l_min_idx:l_max_idx]
        right_slice = histogram[r_min_idx:r_max_idx]

        leftx_base = (np.argmax(left_slice) + l_min_idx) if len(left_slice) > 0 and np.max(left_slice) > 0 else int((l_min_idx + l_max_idx) / 2)
        rightx_base = (np.argmax(right_slice) + r_min_idx) if len(right_slice) > 0 and np.max(right_slice) > 0 else int((r_min_idx + r_max_idx) / 2)

        window_height = int(h // self.nwindows)
        nonzeroy, nonzerox = binary_warped.nonzero()

        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        out_img = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR) if self.debug_mode else None

        for window in range(self.nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low, win_xleft_high = leftx_current - self.margin, leftx_current + self.margin
            win_xright_low, win_xright_high = rightx_current - self.margin, rightx_current + self.margin

            if self.debug_mode:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > self.minpix: leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > self.minpix: rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([], dtype=int)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 100 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 100 else None

        if self.debug_mode:
            if left_fit is not None: out_img[lefty, leftx] = [0, 0, 255]
            if right_fit is not None: out_img[righty, rightx] = [255, 0, 0]
            range_visual_img = self.generate_range_visual_img(out_img, l_min_idx, l_max_idx, r_min_idx, r_max_idx)
        else:
            range_visual_img = None

        left_fit, right_fit = self.process_lane_fitting(left_fit, right_fit)
        return out_img, range_visual_img, left_fit, right_fit

    def search_around_poly(self, binary_warped):
        nonzeroy, nonzerox = binary_warped.nonzero()
        margin = self.margin

        left_fitx = self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2]
        right_fitx = self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2]

        left_lane_inds = (nonzerox > left_fitx - margin) & (nonzerox < left_fitx + margin)
        right_lane_inds = (nonzerox > right_fitx - margin) & (nonzerox < right_fitx + margin)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        out_img = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR) if self.debug_mode else None

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 100 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 100 else None

        if self.debug_mode:
            if left_fit is not None:
                out_img[lefty, leftx] = [0, 0, 255]
                fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
                pts = np.dstack((fitx, self.ploty)).astype(np.int32)
                cv2.polylines(out_img, [pts], False, (0, 255, 255), 2)
                
            if right_fit is not None:
                out_img[righty, rightx] = [255, 0, 0]
                fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
                pts = np.dstack((fitx, self.ploty)).astype(np.int32)
                cv2.polylines(out_img, [pts], False, (0, 255, 255), 2)

            range_visual_img = out_img.copy()
            cv2.putText(range_visual_img, "MARGIN SEARCH MODE (Active)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            range_visual_img = None

        left_fit, right_fit = self.process_lane_fitting(left_fit, right_fit)
        return out_img, range_visual_img, left_fit, right_fit

    def generate_range_visual_img(self, out_img, l_min_idx, l_max_idx, r_min_idx, r_max_idx):
        h, w = out_img.shape[:2]
        range_visual_img = out_img.copy()
        overlay = range_visual_img.copy()

        cv2.rectangle(overlay, (0, h//2), (l_min_idx, h), (0, 0, 255), -1)
        cv2.rectangle(overlay, (l_max_idx, h//2), (r_min_idx, h), (0, 0, 255), -1)
        cv2.rectangle(overlay, (r_max_idx, h//2), (w, h), (0, 0, 255), -1)

        cv2.rectangle(overlay, (l_min_idx, h//2), (l_max_idx, h), (0, 255, 0), -1)
        cv2.rectangle(overlay, (r_min_idx, h//2), (r_max_idx, h), (0, 255, 0), -1)

        cv2.addWeighted(overlay, 0.35, range_visual_img, 0.65, 0, range_visual_img)

        for x_line in [l_min_idx, l_max_idx, r_min_idx, r_max_idx]:
            cv2.line(range_visual_img, (x_line, 0), (x_line, h), (0, 255, 255), 2)

        cv2.putText(range_visual_img, "SLIDING WINDOW MODE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return range_visual_img

    def draw_lane_area(self, original_img, binary_warped, left_fit, right_fit, inv_matrix):
        if left_fit is None or right_fit is None:
            return original_img

        color_warp = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8)

        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        new_warp = cv2.warpPerspective(color_warp, inv_matrix, (original_img.shape[1], original_img.shape[0]))
        return cv2.addWeighted(original_img, 1, new_warp, 0.4, 0)

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        warped_bgr, inv_matrix = self.bird_eye_view(frame)
        binary_bev = self.binarize_lane(warped_bgr)

        current_left_fit, current_right_fit = None, None
        is_margin_search_success = False  #Margin Search 성공 여부 플래그

        # 1. 기존 Fit이 존재하면 Margin Search 시도
        if self.left_fit is not None and self.right_fit is not None:
            sliding_window_img, range_visual_img, current_left_fit, current_right_fit = self.search_around_poly(binary_bev)

            if current_left_fit is not None and current_right_fit is not None:
                is_margin_search_success = True

        # 2. Margin Search 실패 시 Sliding Window로 Fallback
        if current_left_fit is None or current_right_fit is None:
            sliding_window_img, range_visual_img, current_left_fit, current_right_fit = self.fit_polynomial_sliding_window(binary_bev)

    
        # 3. 연속 실패 감지 및 Fit Reset 로직 (핵심 수정 부분)
        if current_left_fit is None or current_right_fit is None:
            self.fail_count += 1
            if self.fail_count >= self.max_fail_count:
                # 커브 탈출 후 어긋남 방지를 위해 이전 Fit 기록 완전 삭제
                self.left_fit = None
                self.right_fit = None
                self.fail_count = 0
        else:
            self.fail_count = 0
            self.left_fit = current_left_fit
            self.right_fit = current_right_fit

        result_lane_img = self.draw_lane_area(frame, binary_bev, self.left_fit, self.right_fit, inv_matrix)

        if self.debug_mode and sliding_window_img is not None:
            if is_margin_search_success:
                status_text = "MODE: MARGIN SEARCH (SUCCESS)"
                text_color = (0, 255, 0)  # 성공: 초록색
            else:
                status_text = "MODE: SLIDING WINDOW (FALLBACK)"
                text_color = (0, 0, 255)  # 실패 및 전환: 빨간색

            cv2.putText(sliding_window_img, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
        # [최적화 5] 디버그 모드 상태에서만 cv2.imshow 연산 수행
        # 디버그 창 시각화
        if self.debug_mode:
            cv2.imshow("0. Binary BEV", binary_bev)  # 이진화 영상 확인용 창
            if sliding_window_img is not None:
                cv2.imshow("1. Sliding Window / Margin Search", sliding_window_img)
            cv2.imshow("2. Final Lane Tracking", result_lane_img)
            if range_visual_img is not None:
                cv2.imshow("3. Search Window Range Mode", range_visual_img)
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