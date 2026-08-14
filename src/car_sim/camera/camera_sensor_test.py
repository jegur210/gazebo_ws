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
        
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.br = CvBridge()
        
        # 제어 파라미터 튜닝
        self.prev_error = 0.0
        self.prev_angular_z = 0.0    # LPF 필터용 이전 조향값
        self.LANE_WIDTH_PX = 220     # 근거리 기준 기본 차선 픽셀 폭
        
        # 스탠리 / PD 게인 (진동 억제를 위해 게인 하향 및 D항 강화)
        self.Kp_y = 0.003            # 횡오차 비례 게인
        self.Kp_theta = 0.004        # 헤딩(기울기) 게인
        self.Kd = 0.004              # D(미분) 게인 - 진동 억제
        
        # 속도 설정 (m/s)
        self.MAX_SPEED = 0.22
        self.MIN_SPEED = 0.08

    def get_lane_center_at_y(self, mask_yellow, mask_white, scan_y, w):
        """특정 Y 높이(Scan Line)에서 차선 중앙 X좌표 추출"""
        y_min = max(0, scan_y - 8)
        y_max = min(mask_yellow.shape[0], scan_y + 8)
        
        sub_yellow = np.zeros_like(mask_yellow)
        sub_white = np.zeros_like(mask_white)
        
        sub_yellow[y_min:y_max, :] = mask_yellow[y_min:y_max, :]
        sub_white[y_min:y_max, :] = mask_white[y_min:y_max, :]

        # 노란선 추출
        contours_y, _ = cv2.findContours(sub_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx_yellow = None
        for cnt in contours_y:
            if cv2.contourArea(cnt) > 20:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx_yellow = int(M['m10'] / M['m00'])
                    break

        # 흰선 추출
        contours_w, _ = cv2.findContours(sub_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_cxs = []
        for cnt in contours_w:
            if cv2.contourArea(cnt) > 20:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    white_cxs.append(int(M['m10'] / M['m00']))

        if cx_yellow is not None:
            valid_rights = [x for x in white_cxs if x > cx_yellow]
            if valid_rights:
                cx_right = min(valid_rights)
                return int((cx_yellow + cx_right) / 2)
            else:
                return cx_yellow + int(self.LANE_WIDTH_PX / 2)
        elif len(white_cxs) >= 2:
            white_cxs.sort()
            return int((white_cxs[0] + white_cxs[1]) / 2)
        elif len(white_cxs) == 1:
            if white_cxs[0] > w / 2:
                return white_cxs[0] - int(self.LANE_WIDTH_PX / 2)
            else:
                return white_cxs[0] + int(self.LANE_WIDTH_PX / 2)
        
        return None

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. 색상 마스크 생성
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([179, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # 2. 근거리 / 원거리 스캔 라인
        y_near = int(h * 0.85)
        y_far = int(h * 0.68)
        screen_center_x = int(w / 2)

        center_near = self.get_lane_center_at_y(mask_yellow, mask_white, y_near, w)
        center_far = self.get_lane_center_at_y(mask_yellow, mask_white, y_far, w)

        twist = Twist()

        if center_near is not None:
            # 횡오차 ($e_y$)
            e_y = screen_center_x - center_near
            
            # 헤딩 오차 ($e_\theta$): 원근법 왜곡 보정을 위해 dx에 감쇄 계수(0.4) 곱함
            if center_far is not None:
                dx = (center_far - center_near) * 0.4
                dy = y_near - y_far
                heading_error = dx / dy  # Radian 수치 근사
            else:
                heading_error = 0.0

            # D항 연산
            d_error = e_y - self.prev_error
            self.prev_error = e_y

            # 목표 조향각 계산
            raw_angular_z = float(self.Kp_y * e_y + self.Kp_theta * heading_error + self.Kd * d_error)

            # 저주파 필터 (Low-Pass Filter)로 급격한 조향 흔들림 방지
            filtered_angular_z = 0.6 * self.prev_angular_z + 0.4 * raw_angular_z
            self.prev_angular_z = filtered_angular_z

            # 곡률 기반 선속도 제어
            abs_error = abs(e_y)
            linear_x = self.MAX_SPEED - (abs_error / screen_center_x) * (self.MAX_SPEED - self.MIN_SPEED)
            linear_x = max(self.MIN_SPEED, min(self.MAX_SPEED, linear_x))

            twist.linear.x = linear_x
            twist.angular.z = filtered_angular_z

            # 시각화
            cv2.circle(frame, (screen_center_x, y_near), 5, (0, 0, 255), -1)
            cv2.circle(frame, (center_near, y_near), 6, (0, 255, 0), -1)
            if center_far is not None:
                cv2.circle(frame, (center_far, y_far), 6, (255, 255, 0), -1)
                cv2.line(frame, (center_near, y_near), (center_far, y_far), (0, 255, 255), 2)
            cv2.line(frame, (screen_center_x, y_near), (center_near, y_near), (255, 0, 0), 2)

        else:
            # 완벽 이탈 시 완만한 회전 유지
            twist.linear.x = self.MIN_SPEED
            twist.angular.z = self.prev_angular_z * 0.8

        self.publisher_.publish(twist)

        mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
        cv2.imshow("Multi-ROI Lane Tracking", frame)
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