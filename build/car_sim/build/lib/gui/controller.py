import threading
import tkinter as tk
import os
import yaml

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter as RclpyParameter
from geometry_msgs.msg import Twist
from rcl_interfaces.srv import SetParameters

class CarControlNode(Node):
    def __init__(self):
        super().__init__('car_control_ui_node')
        
        # 1. 속도 퍼블리셔
        self.cmd_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 2. 파라미터 변경을 위한 서비스 클라이언트
        self.param_client = self.create_client(SetParameters, '/camera_controller_test/set_parameters')
        
    def send_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_publisher_.publish(msg)

    def update_parameters_batch(self, param_dict):
        if not param_dict or not self.param_client.wait_for_service(timeout_sec=0.1):
            return False
        
        request = SetParameters.Request()
        request.parameters = [
            RclpyParameter(name, RclpyParameter.Type.DOUBLE, float(val)).to_parameter_msg()
            for name, val in param_dict.items()
        ]
        
        self.param_client.call_async(request)
        return True


class CarControlApp(tk.Tk):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node

        self.title("Car & Vision Parameter Control Panel")
        self.geometry("520x920")
        self.resizable(False, False)

        # 타이틀
        self.title_label = tk.Label(self, text="🚗 차량 및 비전 파라미터 통합 제어기", font=("Arial", 14, "bold"))
        self.title_label.pack(pady=10)

        # --- [1] 주행 속도 제어 프레임 ---
        self.speed_frame = tk.LabelFrame(self, text=" 주행 속도 제어 ", font=("Arial", 10, "bold"))
        self.speed_frame.pack(pady=5, padx=15, fill="x")

        #self.speed_label = tk.Label(self.speed_frame, text="Linear X: 0.0 m/s")
        #self.speed_label.pack()
        #self.speed_slider = tk.Scale(self.speed_frame, from_=-2.0, to=2.0, resolution=0.1, orient="horizontal", command=self.on_speed_change)
        #self.speed_slider.set(0.0)
        #self.speed_slider.pack(fill="x", padx=10, pady=2)

        self.turn_label = tk.Label(self.speed_frame, text="Angular Z: 0.0 rad/s")
        self.turn_label.pack()
        self.turn_slider = tk.Scale(self.speed_frame, from_=-3.0, to=3.0, resolution=0.1, orient="horizontal", command=self.on_turn_change)
        self.turn_slider.set(0.0)
        self.turn_slider.pack(fill="x", padx=10, pady=2)

        # --- [2] 비전 파라미터 제어 프레임 ---
        self.param_frame = tk.LabelFrame(self, text=" 비전 알고리즘 파라미터 제어 (ROI, Search Range, PID) ", font=("Arial", 10, "bold"))
        self.param_frame.pack(pady=10, padx=15, fill="x")

        self.sliders = {}

        self.speed_label = tk.Label(self.param_frame, text="Linear X: 0.0 m/s")
        self.speed_label.pack()
        self.create_param_slider("target_speed", 0.0, 2.0, 0.1, 0.0)


        # ROI 설정
        self.create_param_slider("top_x", 0.0, 1.0, 0.01, 0.15)
        self.create_param_slider("top_y", 0.0, 1.0, 0.01, 0.45)
        self.create_param_slider("bottom_x", 0.0, 1.0, 0.01, 0.00)
        self.create_param_slider("bottom_y", 0.0, 1.0, 0.01, 0.95)

        # 탐색 영역 설정
        self.create_param_slider("left_search_min", 0.0, 1.0, 0.01, 0.00)
        self.create_param_slider("left_search_max", 0.0, 1.0, 0.01, 0.45)
        self.create_param_slider("right_search_min", 0.0, 1.0, 0.01, 0.55)
        self.create_param_slider("right_search_max", 0.0, 1.0, 0.01, 1.00)

        # PID 제어 이득
        self.create_param_slider("kp", 0.0, 0.3, 0.01, 0.03)
        self.create_param_slider("kd", 0.0, 0.002, 0.0001, 0.0005)

        # --- 버튼 배치 영역 (통신 적용 버튼 & YAML 저장 버튼) ---
        self.btn_action_frame = tk.Frame(self.param_frame)
        self.btn_action_frame.pack(pady=10, fill="x", padx=10)

        self.apply_btn = tk.Button(self.btn_action_frame, text="🔄 통신으로 실시간 적용", fg="white", bg="blue", font=("Arial", 9, "bold"), command=self.apply_parameters_via_ros)
        self.apply_btn.pack(side="left", expand=True, fill="x", padx=2)

        self.save_yaml_btn = tk.Button(self.btn_action_frame, text="💾 YAML 파일로 저장", fg="white", bg="green", font=("Arial", 9, "bold"), command=self.save_parameters_to_yaml)
        self.save_yaml_btn.pack(side="right", expand=True, fill="x", padx=2)

        # 상태 표시 라벨
        self.status_label = tk.Label(self.param_frame, text="상태: 대기 중", fg="gray", font=("Arial", 9))
        self.status_label.pack(pady=2)

        # --- [3] 하단 제어 버튼 ---
        self.btn_frame = tk.Frame(self)
        self.btn_frame.pack(pady=2)

        self.stop_btn = tk.Button(self.btn_frame, text="🛑 긴급 정지", fg="white", bg="red", font=("Arial", 11, "bold"), command=self.stop_robot)
        self.stop_btn.pack(side="left", padx=10, ipadx=15, ipady=3)

        self.current_linear = 0.0
        self.current_angular = 0.0
        self.after(50, self.publish_command)

    def create_param_slider(self, param_name, min_val, max_val, res, default_val):
        frame = tk.Frame(self.param_frame)
        frame.pack(fill="x", padx=5, pady=1)

        lbl = tk.Label(frame, text=f"{param_name}: {default_val}", width=18, anchor="w", font=("Arial", 9))
        lbl.pack(side="left")

        slider = tk.Scale(
            frame, from_=min_val, to=max_val, resolution=res, orient="horizontal",
            command=lambda val, name=param_name, label=lbl: self.on_slider_move(name, val, label)
        )
        slider.set(default_val)
        slider.pack(side="right", fill="x", expand=True, padx=5)

        self.sliders[param_name] = {"slider": slider, "label": lbl}

    def on_slider_move(self, param_name, value, label_widget):
        val_float = float(value)
        label_widget.configure(text=f"{param_name}: {val_float:.4f}")

    def get_current_slider_values(self):
        param_dict = {}
        for name, item in self.sliders.items():
            param_dict[name] = item["slider"].get()
        return param_dict

    def apply_parameters_via_ros(self):
        # 1. 통신을 통한 실시간 적용 버튼
        param_dict = self.get_current_slider_values()
        success = self.ros_node.update_parameters_batch(param_dict)
        if success:
            self.status_label.configure(text="상태: [통신] 실시간 적용 완료!", fg="blue")
            print(">>> ROS 통신을 통해 파라미터가 실시간 적용되었습니다.")
        else:
            self.status_label.configure(text="상태: 노드 연결 실패 (camera_test 확인)", fg="red")
            print(">>> 통신 실패: camera_test 노드가 실행 중인지 확인하세요.")

    def save_parameters_to_yaml(self):
        # 2. YAML 파일에 영구 저장하는 버튼
        param_dict = self.get_current_slider_values()
        
        # 본인의 파라미터 YAML 파일 경로로 수정하세요
        yaml_path = os.path.expanduser("~/gazebo_ws/src/car_sim/param/new_params.yaml")

        try:
            # 파일이 존재하지 않는 경우를 대비한 기본 구조 방어 코드
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}

            # ROS 2 파라미터 파일 포맷 구조 맞추기
            if 'camera_test' not in data:
                data['camera_test'] = {}
            if 'ros__parameters' not in data['camera_test']:
                data['camera_test']['ros__parameters'] = {}

            # 값 갱신
            for name, val in param_dict.items():
                data['camera_test']['ros__parameters'][name] = val

            # 파일에 저장
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            self.status_label.configure(text="상태: [YAML] 파일 저장 완료!", fg="green")
            print(f">>> 성공적으로 YAML 파일에 저장되었습니다: {yaml_path}")

        except Exception as e:
            self.status_label.configure(text="상태: YAML 저장 실패", fg="red")
            print(f">>> YAML 저장 중 에러 발생: {e}")

    def on_speed_change(self, value):
        self.current_linear = float(value)
        self.speed_label.configure(text=f"Linear X: {self.current_linear:.2f} m/s")

    def on_turn_change(self, value):
        self.current_angular = float(value)
        self.turn_label.configure(text=f"Angular Z: {self.current_angular:.2f} rad/s")

    def stop_robot(self):
        self.speed_slider.set(0.0)
        self.turn_slider.set(0.0)
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.ros_node.send_cmd(0.0, 0.0)

    def publish_command(self):
        self.ros_node.send_cmd(self.current_linear, self.current_angular)
        self.after(50, self.publish_command)


def main():
    rclpy.init()
    ros_node = CarControlNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    app = CarControlApp(ros_node)
    app.mainloop()

    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()