import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading
import pygame
import serial
from collections import deque
import math
import logging

# ==================== 配置参数 ====================
ERROR_FACTOR = 6  # 误差参数，范围1到10，数值越大容忍度越高
SEARCH_COLOR = 'red'  # 搜索球的颜色（'red', 'blue', 'black', 'yellow'）
COLOR_PRIORITY = [SEARCH_COLOR, 'yellow', 'black']  # 颜色优先级：红色 > 黄色 > 黑色

# 任务一动作序列时间参数（单位：秒）
TASK1_STOP_BEFORE_FORWARD = 0.5    # 抓取前停止等待时间
TASK1_FORWARD_TO_GRAB = 0.8        # 抓取前前进时间
TASK1_STOP_AFTER_GRAB = 0.5        # 抓取后停止等待时间

# 任务二动作序列时间参数（单位：秒）
TASK2_STOP_BEFORE_FORWARD = 0.5    # 放置前停止等待时间
TASK2_FORWARD_TO_PLACE = 0.9       # 放置前前进时间
TASK2_STOP_AFTER_PLACE = 0.5       # 放置后停止等待时间

# 任务二后动作序列时间参数（单位：秒）
STOP_AFTER_TASK2 = 1.0       # 停止1.0秒
REVERSE_AFTER_TASK2 = 1.5    # 后退1.5秒
STOP_AFTER_REVERSE = 0.5     # 停止0.5秒
TURN_RIGHT_AFTER_REVERSE = 1.5 # 右转1.5秒
STOP_AFTER_TURN = 0.5        # 停止0.5秒

# HSV 颜色范围
LOWER_RED1 = np.array([0, 0, 100])
UPPER_RED1 = np.array([20, 249, 151])
LOWER_RED2 = np.array([160, 70, 50])
UPPER_RED2 = np.array([179, 255, 255])
LOWER_BLUE = np.array([57, 94, 60])
UPPER_BLUE = np.array([103, 236, 255])
LOWER_PURPLE = np.array([121, 85, 84])
UPPER_PURPLE = np.array([153, 255, 255])

# 任务2相关参数
SELECTED_FIELD_COLOR = 'red'  # 用户选择的场地颜色（'red', 'blue'）
FIELD_DETECTION_CONFIDENCE = 0.75  # 场地检测置信度阈值
FIELD_APPROACH_DISTANCE = 0.50      # 接近场地的目标距离（米）

# 配置参数部分添加或修改以下参数
FIELD_SEARCH_ROTATION_TIME = 0.3  # 每次旋转的时间
FIELD_SEARCH_PAUSE_TIME = 0.2    # 旋转后的暂停时间
FIELD_MIN_AREA = 1000             # 最小场地面积
FIELD_ASPECT_RATIO_MIN = 1.5      # 最小长宽比

APPROACH_THRESHOLD = 0.25  # 接近阈值，米
EXIT_THRESHOLD = 0.3        # 退出阈值，米

# 新增的膨胀与腐蚀参数
FIELD_DILATION_ITERATIONS = 1     # 场地的膨胀次数
FIELD_EROSION_ITERATIONS = 0      # 场地的腐蚀次数
BALL_DILATION_ITERATIONS = 1      # 球的膨胀次数
BALL_EROSION_ITERATIONS = 1       # 球的腐蚀次数

# 曝光调整参数
EXPOSURE_COMPENSATION = 1.1       # 曝光补偿系数，默认1.0，不调整

# ==================== ROS2 图像订阅器模块 ====================

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')  # 初始化 ROS2 节点名称为'image_subscriber'
        # 订阅深度图像主题
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10)
        # 订阅彩色图像主题
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # 根据实际设置调整主题名称
            self.color_callback,
            10)
        self.bridge = CvBridge()  # 初始化 CvBridge，用于 ROS 图像消息与 OpenCV 图像之间的转换
        self.depth_image = None  # 存储深度图像
        self.color_image = None  # 存储彩色图像
        self.lock = threading.Lock()  # 线程锁，确保多线程环境下的数据一致性

    def depth_callback(self, msg):
        """处理深度图像回调函数，将 ROS 图像消息转换为 OpenCV 格式"""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.depth_image = depth_image
        except Exception as e:
            self.get_logger().error(f"深度图像转换失败: {e}")

    def color_callback(self, msg):
        """处理彩色图像回调函数，将 ROS 图像消息转换为 OpenCV 格式，并调整曝光"""
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # 调整曝光
            color_image = cv2.convertScaleAbs(color_image, alpha=EXPOSURE_COMPENSATION, beta=0)
            with self.lock:
                self.color_image = color_image
        except Exception as e:
            self.get_logger().error(f"彩色图像转换失败: {e}")

    def get_images(self):
        """获取当前的彩色图像和深度图像"""
        with self.lock:
            color = self.color_image.copy() if self.color_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None
        return color, depth

# ==================== 摇杆控制模块 ====================

class JoystickController:
    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=115200, timeout=1):
        """初始化摇杆控制器"""
        pygame.init()
        pygame.joystick.init()
        pygame.display.set_mode((1, 1))  # 初始化视频系统，防止 Pygame 错误

        self.ser = None
        try:
            self.ser = serial.Serial(
                port=serial_port,
                baudrate=baudrate,
                timeout=timeout
            )

            if self.ser.isOpen():
                print("串口成功打开。")

        except serial.SerialException as e:
            print(f"无法打开串口: {e}")

        if pygame.joystick.get_count() == 0:
            print("未检测到摇杆。")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"检测到摇杆: {self.joystick.get_name()}")

        self.last_command = ""
        self.button_states = [False] * 4
        self.deadzone = 0.2
        self.lb_pressed = False
        self.rb_pressed = False
        self.auto_mode = True
        self.running = True  # 控制线程运行

        self.lock = threading.Lock()  # 保护 auto_mode 的访问

    def send_command(self, command):
        """发送指令到串口"""
        if self.ser and self.ser.isOpen():
            try:
                self.ser.write(f"{command}\r\n".encode())
                self.ser.flush()  # 确保数据被发送
                print(f"发送指令: {command}")
            except serial.SerialException as e:
                print(f"串口通信错误: {e}")

    def process_inputs(self):
        """处理摇杆输入并发送相应的串口指令"""
        try:
            while self.running:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.JOYBUTTONDOWN:
                        if event.button == 4:  # 左肩部按钮
                            print("左肩部按钮按下，切换到手动模式。")
                            with self.lock:
                                self.auto_mode = False
                            self.send_command("5")
                        elif event.button == 5:  # 右肩部按钮
                            print("右肩部按钮按下，切换回自动模式。")
                            with self.lock:
                                self.auto_mode = True
                            self.send_command("5")
                        else:
                            # 处理其他按钮
                            buttons = ["A", "B", "X", "Y"]
                            button_indices = [0, 1, 2, 3]
                            if event.button in button_indices:
                                button_name = buttons[button_indices.index(event.button)]
                                self.send_command(f"{button_name}")
                    elif event.type == pygame.JOYAXISMOTION:
                        if not self.get_auto_mode():
                            left_stick_x = self.joystick.get_axis(0)
                            left_stick_y = self.joystick.get_axis(1)

                            command = ""
                            deadzone = self.deadzone

                            if abs(left_stick_x) < deadzone and abs(left_stick_y) < deadzone:
                                if self.last_command != "5":
                                    command = "5"  # 停止
                            else:
                                if left_stick_y < -deadzone:
                                    command = "1"  # 前进
                                elif left_stick_y > deadzone:
                                    command = "4"  # 后退
                                if left_stick_x < -deadzone:
                                    command = "6"  # 左转
                                elif left_stick_x > deadzone:
                                    command = "7"  # 右转

                            if command and command != self.last_command:
                                self.send_command(command)
                                self.last_command = command
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n退出摇杆控制器。")

        finally:
            if self.ser:
                self.ser.close()
                print("串口已关闭。")
            pygame.quit()

    def get_auto_mode(self):
        """获取当前是否为自动模式"""
        with self.lock:
            return self.auto_mode

    def stop(self):
        """停止输入处理循环"""
        self.running = False

# ==================== YOLO 处理器类 ====================

class YOLOProcessor:
    def __init__(self, model_path, image_subscriber, joystick_controller, confidence_threshold=0.7,
                 target_distance=0.17, search_color=SEARCH_COLOR, error_factor=ERROR_FACTOR):
        """
        初始化 YOLO 模型及相关参数。

        :param model_path: YOLO 模型文件路径
        :param image_subscriber: 图像订阅器实例
        :param joystick_controller: 摇杆控制器实例
        :param confidence_threshold: 置信度阈值
        :param target_distance: 目标距离（米）
        :param search_color: 搜索的球的颜色（'red', 'blue', 'black', 'yellow'）
        :param error_factor: 误差参数，范围1到10
        """
        self.logger = logging.getLogger('YOLOProcessor')
        self.model = None
        self.model_loaded = threading.Event()  # 线程事件，用于标识模型是否加载完成
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 启动模型加载线程
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()

        # 使用提供的图像订阅器和摇杆控制器
        self.image_subscriber = image_subscriber
        self.joystick_controller = joystick_controller
        self.target_distance = target_distance
        self.current_target = None
        self.target_locked = False

        # 初始化 FPS 计算
        self.prev_time = time.time()
        self.fps_deque = []

        # 相机参数
        self.fx = 476.2635192871094
        self.fy = 476.2635192871094
        self.cx = 315.7132873535156
        self.cy = 199.8099365234375

        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                     [0, self.fy, self.cy],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # 已知目标物体尺寸（米）
        self.known_width = {'ball': 0.030}
        self.known_height = {'ball': 0.040}

        self.current_command = None
        self.task_state = 'SEARCH_BALL'  # 初始任务状态
        self.color_priority = COLOR_PRIORITY
        self.search_color = search_color.lower()
        self.error_factor = error_factor

        self.latest_field_distance = None  # 最新检测到的场地距离
        self.latest_field_center = None    # 最新检测到的场地中心点

        # 平滑距离测量
        self.distance_window_size = 5  # 滑动窗口大小
        self.distance_deque = deque(maxlen=self.distance_window_size)

        # 状态标记
        self.in_grab_action = False  # 标记是否在执行抓取动作

    def load_model(self):
        """加载 YOLO 模型并设置加载完成事件"""
        self.logger.info("开始加载模型...")
        try:
            if not self.model_path.endswith('.pt'):
                raise ValueError(f"模型路径错误: '{self.model_path}' 不是一个 .pt 文件")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.logger.info("模型加载成功")
            self.class_labels = self.model.names
            self.logger.info(f"类别名称: {self.class_labels}")
            self.model_loaded.set()
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            self.model = None
            self.model_loaded.set()

    def undistort_points(self, x, y):
        """校正单个点的畸变"""
        points = np.array([[[x, y]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        return undistorted[0][0]

    def get_depth_at_point(self, depth_image, x, y, window_size=5):
        """获取指定点周围的深度值"""
        half_size = window_size // 2
        window = depth_image[max(y - half_size, 0):min(y + half_size + 1, depth_image.shape[0]),
                 max(x - half_size, 0):min(x + half_size + 1, depth_image.shape[1])]
        valid_depth = window[(window > 0) & (window < 5000)]
        if valid_depth.size == 0:
            return None
        depth = np.median(valid_depth) / 1000.0
        return depth

    def estimate_distance_width(self, bbox_width_pixels, class_label):
        """使用已知物体宽度和相机参数估计距离"""
        if class_label in self.known_width:
            real_width = self.known_width[class_label]
            if bbox_width_pixels > 0:
                distance = (real_width * self.fx) / bbox_width_pixels
                return distance
        return None

    def estimate_distance_height(self, bbox_height_pixels, class_label):
        """使用已知物体高度和相机参数估计距离"""
        if class_label in self.known_height:
            real_height = self.known_height[class_label]
            if bbox_height_pixels > 0:
                distance = (real_height * self.fy) / bbox_height_pixels
                return distance
        return None

    def detect_field_regions(self, frame, depth_image):
        """
        检测帧中的指定颜色场地。

        :param frame: 彩色图像帧
        :param depth_image: 深度图像
        :return: 是否检测到场地 (True/False)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 使用用户提供的 HSV 参数
        LOWER_RED1 = np.array([0, 0, 100])
        UPPER_RED1 = np.array([20, 249, 151])
        LOWER_RED2 = np.array([160, 70, 50])
        UPPER_RED2 = np.array([179, 255, 255])
        LOWER_BLUE = np.array([57, 94, 60])
        UPPER_BLUE = np.array([103, 236, 255])
        LOWER_PURPLE = np.array([121, 85, 84])
        UPPER_PURPLE = np.array([153, 255, 255])

        mask_purple = cv2.inRange(hsv, LOWER_PURPLE, UPPER_PURPLE)

        # 使用膨胀与腐蚀进行形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)

        # 查找紫色条纹
        purple_strips = self.find_purple_strips(mask_purple)
        if not purple_strips:
            self.logger.warning("未检测到任何紫色条纹，无法检测场地。")
            self.latest_field_distance = None
            self.latest_field_center = None  # 重置场地中心
            return False

        # 根据选择的场地颜色检测相应的场地
        if SELECTED_FIELD_COLOR == 'red':
            mask_field1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
            mask_field2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
            mask_field = cv2.bitwise_or(mask_field1, mask_field2)
            field_color = (0, 0, 255)
            field_name = "Red Field"
        elif SELECTED_FIELD_COLOR == 'blue':
            mask_field = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
            field_color = (255, 0, 0)
            field_name = "Blue Field"
        else:
            self.logger.error(f"未知的场地颜色: {SELECTED_FIELD_COLOR}")
            return False

        # 对场地掩码进行形态学处理
        mask_field = cv2.erode(mask_field, kernel, iterations=FIELD_EROSION_ITERATIONS)
        mask_field = cv2.dilate(mask_field, kernel, iterations=FIELD_DILATION_ITERATIONS)

        # 处理场地
        field_found = self.process_field(mask_field, field_name, field_color, purple_strips, mask_purple, frame, depth_image)

        if field_found:
            self.logger.info(f"检测到{field_name}。")
            return True
        else:
            self.logger.warning(f"未检测到{field_name}。")
            self.latest_field_distance = None
            self.latest_field_center = None  # 重置场地中心
            return False

    def find_purple_strips(self, mask_purple):
        """查找紫色条纹"""
        purple_contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        purple_strips = []

        for contour in purple_contours:
            rect = cv2.minAreaRect(contour)
            width = min(rect[1])
            height = max(rect[1])
            if width == 0:
                continue

            aspect_ratio = height / width
            if aspect_ratio > 2.5:
                purple_strips.append(contour)

        self.logger.info(f"找到 {len(purple_strips)} 个竖直紫色条纹。")
        return purple_strips

    def check_connection_with_purple(self, contour, purple_strips, mask_purple):
        """检查颜色区域是否与紫色条纹相连"""
        mask = np.zeros_like(mask_purple)
        cv2.drawContours(mask, [contour], -1, 255, 2)

        dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

        for purple_strip in purple_strips:
            purple_mask = np.zeros_like(mask_purple)
            cv2.drawContours(purple_mask, [purple_strip], -1, 255, 2)

            overlap = cv2.bitwise_and(dilated_mask, purple_mask)
            if cv2.countNonZero(overlap) > 0:
                return True
        return False

    def detect_field_region_conditions(self, contour, purple_strips, mask_purple):
        """检查场地区域是否满足所有条件"""
        # 放宽与紫色条纹连接的要求
        if not self.check_connection_with_purple(contour, purple_strips, mask_purple):
            self.logger.debug("颜色区域未与任何紫色条纹相连。")
            return False

        area = cv2.contourArea(contour)
        if area < FIELD_MIN_AREA:  # 降低面积阈值
            self.logger.debug(f"区域面积太小，面积={area}")
            return False

        rect = cv2.minAreaRect(contour)
        width = min(rect[1])
        height = max(rect[1])
        if width == 0:
            self.logger.debug("区域宽度为0，忽略此区域。")
            return False

        aspect_ratio = height / width
        if aspect_ratio < FIELD_ASPECT_RATIO_MIN:  # 降低长宽比要求
            self.logger.debug(f"区域长宽比太低，长宽比={aspect_ratio}")
            return False

        return True

    def process_field(self, mask, field_name, field_color, purple_strips, mask_purple, frame, depth_image):
        """处理场地区域"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            # 检查场地区域是否满足所有条件
            if not self.detect_field_region_conditions(contour, purple_strips, mask_purple):
                continue

            # 计算场地中最近的像素点的距离
            distance = self.get_closest_field_distance(contour, depth_image)

            # 距离条件
            if distance is not None and distance > 2.2:
                self.logger.debug(f"场地距离大于2.2米，忽略此场地，距离={distance}米")
                continue

            # 计算场地中心点（保留用于其他用途，如导航）
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), field_color, 2)
            if distance is not None:
                label_text = f"{field_name}: {distance:.2f}m"
            else:
                label_text = f"{field_name}: 未知距离"
            cv2.putText(frame, label_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, field_color, 2)
            self.logger.info(f"检测到{field_name}，距离={'{:.2f}'.format(distance) if distance is not None else '未知'}米，中心点=({center_x}, {center_y})")

            # 保存最新的场地距离和中心点
            self.latest_field_distance = distance
            self.latest_field_center = (center_x, center_y)

            return True

        return False

    def get_closest_field_distance(self, contour, depth_image):
        """
        获取场地中最近的像素点的距离。

        :param contour: 场地的轮廓
        :param depth_image: 深度图像
        :return: 最小距离（米）或 None
        """
        # 创建掩码仅包含当前场地轮廓
        mask = np.zeros_like(depth_image, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)  # 填充轮廓

        # 获取所有场地区域的像素点
        field_pixels = np.where(mask == 255)
        field_depths = depth_image[field_pixels]

        # 过滤无效深度值
        valid_depths = field_depths[(field_depths > 0) & (field_depths < 5000)]
        if valid_depths.size == 0:
            return None

        # 获取最小深度值
        min_depth = np.min(valid_depths) / 1000.0  # 转换为米

        # 添加到滑动窗口
        self.distance_deque.append(min_depth)

        # 计算滑动平均
        smoothed_distance = np.mean(self.distance_deque)

        return smoothed_distance

    def get_ball_color(self, frame, x1, y1, x2, y2):
        """获取球的颜色"""
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 定义颜色范围
        color_ranges = {
            'red': [([0, 114, 0], [8, 255, 255]), ([160, 100, 100], [179, 255, 255])],
            'blue': [([105, 121, 48], [117, 255, 165])],
            'yellow': [([15, 171, 104], [60, 255, 255])],
            'black': [([0, 0, 0], [180, 255, 70])]
        }

        # 使用膨胀与腐蚀进行形态学操作
        kernel = np.ones((5, 5), np.uint8)

        for color, ranges in color_ranges.items():
            mask = None
            for lower, upper in ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                current_mask = cv2.inRange(hsv, lower, upper)
                # 对掩码进行腐蚀和膨胀
                current_mask = cv2.erode(current_mask, kernel, iterations=BALL_EROSION_ITERATIONS)
                current_mask = cv2.dilate(current_mask, kernel, iterations=BALL_DILATION_ITERATIONS)
                if mask is None:
                    mask = current_mask
                else:
                    mask = cv2.bitwise_or(mask, current_mask)
            if cv2.countNonZero(mask) > 0:
                return color
        return 'unknown'

    def detect_and_process_balls(self, color_frame, depth_image):
        """任务一模式：搜索并夹取球体"""
        self.logger.info("任务一模式：搜索并夹取球体")

        ball_targets = []

        # 检测球体
        results = self.model.predict(source=color_frame, device=self.device, conf=self.confidence_threshold)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.class_labels.get(cls_id, f"Class {cls_id}")
                if label.lower() != 'ball' or confidence < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                depth = self.get_depth_at_point(depth_image, center_x, center_y)
                if depth is not None and depth > 0.0:
                    # 获取球颜色
                    ball_color = self.get_ball_color(color_frame, x1, y1, x2, y2)
                    if ball_color in self.color_priority:
                        ball_targets.append({
                            'label': label,
                            'distance': depth,
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'color': ball_color
                        })

                    color_box = (0, 255, 0)
                    label_text = f"{label}: {confidence:.2f}"
                    cv2.rectangle(color_frame, (x1, y1), (x2, y2), color_box, 2)
                    cv2.circle(color_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(color_frame, label_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        # 处理球体目标
        if ball_targets:
            if not self.target_locked:
                # 按颜色优先级排序
                ball_targets.sort(key=lambda t: self.color_priority.index(t['color']))
                self.current_target = ball_targets[0]
                self.target_locked = True
                self.logger.info(f"锁定目标，颜色: {self.current_target['color']}，距离: {self.current_target['distance']:.2f} 米")
            else:
                target_still_detected = any(
                    (t['color'] == self.current_target['color'] and
                     np.linalg.norm(np.array(t['center']) - np.array(self.current_target['center'])) < 20)
                    for t in ball_targets
                )
                if target_still_detected:
                    updated_target = next(
                        (t for t in ball_targets if t['color'] == self.current_target['color'] and
                         np.linalg.norm(np.array(t['center']) - np.array(self.current_target['center'])) < 20),
                        None
                    )
                    if updated_target:
                        self.current_target['distance'] = updated_target['distance']
                else:
                    self.target_locked = False
                    self.current_target = None
                    self.logger.info("目标丢失，停止。")
                    self.joystick_controller.send_command("5")
                    return
            # 检查自动模式是否被切换
            if not self.joystick_controller.get_auto_mode():
                self.logger.info("自动模式已关闭，停止球体处理。")
                self.target_locked = False
                self.current_target = None
                self.joystick_controller.send_command("5")
                return

            distance = self.current_target['distance']
            target_center_x, target_center_y = self.current_target['center']

            self.logger.info(f"当前目标距离: {distance:.3f} 米，目标距离设定: {self.target_distance:.3f} 米")

            frame_width = color_frame.shape[1]
            center_offset = target_center_x - (frame_width / 2)
            tolerance = frame_width * (self.error_factor / 10) * 0.15

            if distance <= APPROACH_THRESHOLD and distance > self.target_distance:
                self.logger.info("距离目标25cm，执行抓取动作序列。")
                self.joystick_controller.send_command("5")  # 停止
                time.sleep(TASK1_STOP_BEFORE_FORWARD)
                # 检查自动模式是否被切换
                if not self.joystick_controller.get_auto_mode():
                    self.logger.info("自动模式已关闭，停止抓取动作。")
                    return
                self.joystick_controller.send_command("1")  # 前进
                time.sleep(TASK1_FORWARD_TO_GRAB)
                self.joystick_controller.send_command("X")  # 夹取小球
                self.joystick_controller.send_command("5")  # 停止
                time.sleep(TASK1_STOP_AFTER_GRAB)
                self.logger.info("完成抓取动作序列，进入任务二模式。")
                self.task_state = 'TASK_TWO'  # 切换到任务二模式
                self.in_grab_action = False
            else:
                if center_offset > tolerance:
                    self.joystick_controller.send_command("3")
                    self.logger.info("目标在右侧，右转。")
                elif center_offset < -tolerance:
                    self.joystick_controller.send_command("2")
                    self.logger.info("目标在左侧，左转。")
                else:
                    self.joystick_controller.send_command("1")
                    self.logger.info("向目标前进。")
        else:
            # 未检测到球，执行搜索模式
            self.logger.info("未检测到球，执行搜索模式")
            self.joystick_controller.send_command("3")  # 右转
            time.sleep(FIELD_SEARCH_ROTATION_TIME)
            self.joystick_controller.send_command("5")  # 停止
            time.sleep(FIELD_SEARCH_PAUSE_TIME)

    def detect_and_process_fields(self, color_frame, depth_image):
        """任务二模式：搜索并处理场地"""
        self.logger.info("任务二模式：搜索场地")

        field_detected = self.detect_field_regions(color_frame, depth_image)

        if field_detected:
            # 检查自动模式是否被切换
            if not self.joystick_controller.get_auto_mode():
                self.logger.info("自动模式已关闭，停止场地处理。")
                return

            # 执行与场地相关的逻辑，例如靠近场地
            distance = self.latest_field_distance
            center_x, center_y = self.latest_field_center
            frame_width = color_frame.shape[1]
            frame_height = color_frame.shape[0]
            image_center_x = frame_width // 2
            image_center_y = frame_height // 2

            # 计算中心偏移量
            offset_x = center_x - image_center_x
            offset_y = center_y - image_center_y  # 可用于未来的深度调整

            # 定义一个容忍度范围
            TOLERANCE_X = frame_width * 0.05  # 5%的容忍度

            self.logger.info(f"场地中心偏移: X={offset_x}, Y={offset_y}, 距离={'{:.2f}'.format(distance) if distance is not None else '未知'}米")

            if not self.in_grab_action and distance is not None and distance < APPROACH_THRESHOLD:
                self.logger.info("距离场地25厘米，执行放置动作序列。")
                self.in_grab_action = True
                self.joystick_controller.send_command("5")  # 停止
                time.sleep(TASK2_STOP_BEFORE_FORWARD)
                # 检查自动模式是否被切换
                if not self.joystick_controller.get_auto_mode():
                    self.logger.info("自动模式已关闭，停止放置动作。")
                    return
                self.joystick_controller.send_command("1")  # 前进
                time.sleep(TASK2_FORWARD_TO_PLACE)
                self.joystick_controller.send_command("X")  # 放置小球
                self.joystick_controller.send_command("5")  # 停止
                time.sleep(TASK2_STOP_AFTER_PLACE)
                # 任务二完成后，发送 'Y' 指令打开夹子
                self.joystick_controller.send_command("Y")
                self.logger.info("完成放置动作序列，执行延迟程序。")
                # 延迟程序开始
                self.joystick_controller.send_command("5")  # 停止
                time.sleep(STOP_AFTER_TASK2)
                self.joystick_controller.send_command("4")  # 后退
                time.sleep(REVERSE_AFTER_TASK2)
                self.joystick_controller.send_command("5")  # 停止
                time.sleep(STOP_AFTER_REVERSE)
                self.joystick_controller.send_command("3")  # 右转
                time.sleep(TURN_RIGHT_AFTER_REVERSE)
                self.joystick_controller.send_command("5")  # 停止
                time.sleep(STOP_AFTER_TURN)
                # 延迟程序结束
                self.logger.info("延迟程序完成，返回任务一模式。")
                self.task_state = 'SEARCH_BALL'  # 切换回任务一模式
                self.in_grab_action = False
            elif self.in_grab_action and (distance is None or distance > EXIT_THRESHOLD):
                self.logger.info("距离场地大于退出阈值，停止放置动作。")
                self.in_grab_action = False
                self.joystick_controller.send_command("5")  # 停止
            else:
                if distance is not None and distance >= APPROACH_THRESHOLD:
                    # 调整方向
                    if abs(offset_x) > TOLERANCE_X:
                        if offset_x > 0:
                            self.logger.info("场地在右侧，缓慢右转")
                            self.joystick_controller.send_command("3")  # 右转
                            time.sleep(0.2)
                            self.joystick_controller.send_command("5")  # 停止
                            time.sleep(0.1)
                        else:
                            self.logger.info("场地在左侧，缓慢左转")
                            self.joystick_controller.send_command("2")  # 左转
                            time.sleep(0.2)
                            self.joystick_controller.send_command("5")  # 停止
                            time.sleep(0.1)
                    else:
                        # 场地居中，继续前进
                        if distance > 1.0:
                            self.logger.info("距离大于1米，快速前进")
                            self.joystick_controller.send_command("1")
                            time.sleep(0.3)
                        else:
                            self.logger.info("距离小于1米，缓慢前进")
                            self.joystick_controller.send_command("1")
                            time.sleep(0.2)
                            self.joystick_controller.send_command("5")
                            time.sleep(0.1)
        else:
            # 未检测到场地时的搜索策略
            self.logger.info("未检测到场地，执行搜索模式")
            self.joystick_controller.send_command("3")  # 右转
            time.sleep(FIELD_SEARCH_ROTATION_TIME)
            self.joystick_controller.send_command("5")  # 停止
            time.sleep(FIELD_SEARCH_PAUSE_TIME)

    def send_single_command(self, command):
        """发送单个指令"""
        if command == "5":
            self.joystick_controller.send_command(command)
            self.current_command = None
        elif self.current_command != command:
            self.joystick_controller.send_command(command)
            self.current_command = command

    def reset_command(self):
        """重置当前指令"""
        if self.current_command is not None:
            self.joystick_controller.send_command("5")
            self.current_command = None
            self.logger.info("指令已重置。")

    def process_frame(self, color_frame, depth_image):
        """处理帧并进行目标检测"""
        if self.model is None:
            self.logger.warning("模型未加载，无法进行检测。")
            return False

        # 检测是否切换了模式
        current_auto_mode = self.joystick_controller.get_auto_mode()
        if current_auto_mode != getattr(self, 'prev_auto_mode', None):
            self.reset_command()
            if current_auto_mode:
                self.logger.info("重新进入自动模式。")
            else:
                self.logger.info("切换到手动模式。")
            self.prev_auto_mode = current_auto_mode

        # 在自动模式下才处理自动任务
        if current_auto_mode:
            # 根据当前任务状态选择检测模式
            if self.task_state == 'SEARCH_BALL':
                self.detect_and_process_balls(color_frame, depth_image)
            elif self.task_state == 'TASK_TWO':
                self.detect_and_process_fields(color_frame, depth_image)
        else:
            # 手动模式下，停止所有自动指令
            self.reset_command()
            self.target_locked = False
            self.current_target = None

        # FPS 计算和显示
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        if elapsed_time > 0:
            fps = 1 / elapsed_time
            self.fps_deque.append(fps)
            if len(self.fps_deque) > 30:
                self.fps_deque.pop(0)
            average_fps = sum(self.fps_deque) / len(self.fps_deque)
            cv2.putText(color_frame, f"平均 FPS: {average_fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        self.prev_time = current_time

        # 可视化图像中心和场地中心
        frame_height, frame_width = color_frame.shape[:2]
        cv2.circle(color_frame, (frame_width // 2, frame_height // 2), 5, (255, 0, 0), -1)  # 图像中心
        if self.latest_field_center:
            cv2.circle(color_frame, self.latest_field_center, 5, (0, 255, 0), -1)  # 场地中心

        cv2.imshow('YOLO Object Detection with Depth', color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.logger.info("退出程序。")
            return True

        return False

    def run(self):
        """运行目标检测和距离计算"""
        self.logger.info("进入自主模式，发送 'Y' 命令确保夹子打开。")
        self.joystick_controller.send_command("Y")

        self.logger.info("等待模型加载完成...")
        self.model_loaded.wait()
        if self.model is None:
            self.logger.error("模型加载失败，无法运行处理器。")
            return

        try:
            while True:
                color_image, depth_image = self.image_subscriber.get_images()
                if color_image is None or depth_image is None:
                    self.logger.debug("等待图像...")
                    time.sleep(0.1)
                    continue

                stop = self.process_frame(color_image, depth_image)

                if stop:
                    break

        except KeyboardInterrupt:
            self.logger.info("捕获到键盘中断，退出...")
        finally:
            cv2.destroyAllWindows()
            if 'joystick_controller' in locals():
                self.joystick_controller.stop()  # 停止摇杆输入处理
            if 'image_subscriber' in locals():
                self.image_subscriber.destroy_node()
            rclpy.shutdown()
            # pygame.quit()  # 让 JoystickController 处理 Pygame 退出

# ==================== ROS2 节点运行器 ====================

def ros_spin(node):
    rclpy.spin(node)

# ==================== 主程序 ====================

def main_program():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('MainProgram')

    try:
        # 初始化 ROS2
        rclpy.init()
        image_subscriber = ImageSubscriber()

        # 初始化摇杆控制器
        joystick_controller = JoystickController(serial_port='/dev/ttyUSB0', baudrate=115200, timeout=1)

        # 创建 YOLO 处理器并传入摇杆控制器
        processor = YOLOProcessor(
            model_path='train14.pt',
            image_subscriber=image_subscriber,
            joystick_controller=joystick_controller,
            confidence_threshold=0.7,
            target_distance=0.21,
            search_color=SEARCH_COLOR,
            error_factor=ERROR_FACTOR
        )

        # 在单独的线程中启动 ROS2 spin
        ros_thread = threading.Thread(target=ros_spin, args=(image_subscriber,), daemon=True)
        ros_thread.start()

        # 在单独的线程中启动摇杆输入处理
        joystick_thread = threading.Thread(target=joystick_controller.process_inputs, daemon=True)
        joystick_thread.start()

        # 运行 YOLO 处理器
        processor.run()

    except Exception as e:
        logger.error(f"程序运行出错: {e}")
    finally:
        # 清理资源
        cv2.destroyAllWindows()
        if 'joystick_controller' in locals():
            joystick_controller.stop()
            if joystick_controller.ser:
                joystick_controller.ser.close()
        if 'image_subscriber' in locals():
            image_subscriber.destroy_node()
        rclpy.shutdown()
        # pygame.quit()  # 让 JoystickController 处理 Pygame 退出

if __name__ == "__main__":
    main_program()
