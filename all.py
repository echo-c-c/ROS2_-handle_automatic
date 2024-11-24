import time
import threading
import serial
import pygame
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ==================== 配置参数 ====================
ERROR_FACTOR = 7  # 误差参数，范围1到10，数值越大容忍度越高
BACKWARD_DURATION = 2  # 后退持续时间（秒）
RIGHT_TURN_DURATION = 5  # 右转持续时间（秒）
SEARCH_COLOR = 'red'  # 搜索球的颜色（'red', 'blue', 'black', 'yellow'）
COLOR_PRIORITY = ['yellow', 'black', SEARCH_COLOR]  # 颜色优先级：黄色 > 黑色 > SEARCH_COLOR

# 新增延迟参数
STOP_DURATION = 0.5  # 停止持续时间（秒）
FORWARD_DURATION = 0.8  # 前进持续时间（秒）
RIGHT_TURN_AFTER_COMPLETE_DURATION = 1.8  # 完成后右转持续时间（秒）
FORWARD_LONG_DURATION = 7  # 长时间前进持续时间（秒）

# HSV颜色范围配置（用于检测红色和蓝色场地）
# 红色场地HSV范围
LOWER_RED1 = np.array([0, 50, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([160, 50, 50])
UPPER_RED2 = np.array([180, 255, 255])

# 蓝色场地HSV范围
LOWER_BLUE = np.array([122, 72, 60])
UPPER_BLUE = np.array([159, 255, 255])

# 紫色边框HSV范围
LOWER_PURPLE = np.array([130, 30, 30])
UPPER_PURPLE = np.array([155, 255, 255])


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
        """处理彩色图像回调函数，将 ROS 图像消息转换为 OpenCV 格式"""
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
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


# ==================== YOLO 处理器类 ====================

class YOLOProcessor:
    def __init__(self, model_path, image_subscriber, joystick_controller, confidence_threshold=0.6,
                 target_distance=0.17, search_color=SEARCH_COLOR, error_factor=ERROR_FACTOR):
        """
        初始化 YOLO 模型及相关。

        :param model_path: YOLO 模型文件路径
        :param image_subscriber: 图像订   器实例
        :param joystick_controller: 摇杆控制器实例
        :param confidence_threshold: 置信度阈值
        :param target_distance: 目标距离（米）
        :param search_color: 搜索的球的颜色（'red', 'blue', 'black', 'yellow'）
        :param error_factor: 误差参数，范围1到10
        """
        self.model = None
        self.model_loaded = threading.Event()  # 线程事件，用于标识模型是否加载完成
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置计算设备为 GPU 或 CPU

        # 启动模型加载线程
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()

        # 使用提供的图像订阅器和摇杆控制器
        self.image_subscriber = image_subscriber
        self.joystick_controller = joystick_controller
        self.target_distance = target_distance
        self.current_target = None
        self.target_locked = False  # 标识当前是否锁定了一个目标

        # 初始化 FPS 计算
        self.prev_time = time.time()
        self.fps_deque = []

        # ==================== 相机内参参数 ====================
        self.fx = 476.2635192871094  # 焦距 x
        self.fy = 476.2635192871094  # 焦距 y
        self.cx = 315.7132873535156  # 主点 x
        self.cy = 199.8099365234375  # 主点 y

        # 相机矩阵和畸变系数
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                       [0, self.fy, self.cy],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 假设无畸变

        # ==================== 已知目标物体尺寸（米） ====================
        self.known_width = {

            'ball': 0.030,  # 球的平均直径
            # 根据需要添加更多类别和已知宽度
        }
        self.known_height = {

            'ball': 0.040,  # 球的平均高度
            # 根据需要添加更多类别和已知高度
        }

        self.current_command = None  # 当前发送的指令

        # ==================== 任务状态管理 ====================
        self.task_state = 'SEARCH_BALL'  # 初始化任务状态为搜索球
        self.color_priority = COLOR_PRIORITY  # 颜色优先级
        self.search_color = search_color.lower()  # 确保颜色为小写
        self.completion_done = False  # 标识是否完成后续动作

        # 延迟参数
        self.backward_duration = BACKWARD_DURATION  # 后退持续时间
        self.right_turn_duration = RIGHT_TURN_DURATION  # 右转持续时间
        self.stop_duration = STOP_DURATION  # 停止持续时间
        self.forward_duration = FORWARD_DURATION  # 前进持续时间
        self.right_turn_after_complete_duration = RIGHT_TURN_AFTER_COMPLETE_DURATION  # 完成后右转持续时间
        self.forward_long_duration = FORWARD_LONG_DURATION  # 长时间前进持续时间

        self.error_factor = error_factor  # 误差参数

    def load_model(self):
        """
        加载 YOLO 模型并设置加载完成事件。
        """
        print("开始加载模型...")
        try:
            # 确保使用的是 .pt 模型
            if not self.model_path.endswith('.pt'):
                raise ValueError(
                    f"模型路径错误: '{self.model_path}' 不是一个 .pt 型文件。请使用有效的 PyTorch 模型文件。")

            # 加载模型
            self.model = YOLO(self.model_path)
            # 设置设备
            self.model.to(self.device)
            print("模型加载成功。")
            self.class_labels = self.model.names
            print("名称:", self.class_labels)
            self.model_loaded.set()  # 设置事件，表示模型加载完成
        except Exception as e:
            print(f"加载型失败: {e}")
            self.model = None
            self.model_loaded.set()  # 即使失败也设置事件，防止阻塞

    def undistort_points(self, x, y):
        """
        校正单个点的头畸变。

        :param x: 点的 x 坐标
        :param y: 点的 y 坐标
        :return: 校正后的点坐标
        """
        points = np.array([[[x, y]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        return undistorted[0][0]

    def get_depth_at_point(self, depth_image, x, y, window_size=5):
        """
        获取指定点周围的深度值。

        :param depth_image: 深度图像
        :param x: 点的 x 坐标
        :param y: 点的 y 坐标
        :param window_size: 窗口大小
        :return: 深度值（米）或 None
        """
        half_size = window_size // 2
        window = depth_image[max(y - half_size, 0):min(y + half_size + 1, depth_image.shape[0]),
                 max(x - half_size, 0):min(x + half_size + 1, depth_image.shape[1])]
        # 过滤掉无效的深度值（假设 0 为无效值）
        valid_depth = window[(window > 0) & (window < np.inf)]
        if valid_depth.size == 0:
            return None
        # 计算中位数深度以减少噪声
        depth = np.median(valid_depth) / 1000.0  # 假设深度单位为毫米，转换为米
        return depth

    def estimate_distance_width(self, bbox_width_pixels, class_label):
        """
        使用已知物体宽度和相机距估计距离。

        :param bbox_width_pixels: 边界框的像素宽度
        :param class_label: 物体类别标签
        :return: 估计的距离（米）或 None
        """
        if class_label in self.known_width:
            real_width = self.known_width[class_label]
            if bbox_width_pixels > 0:
                # 距离估计算法
                distance = (real_width * self.fx) / bbox_width_pixels
                return distance
        return None

    def estimate_distance_height(self, bbox_height_pixels, class_label):
        """
        使用已知物体高度和相机焦距估计距离。

        :param bbox_height_pixels: 边界框的像素高度
        :param class_label: 物体类别标签
        :return: 估计的距离（米）或 None
        """
        if class_label in self.known_height:
            real_height = self.known_height[class_label]
            if bbox_height_pixels > 0:
                distance = (real_height * self.fy) / bbox_height_pixels
                return distance
        return None

    def detect_color_region(self, frame):
        """
        在帧中检测指定颜色的矩形区域，排除圆形区域以避免检测球或球形物体。

        :param frame: 输入的彩色帧
        :return: 是否找到目标区域，区域的边界框，颜色掩码
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义红色、蓝色、黄色和黑色的范围
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])

        # 创建掩码
        red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
        black_mask = cv2.inRange(hsv_frame, lower_black, upper_black)

        # 根据颜色优先级选择掩码
        mask = None
        for color in self.color_priority:
            if color == 'red':
                current_mask = red_mask
            elif color == 'blue':
                current_mask = blue_mask
            elif color == 'yellow':
                current_mask = yellow_mask
            elif color == 'black':
                current_mask = black_mask
            else:
                current_mask = None

            if current_mask is not None and cv2.countNonZero(current_mask) > 0:
                mask = current_mask
                break  # 找到优先级最高的匹配颜色后退出循环

        if mask is not None:
            # 进行形态学操作以去除噪声
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 在掩码中查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 排除圆形轮廓
            non_circular_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area == 0:
                    continue
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity < 0.7:  # 调整阈值根据需要
                    non_circular_contours.append(contour)
                else:
                    # 调试时可打印或绘制被排除的圆形轮廓
                    print("排除圆形轮廓，圆形度:", circularity)

            # 寻找最大的非圆形轮廓
            if non_circular_contours:
                largest_contour = max(non_circular_contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                # 调整面积阈值根据需要
                area_threshold = frame.shape[0] * frame.shape[1] * 0.05  # 例如，占帧面积的5%

                if area > area_threshold:
                    # 找到一个大的区域
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    return True, (x, y, w, h), mask
        return False, None, mask

    def detect_field_regions(self, frame, depth_image):
        """
        检测帧中的红色和蓝色场地，确保不会同时出现，且必须与长条状紫色区域相连。
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建掩膜
        mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        mask_purple = cv2.inRange(hsv, LOWER_PURPLE, UPPER_PURPLE)

        # 形态学操作
        kernel = np.ones((9, 9), np.uint8)  # 使用更大的核
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)

        def find_purple_strips():
            """查找长条状的紫色区域"""
            purple_contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            purple_strips = []

            for contour in purple_contours:
                rect = cv2.minAreaRect(contour)
                width = min(rect[1])
                height = max(rect[1])
                if width == 0:
                    continue

                # 检查是否为长条状（长宽比大于3）
                aspect_ratio = height / width
                if aspect_ratio > 2.5:
                    purple_strips.append(contour)

            return purple_strips

        def check_connection_with_purple(contour, purple_strips):
            """检查轮廓是否与紫色长条相连"""
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

        def process_field(mask, field_name, field_color, purple_strips):
            """处理单个场地的检测"""
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 2000:  # 最小面积阈值
                    continue

                rect = cv2.minAreaRect(contour)
                width = min(rect[1])
                height = max(rect[1])
                if width == 0:
                    continue

                aspect_ratio = height / width
                if aspect_ratio < 1.5:  # 确保是长方形
                    continue

                if not check_connection_with_purple(contour, purple_strips):
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                distance = self.get_field_distance(x, y, w, h, depth_image)

                # 过滤掉远距离的检测
                if distance is not None and distance > 2.2:  # 只保留2.2米以内的场地
                    continue

                # 绘制标记
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                if distance is not None:
                    label_text = f"{field_name}: {distance:.2f}m"
                else:
                    label_text = f"{field_name}"
                cv2.putText(frame, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                return True

            return False

        purple_strips = find_purple_strips()
        if not purple_strips:
            return  # 如果没有找到紫色长条，直接返回

        red_found = process_field(mask_red, "Red Field", (0, 0, 255), purple_strips)

        # 只有在没有找到红色场地时才检查蓝色场地
        if not red_found:
            process_field(mask_blue, "Blue Field", (255, 0, 0), purple_strips)

    def process_frame(self, color_frame, depth_image):
        """
        使用 YOLO 模型进行目标检测并计算每个目标的距离。

        :param color_frame: 彩色帧
        :param depth_image: 深度图像
        :return: 是否需要停止程序的标志
        """
        if self.model is None:
            print("模型未加载，无法进行检测。")
            return False

        # 检查自动模式状态是否变化
        if self.joystick_controller.auto_mode != getattr(self, 'prev_auto_mode', None):
            self.reset_command()
            if self.joystick_controller.auto_mode:
                print("重新进入自动模式。")
            else:
                print("切换到手动模式。")
            self.prev_auto_mode = self.joystick_controller.auto_mode

        # 无论模式如何，都进行预测以显示检测结果
        results = self.model.predict(source=color_frame, device=self.device, conf=self.confidence_threshold)

        # 处理检测结果
        ball_targets = []
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

                # 绘制检测框和标签
                color_box = (0, 255, 0)  # 绿色
                label_text = f"{label}: {confidence:.2f}"
                cv2.rectangle(color_frame, (x1, y1), (x2, y2), color_box, 2)
                cv2.circle(color_frame, (center_x, center_y), 5, (0, 0, 255), -1)  # 中心点
                cv2.putText(color_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        # 检测并标记场地区域 - 更新为传入深度图像
        self.detect_field_regions(color_frame, depth_image)

        # 自主控制逻辑
        if self.joystick_controller.auto_mode:
            if self.task_state == 'SEARCH_BALL':
                self.search_ball(ball_targets, color_frame)

        # 计算并显示 FPS
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

        # 显示帧
        cv2.imshow('YOLO Object Detection with Depth', color_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("退出程序。")
            return True  # 指示需要停止

        return False  # 无需停止

    def send_single_command(self, command):
        """
        发送单个指令，确保指令冲突。

        :param command: 要发送的令
        """
        if command == "5":
            # 总是发送停止指令并重置当前指令
            self.joystick_controller.send_command(command)
            self.current_command = None
        elif self.current_command != command:
            # 发送新指令
            self.joystick_controller.send_command(command)
            self.current_command = command

    def reset_command(self):
        """
        重置当前指令。
        """
        if self.current_command is not None:
            self.joystick_controller.send_command("5")  # 停止
            self.current_command = None
            print("指令已重置。")

    def run(self):
        """
        运行目标检测和距离计算。
        """
        # 进入自主模式时发送 'Y' 命令确保夹子打开
        print("进入自主模式，发送 'Y' 命令确保夹子打开。")
        self.joystick_controller.send_command("Y")

        # 等待模型加载完成
        print("等待模型加载完成...")
        self.model_loaded.wait()
        if self.model is None:
            print("模型加载失败，无法运行处理器。")
            return

        try:
            while True:
                # 从 ROS2 获取图像
                color_image, depth_image = self.image_subscriber.get_images()
                if color_image is None or depth_image is None:
                    print("等待图像...")
                    time.sleep(0.1)
                    continue

                # 处理帧
                stop = self.process_frame(color_image, depth_image)

                if stop:
                    break

        except KeyboardInterrupt:
            print("捕获到键盘中断，退出...")
        finally:
            cv2.destroyAllWindows()

    def get_ball_color(self, frame, x1, y1, x2, y2):
        """
        获取球的颜色。

        :param frame: 彩色帧
        :param x1: 边界框左上角 x 坐标
        :param y1: 边界框左上角 y 坐标
        :param x2: 边界框右下角 x 坐标
        :param y2: 边界框右下角 y 坐标
        :return: 球的颜色（'red', 'blue', 'yellow', 'black' 或 'unknown'）
        """
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 定义颜色范围
        color_ranges = {
            'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [179, 255, 255])],
            'blue': [([100, 150, 0], [140, 255, 255])],
            'yellow': [([20, 100, 100], [30, 255, 255])],
            'black': [([0, 0, 0], [180, 255, 30])]
        }

        for color, ranges in color_ranges.items():
            mask = None
            for lower, upper in ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                if mask is None:
                    mask = cv2.inRange(hsv, lower, upper)
                else:
                    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
            if cv2.countNonZero(mask) > 0:
                return color
        return 'unknown'

    def get_field_distance(self, x, y, w, h, depth_image):
        """
        获取场地区域的距离。

        :param x: 场地边界框的x坐标
        :param y: 场地边界框的y坐标
        :param w: 场地边界框的宽度
        :param h: 场地边界框的高度
        :param depth_image: 深度图像
        :return: 距离（米）或 None
        """
        # 计算场地中心点
        center_x = x + w // 2
        center_y = y + h // 2

        # 创建ROI，使用5x5的窗口来获取更稳定的深度值
        roi_size = 5
        roi_x_start = max(0, center_x - roi_size // 2)
        roi_x_end = min(depth_image.shape[1], center_x + roi_size // 2 + 1)
        roi_y_start = max(0, center_y - roi_size // 2)
        roi_y_end = min(depth_image.shape[0], center_y + roi_size // 2 + 1)

        # 获取ROI区域的深度值
        depth_roi = depth_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # 过滤掉无效的深度值（0和异常值）
        valid_depths = depth_roi[(depth_roi > 0) & (depth_roi < 5000)]

        if len(valid_depths) > 0:
            # 使用中位数来获取稳定的距离值
            distance = np.median(valid_depths) / 1000.0  # 转换为米
            return distance
        return None

    def search_ball(self, ball_targets, color_frame):
        """搜索并处理球体。"""
        if ball_targets:
            if not self.target_locked:
                # 按颜色优先级锁定目标
                ball_targets.sort(key=lambda t: self.color_priority.index(t['color']))
                self.current_target = ball_targets[0]
                self.target_locked = True
                print(f"锁定目标，颜色: {self.current_target['color']}，距离: {self.current_target['distance']:.2f} 米")
            else:
                # 检查锁定的目标是否仍然被检测到
                target_still_detected = any(
                    (t['color'] == self.current_target['color'] and
                     np.linalg.norm(np.array(t['center']) - np.array(self.current_target['center'])) < 20)
                    for t in ball_targets
                )
                if target_still_detected:
                    # 更新距离
                    updated_target = next(
                        (t for t in ball_targets if t['color'] == self.current_target['color'] and
                         np.linalg.norm(np.array(t['center']) - np.array(self.current_target['center'])) < 20),
                        None
                    )
                    if updated_target:
                        self.current_target['distance'] = updated_target['distance']
                else:
                    # 目标丢失
                    self.target_locked = False
                    self.current_target = None
                    print("目标丢失，停止。")
                    self.joystick_controller.send_command("5")  # 停止
                    return

            distance = self.current_target['distance']
            target_center_x, target_center_y = self.current_target['center']

            print(f"当前目标距离: {distance:.3f} 米，目标距离设定: {self.target_distance:.3f} 米")

            frame_width = color_frame.shape[1]
            center_offset = target_center_x - (frame_width / 2)
            tolerance = frame_width * (self.error_factor / 10) * 0.15  # 根据误差参数调整偏移容忍度

            # 移动逻辑
            if distance <= 0.25 and distance > self.target_distance:
                print("距离目标25cm，执行动作序列并退出自动模式。")
                self.joystick_controller.send_command("5")  # 停止
                time.sleep(self.stop_duration)
                self.joystick_controller.send_command("1")  # 前进
                time.sleep(self.forward_duration)
                self.joystick_controller.send_command("X")  # 夹取小球
                self.joystick_controller.send_command("5")  # 停止
                time.sleep(self.stop_duration)
                self.joystick_controller.send_command("3")  # 右转
                time.sleep(self.right_turn_after_complete_duration)
                self.joystick_controller.send_command("1")  # 前进
                time.sleep(self.forward_long_duration)
                print("释放小球。")
                self.joystick_controller.send_command("Y")  # 释放小球
                self.joystick_controller.send_command("5")  # 停止

                # 退出自动模式
                self.joystick_controller.auto_mode = False
                print("自动模式结束。")

            else:
                # 当距离大于目标距离时，继续前进并调整方向
                if center_offset > tolerance:
                    # 目标在右侧，右转
                    self.joystick_controller.send_command("3")  # 右转
                    print("目标在右侧，右转。")
                elif center_offset < -tolerance:
                    # 目标在左侧，左转
                    self.joystick_controller.send_command("2")  # 左转
                    print("目标在左侧，左转。")
                else:
                    # 目标在中央，前进
                    self.joystick_controller.send_command("1")  # 前进
                    print("向目标前进。")


# ==================== 摇杆控制模块 ====================

class JoystickController:
    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=115200, timeout=1):
        """初始化摇杆控制器，包括 pygame 和串口通信"""
        # 初始化 pygame 和摇杆
        pygame.init()
        pygame.joystick.init()

        # 初始化串口
        self.ser = None  # 串口变量
        try:
            self.ser = serial.Serial(
                port=serial_port,  # 设备名称，根据实际设置修改
                baudrate=baudrate,  # 波特率
                timeout=timeout  # 超时时间
            )

            if self.ser.isOpen():
                print("串口成功打开。")

        except serial.SerialException as e:
            print(f"无法打开串口: {e}")

        # 检查是否连接了摇杆
        if pygame.joystick.get_count() == 0:
            print("未检测到摇杆。")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)  # 使用第一个摇杆
            self.joystick.init()
            print(f"检测到摇杆: {self.joystick.get_name()}")

        # 初始化状态变量
        self.last_command = ""
        self.button_states = [False] * 4  # A, B, X, Y 按钮状态
        self.deadzone = 0.2  # 定死区，防止摇杆微小抖动引起动作
        self.lb_pressed = False  # 左肩部按钮状态
        self.rb_pressed = False  # 右肩部按钮状态
        self.auto_mode = True  # 添加自动模式标志，默认为True

    def send_command(self, command):
        """
        通过串口发送指令。

        :param command: 要发送的指令
        """
        if self.ser and self.ser.isOpen():
            try:
                self.ser.write(f"{command}\r\n".encode())
                print(f"发送指令: {command}")
            except serial.SerialException as e:
                print(f"串口通信错误: {e}")

    def process_inputs(self):
        """
        处理摇杆输入并发送相应的串口指令。
        """
        try:
            while True:
                pygame.event.pump()  # 处理内部 pygame 事件

                if self.joystick:
                    # 检查左肩部按钮是否被按下（假左肩部按钮索引为4）
                    lb_button = self.joystick.get_button(4)
                    if lb_button and not self.lb_pressed:
                        self.lb_pressed = True
                        print("左肩部按钮按下，切换到手动模式。")
                        # 停止自动模式
                        self.auto_mode = False
                        # 发送停止指令
                        self.send_command("5")
                    elif not lb_button:
                        self.lb_pressed = False

                    # 检查右肩部按钮是否被按下以切换回自动模式
                    rb_button = self.joystick.get_button(5)  # 假设右肩部按钮索引为5
                    if rb_button and not self.rb_pressed:
                        self.rb_pressed = True
                        print("右肩部按钮按下，切换回自动模式。")
                        # 启动自动模式
                        self.auto_mode = True
                    elif not rb_button:
                        self.rb_pressed = False

                    if not self.auto_mode:
                        # 在手动模式下此遥控控制
                        # 获取左摇杆轴
                        left_stick_x = self.joystick.get_axis(0)  # 左摇杆 X 轴
                        left_stick_y = self.joystick.get_axis(1)  # 左摇杆 Y 轴

                        command = ""

                        # 定义死区
                        deadzone = self.deadzone

                        # 检查摇杆位置
                        if abs(left_stick_x) < deadzone and abs(left_stick_y) < deadzone:
                            if self.last_command != "5":
                                command = "5"  # 停止
                        else:
                            if left_stick_y < -deadzone:
                                command = "1"  # 前进
                            elif left_stick_y > deadzone:
                                command = "4"  # 后退
                            if left_stick_x < -deadzone:
                                command = "2"  # 左转
                            elif left_stick_x > deadzone:
                                command = "3"  # 右转

                        # 如果指令变化则发送
                        if command and command != self.last_command:
                            self.send_command(command)
                            self.last_command = command

                    else:
                        # 在自动模式下，确保 last_command 被重置
                        self.last_command = ""

                    # 处理其他按钮
                    buttons = ["A", "B", "X", "Y"]
                    button_indices = [0, 1, 2, 3]
                    for i, button_name in enumerate(buttons):
                        button_state = self.joystick.get_button(button_indices[i])
                        if button_state != self.button_states[i]:
                            if button_state:
                                self.send_command(f"{button_name} pressed")
                            else:
                                self.send_command(f"{button_name} released")
                            self.button_states[i] = button_state

                time.sleep(0.1)  # 每0.1秒处理一次

        except KeyboardInterrupt:
            print("\n退出摇杆控制器。")

        finally:
            if self.ser:
                self.ser.close()
                print("串口已关闭。")
            pygame.quit()


# ==================== ROS2 节点运行器 ====================

def ros_spin(node):
    rclpy.spin(node)


# ==================== 主程序 ====================

def main_program():
    # 初始化 ROS2
    rclpy.init()
    image_subscriber = ImageSubscriber()

    # 初始化摇杆控制器
    joystick_controller = JoystickController(serial_port='/dev/ttyUSB0', baudrate=115200, timeout=1)

    # 创建 YOLO 处理器并传入摇杆控制器
    processor = YOLOProcessor(
        model_path='train14.pt',
        image_subscriber=image_subscriber,
        joystick_controller=joystick_controller,  # 传入摇杆控制
        confidence_threshold=0.7,
        target_distance=0.17,  # 默认目标距离为17厘米
        search_color=SEARCH_COLOR  # 使用配置参数中的搜索颜
    )

    # 在单独的线程中启动 ROS2 spin
    ros_thread = threading.Thread(target=ros_spin, args=(image_subscriber,), daemon=True)
    ros_thread.start()

    # 在单独的线程中启动摇杆输入处理
    joystick_thread = threading.Thread(target=joystick_controller.process_inputs, daemon=True)
    joystick_thread.start()

    # 运行 YOLO 处理器
    processor.run()

    # 关闭 ROS2
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main_program()