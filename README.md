# 自主目标检测与控制系统

## 概述

`2.py` 是一个自主系统，旨在使用 ROS2、YOLO 和摇杆输入进行实时目标检测和机器人控制。该系统结合了图像处理、深度感知和基于摇杆的手动控制，能够在预定义环境中执行搜索和操作物体等任务。

## 功能

- **ROS2 图像订阅**：订阅深度和彩色图像主题，实现实时图像处理。
- **YOLO 目标检测**：使用 YOLO（You Only Look Once）模型检测和分类摄像头画面中的物体。
- **摇杆控制**：集成摇杆用于手动控制，允许用户在自动模式和手动模式之间切换。
- **串口通信**：通过串口与硬件组件通信，根据检测到的对象发送指令。
- **深度感知**：利用深度图像计算物体距离，以做出智能决策。
- **任务序列**：执行预定义的动作序列，如抓取和放置物体。
- **日志与调试**：实现详细的日志记录，便于监控系统性能和排除故障。

## 目录

- [安装](#安装)
- [使用方法](#使用方法)
- [配置](#配置)
- [依赖项](#依赖项)
- [文件结构](#文件结构)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 安装

### 前提条件

- **ROS2**：确保已在系统中安装并正确配置 ROS2。
- **Python 3.8+**：脚本使用 Python 编写，需 Python 3.8 或更高版本。
- **串口访问权限**：系统需通过串口与硬件通信（例如 `/dev/ttyUSB0`）。

### 步骤

1. **克隆仓库**

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. **创建虚拟环境**

    推荐使用虚拟环境管理依赖项。

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **安装依赖项**

    ```bash
    pip install -r requirements.txt
    ```

    *如果没有提供 `requirements.txt`，请手动安装必要的包：*

    ```bash
    pip install opencv-python rclpy sensor_msgs cv_bridge numpy torch ultralytics pygame pyserial
    ```

4. **设置 ROS2 环境**

    确保已 source 你的 ROS2 环境。

    ```bash
    source /opt/ros/<ros2-distro>/setup.bash
    ```

    将 `<ros2-distro>` 替换为你的 ROS2 发行版，例如 `humble`。

## 使用方法

1. **连接硬件**

    - **摄像头**：确保深度和彩色摄像头已连接并发布到相应的 ROS2 主题。
    - **摇杆**：将摇杆连接到系统。
    - **串口设备**：将串口设备（例如机器人控制器）连接到指定的串口。

2. **运行脚本**

    ```bash
    python 2.py
    ```

3. **操作模式**

    - **自动模式**：系统根据 YOLO 检测和深度信息自动检测并与物体互动。
    - **手动模式**：使用摇杆手动控制机器人动作。按下左或右肩部按钮切换模式。

4. **退出**

    在显示窗口按 `q` 或在终端中使用 `Ctrl+C` 优雅退出程序。

## 配置

脚本顶部包含多个可配置参数。根据具体的设置和需求调整这些参数。

### 主要配置参数

- **误差因子和阈值**

    ```python
    ERROR_FACTOR = 6
    APPROACH_THRESHOLD = 0.25  # 米
    EXIT_THRESHOLD = 0.3        # 米
    ```

- **任务序列时间**

    ```python
    TASK1_STOP_BEFORE_FORWARD = 0.5
    TASK1_FORWARD_TO_GRAB = 0.8
    TASK1_STOP_AFTER_GRAB = 0.5
    
    TASK2_STOP_BEFORE_FORWARD = 0.5
    TASK2_FORWARD_TO_PLACE = 0.9
    TASK2_STOP_AFTER_PLACE = 0.5
    ```

- **颜色设置**

    ```python
    SEARCH_COLOR = 'red'
    COLOR_PRIORITY = [SEARCH_COLOR, 'yellow', 'black']
    ```

- **串口配置**

    ```python
    serial_port='/dev/ttyUSB0'
    baudrate=115200
    timeout=1
    ```

- **YOLO 模型路径**

    ```python
    model_path='train14.pt'
    ```

- **相机校准参数**

    ```python
    fx = 476.2635192871094
    fy = 476.2635192871094
    cx = 315.7132873535156
    cy = 199.8099365234375
    ```

根据你的环境和硬件设置，适当调整这些参数。

## 依赖项

脚本依赖以下库和工具：

- **OpenCV**：用于图像处理。
- **ROS2 (`rclpy`, `sensor_msgs`, `cv_bridge`)**：用于与 ROS2 主题通信。
- **NumPy**：用于数值运算。
- **PyTorch & Ultralytics YOLO**：用于目标检测。
- **Pygame**：用于处理摇杆输入。
- **PySerial**：用于串口通信。
- **Logging**：用于系统日志记录和调试。

确保所有依赖项已安装并与您的 Python 版本兼容。

## 文件结构

- **`2.py`**：包含自主系统的主要实现。
- **`1.py`**：辅助脚本（此处未提供详细信息）。如果 `2.py` 依赖于此文件，请确保其位于相同目录中。
- **`README.md`**：本文档文件。

## 贡献指南

欢迎贡献！请按照以下步骤进行：

1. **Fork 仓库**

2. **创建功能分支**

    ```bash
    git checkout -b feature/YourFeature
    ```

3. **提交更改**

    ```bash
    git commit -m "添加您的功能"
    ```

4. **推送到分支**

    ```bash
    git push origin feature/YourFeature
    ```

5. **创建 Pull Request**

    提供清晰的更改描述及其背后的原因。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

*如有任何问题或建议，请在仓库中提交 issue 或直接联系维护者。*
