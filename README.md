# 自主目标检测与控制系统

## 系统环境

- **操作系统**: Ubuntu 20.04
- **ROS版本**: ROS2 Foxy
- **Python版本**: Python 3.8+
- **深度相机**: Orbbec Gemini相机

## 概述

`automatic.py`是一个基于ROS2 Foxy的自主目标检测与控制系统,主要用于实时物体检测和机器人控制。系统使用Orbbec深度相机获取RGB-D数据,通过YOLO模型进行目标检测,并结合深度信息实现物体的精确定位与抓取。

## 功能特性

- **深度相机集成**: 
  - 订阅`/camera/color/image_raw`获取RGB图像
  - 订阅`/camera/depth/image_raw`获取深度图像
  - 支持RGB-D点云数据处理
- **目标检测与跟踪**:
  - 使用YOLO模型进行实时目标检测
  - 支持多种颜色物体的识别(红、黄、黑等)
  - 结合深度信息计算目标距离
- **控制系统**:
  - 支持手动/自动模式切换
  - 通过串口与机器人控制器通信
  - 支持预设任务序列执行

## 安装步骤

### 1. ROS2 Foxy安装

```bash
# 设置语言环境
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 添加ROS2软件源
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 安装ROS2 Foxy
sudo apt update
sudo apt install ros-foxy-desktop python3-argcomplete
sudo apt install python3-colcon-common-extensions

# 配置环境
echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. 深度相机配置

```bash
# 安装相机依赖
sudo apt install libgflags-dev nlohmann-json3-dev libgoogle-glog-dev \
    ros-foxy-image-transport ros-foxy-image-publisher

# 安装libuvc库
git clone https://github.com/libuvc/libuvc.git
cd libuvc
mkdir build && cd build
cmake .. && make -j4
sudo make install
sudo ldconfig

# 创建相机工作空间
mkdir -p ~/orbbec_ws/src
cd ~/orbbec_ws/src
# 复制Orbbec相机ROS2功能包到src目录

# 编译
cd ~/orbbec_ws
colcon build
echo "source ~/orbbec_ws/install/setup.bash" >> ~/.bashrc

# 安装udev规则
cd ~/orbbec_ws/src/OrbbecSDK_ROS/orbbec_camera/scripts/
sudo sh install_udev_rules.sh
```

### 3. 项目依赖安装

```bash
# 创建项目工作空间
mkdir -p ~/robot_ws/src
cd ~/robot_ws/src
# 复制项目代码到src目录

# 安装Python依赖
pip install opencv-python numpy torch ultralytics pygame pyserial

# 编译工作空间
cd ~/robot_ws
colcon build
source install/setup.bash
```

## 使用说明

### 1. 启动深度相机

```bash
# 启动相机
ros2 launch astra_camera gemini.launch.xml

# 验证相机话题
ros2 topic list
ros2 topic echo /camera/color/image_raw  # 查看彩色图像数据
ros2 topic echo /camera/depth/image_raw  # 查看深度图像数据
```

### 2. 运行主程序

```bash
# 确保在robot_ws工作空间下
cd ~/robot_ws
source install/setup.bash

# 运行程序
python src/your_package/scripts/automatic.py
```

### 3. 操作说明

- **自动/手动模式切换**:
  - 左肩按钮(LB): 切换到手动模式
  - 右肩按钮(RB): 切换到自动模式

- **手动控制**:
  - 左摇杆: 控制移动方向
  - A/B/X/Y按钮: 执行预设动作

- **自动模式下的任务状态**:
  - SEARCH_BALL: 搜索目标球体
  - TASK_TWO: 寻找放置区域

## 配置参数说明

### 相机参数
```python
# 相机内参
fx = 476.2635192871094
fy = 476.2635192871094
cx = 315.7132873535156
cy = 199.8099365234375
```

### 任务参数
```python
# 搜索颜色优先级
SEARCH_COLOR = 'red'
COLOR_PRIORITY = [SEARCH_COLOR, 'yellow', 'black']

# 距离阈值(米)
APPROACH_THRESHOLD = 0.25
EXIT_THRESHOLD = 0.3
```

### 动作时序参数
```python
# 任务一时序(秒)
TASK1_STOP_BEFORE_FORWARD = 0.5
TASK1_FORWARD_TO_GRAB = 0.8
TASK1_STOP_AFTER_GRAB = 0.5

# 任务二时序(秒)
TASK2_STOP_BEFORE_FORWARD = 0.5
TASK2_FORWARD_TO_PLACE = 0.9
TASK2_STOP_AFTER_PLACE = 0.5
```

## 故障排除

1. **相机连接问题**
   ```bash
   # 检查相机设备
   ll /dev/gemini
   
   # 检查相机话题
   ros2 topic list | grep camera
   ```

2. **串口通信问题**
   ```bash
   # 检查串口权限
   sudo chmod 666 /dev/ttyUSB0
   
   # 验证串口通信
   python -m serial.tools.miniterm /dev/ttyUSB0 115200
   ```

## 文件结构

```
robot_ws/
├── src/
│   ├── 2.py          # 主程序
│   ├── 1.py          # 辅助程序
│   └── README.md     # 说明文档
└── install/
    └── setup.bash    # 环境配置脚本
```

## 注意事项

1. 确保深度相机已正确连接并且能够正常发布图像话题
2. 运行程序前检查串口权限和连接状态
3. 确保YOLO模型文件(train14.pt)位于正确路径下
4. 程序运行时需要有足够的计算资源支持实时图像处理

## 许可证

本项目采用 MIT 许可证

## 技术支持

如有问题,请通过以下方式获取帮助:
1. 提交GitHub Issue
2. 查看ROS2 Foxy官方文档
3. 参考Orbbec相机SDK文档

---

*更多详细信息请参考代码注释和ROS2官方文档*
