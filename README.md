# MuJoCo Shared Control：Franka Pick-and-Place

这是一个独立的 MuJoCo/Gymnasium 仿真环境，包含 Franka Panda 机械臂、平行夹爪、
可自由抓取的立方体、固定放置目标区、工作台和具名 RGB-D 相机。该项目刻意不导入
RSS2023 的 Diffusion 代码。

## 机器人模型来源

机器人使用 Google DeepMind MuJoCo Menagerie 中的
`franka_emika_panda/mjx_panda.xml`，固定在提交
`c1a4eeb85694ae1dffe33ff1797d4e528928a133`。上游署名和许可证文件保留在
`src/mujoco_shared_control/assets/menagerie/`。

## 安装

```bash
cd mujoco_shared_control
uv sync --extra dev
```

本项目只支持使用 NVIDIA EGL 启动仿真。完整启动步骤见
下面的「快速启动」。

## 快速启动：GPU 仿真、VR 和同步采集

启动前先停止之前运行的仿真、VR 和采集节点，避免同名 ROS 节点和重复话题。
以下命令都假设项目位于：

```text
/home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
```

### 第一步：构建环境

首次运行或代码更新后执行一次：

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
uv sync --extra dev
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
cd ..
```

### 第二步：终端 1 启动 GPU 仿真和可视化窗口

打开第一个终端，执行：

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

nvidia-smi
ls -l /dev/nvidiactl /dev/nvidia0 /dev/nvidia-uvm

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

ros2 launch mujoco_pick_place_ros pick_place.launch.py \
  update_rate_hz:=20 \
  camera_name:=front \
  camera_width:=640 \
  camera_height:=480 \
  mujoco_gl:=egl \
  publish_raw_images:=false \
  publish_compressed:=true \
  show_mujoco_viewer:=true \
  show_camera_views:=true \
  dataset_dir:=$(pwd)/datasets \
  task_name:=pick_place
```

启动后应出现两类窗口：

- MuJoCo 机械臂场景窗口；
- `front` 相机的 RGB 和深度图像窗口。

### 第三步：终端 2 启动 VR 控制

保持仿真运行，打开第二个终端，执行：

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export PYTHONPATH="/home/ray/miniconda3/envs/xrt/lib/python3.10/site-packages:${PYTHONPATH}"
python -m mujoco_pick_place_ros.vr_teleop_node
```

VR 默认同时控制末端平移和旋转。按住右手 `grip` 键后手柄与当前机械臂
末端对齐，持续按住时才控制末端；右手 trigger 独立控制夹爪开合。
这里启动的是默认 `direct` 模式，VR 会直接发布机器人控制指令。

如需使用可插拔共享控制，把上述 VR 命令的最后一行改为：

```bash
python -m mujoco_pick_place_ros.vr_teleop_node --ros-args \
  -p control_mode:=shared
```

然后另外打开一个终端，启动共享控制推理节点：

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch mujoco_pick_place_ros shared_control.launch.py \
  policy_plugin:=human_passthrough \
  task_name:=pick_place \
  inference_timeout_ms:=250.0
```

`human_passthrough` 是验证接口用的直通策略。它仍经过统一模型接口和 ROS 输出
适配器，但输出保持为人类的末端目标，因此可以先用它验证共享控制链路。
共享模式下采集控制台作为第四个终端启动；按 `t` 修改任务名时，策略会同步更新
任务名并调用一次 `reset()`。

### 第四步：终端 3 启动采集控制台

保持前两个终端运行，打开第三个终端，执行：

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 run mujoco_pick_place_ros collection_control_node
```

采集控制台使用即时快捷键，无需按 Enter：

```text
t  设置任务名称
s  开始采集当前轨迹
f  保存当前轨迹并 reset
d  丢弃当前轨迹并 reset
q  完成任务并退出采集控制台
h  显示帮助
```

推荐操作顺序：

1. 按 `t` 输入任务名称。
2. 按 `s` 开始当前 Episode。
3. 使用 VR 完成一次抓取放置。
4. 轨迹需要保留时按 `f`；轨迹无效时按 `d`。
5. 继续按 `s` 采集下一条，全部完成后按 `q`。

正在采集时按 `q` 不会直接退出，需要先按 `f` 保存或按 `d` 丢弃。

### 第五步：检查落盘数据

数据默认保存在：

```text
mujoco_shared_control/datasets/<任务名>/episode_*.h5
```

校验某个任务的所有 Episode：

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
uv run --offline python scripts/validate_dataset.py datasets/<任务名>
```

只有时间戳、20 Hz 间隔、RGB、深度、observation 和 action 全部通过校验的 Episode
才会保留在正式目录；失败的 Episode 会移入 `invalid/`。

### 第六步：回放并可视化 HDF5

自动打开 `datasets/` 下最新保存的 Episode：

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
./.venv/bin/python scripts/play_hdf5.py
```

也可以指定一个 HDF5 文件或任务目录：

```bash
./.venv/bin/python scripts/play_hdf5.py datasets/pick_box/episode_xxx.h5
./.venv/bin/python scripts/play_hdf5.py datasets/pick_box
```

回放窗口同步显示 RGB、米制深度图、26 维状态、完整 observation、VR 原始输入、
用户命令、实际执行 action、阶段、事件、图像有效性和同步误差。快捷键如下：

```text
空格       播放或暂停
左/右方向键 逐帧后退或前进
上/下方向键 提高或降低播放速度
Home/End   跳到第一帧或最后一帧
Q/Esc      退出
```

## 动作接口

`step(action)` 接收形状为 `(8,)` 的绝对目标 `float64` 向量：

```text
[joint1, joint2, joint3, joint4, joint5, joint6, joint7, gripper_opening]
```

前七项为弧度制机械臂关节目标，受 Menagerie 模型关节限位约束。最后一项为两指之间的
总开度，单位为米：`0.0` 表示闭合，`0.08` 表示完全张开。也可直接使用以下底层接口：

| 动作索引 | 关节 | 限位 | 执行器 |
|---|---|---:|---|
| 0 | `joint1` | `[-2.8973, 2.8973] rad` | `actuator1` |
| 1 | `joint2` | `[-1.7628, 1.7628] rad` | `actuator2` |
| 2 | `joint3` | `[-2.8973, 2.8973] rad` | `actuator3` |
| 3 | `joint4` | `[-3.0718, -0.0698] rad` | `actuator4` |
| 4 | `joint5` | `[-2.8973, 2.8973] rad` | `actuator5` |
| 5 | `joint6` | `[-0.0175, 3.7525] rad` | `actuator6` |
| 6 | `joint7` | `[-2.8973, 2.8973] rad` | `actuator7` |
| 7 | `finger_joint1` + `finger_joint2` | 总开度 `[0, 0.08] m` | `actuator8` |

夹爪有两个物理滑动关节，每个关节限位为 `[0, 0.04] m`。Menagerie 的 equality
约束将 `finger_joint1` 与 `finger_joint2` 镜像绑定；`actuator8` 只控制其中一个手指，
因此环境会将外部的总开度命令除以二，转换为单指目标。

```python
env.set_joint_position_target(q_cmd)  # (7,)，单位：弧度
env.set_gripper_command(g_cmd)        # 标量，两指总开度，单位：米
env.forward_kinematics(q_cmd)         # (7,) -> (4, 4)，Pinocchio FK
env.set_ee_target(T_ee_cmd)           # (4, 4) world 齐次变换，经 Pinocchio IK 控制
```

`T_ee_cmd` 必须是有限值的齐次变换矩阵，其旋转部分必须为正交矩阵。使用下面命令检查
FK、IK 和实际执行后的末端运动：

```bash
uv run python scripts/test_kinematics.py
```

## 观测接口

`reset()` 和 `step()` 都返回结构化观测字典，包含：

- `q_obs`、`dq_obs`：机械臂关节位置和速度，形状均为 `(7,)`
- `ee_pose`、`object_pose`、`goal_pose`：齐次变换矩阵，形状均为 `(4, 4)`
- `gripper`：形状为 `(1,)` 的两指总开度，单位为米
- 物体线速度、角速度：形状均为 `(3,)`
- 左右手指与物体的接触标志、法向接触力和接触数量
- `object_grasped`：双侧手指接触的抓取启发式状态
- `timestamp`：仿真时间，单位为秒

`env.get_observation()` 返回相同结构。独立且可替换的
`env.get_policy_observation()` 适配器当前返回 `(42,)` 的 `float32` 向量；环境物理
不依赖这一策略输入表示。

默认 `reset()` 会在保守的桌面工作区内同时随机化盒子和目标盘，并保证两者中心至少相距
`0.16 m`：盒子的 `(x, y)` 范围为 `[0.46, 0.54] × [-0.06, 0.06] m`，目标盘范围为
`[0.50, 0.60] × [-0.27, -0.17] m`。这些范围分别围绕原始盒子和目标盘位置小幅变化，
不会采样到桌边或机械臂工作区的极端位置。使用相同 seed 可复现相同位置。需要固定场景或指定位置时：

```python
obs, info = env.reset(
    seed=0,
    options={
        "randomize_object": False,
        "randomize_goal": False,
        "object_xy": [0.50, 0.0],
        "goal_xy": [0.55, -0.22],
    },
)
```

实际采样位置同时通过 `info["object_xy"]` 和 `info["goal_xy"]` 返回。

详细接触点和 geom 名称可通过
`env.observation_reader.get_contact_details()` 获得。

## 相机接口

固定相机名称为 `front`，默认分辨率为 `640x480`，垂直视场角为 45 度。RGB、米制深度
和计算得到的相机内参/外参是独立接口：

```python
rgb = env.render_rgb("front")
depth = env.render_depth("front")
calibration = env.get_camera_calibration("front")
```

以后可以按名称增加场景相机或腕部相机，无需修改相机 API。

## 使用 NVIDIA GPU 启动仿真（ROS 2 Humble）

ROS 集成是位于 `ros2_ws/src/mujoco_pick_place_ros` 的独立 `ament_python` 包。MuJoCo
使用项目虚拟环境；`rclpy` 与 ROS 消息使用系统 ROS 2 Humble 安装。

本项目唯一支持的仿真方式是：使用 NVIDIA EGL 在 GPU 上运行权威仿真和
RGB-D 渲染，并通过独立进程显示 MuJoCo、RGB 和深度窗口。完整且唯一的
启动命令见「快速启动：GPU 仿真、VR 和同步采集」。

`MUJOCO_EGL_DEVICE_ID=0` 表示使用第 0 块 GPU。
`__EGL_VENDOR_LIBRARY_FILENAMES` 会强制 GLVND 加载 NVIDIA EGL，避免在系统
同时安装 Mesa 时静默落到 `llvmpipe` CPU 软件渲染。如果
`nvidia-smi` 失败或 `/dev/nvidia*` 不存在，应先修复驱动、容器 GPU
映射或当前会话的设备权限，不要启动采集。

`show_mujoco_viewer:=true` 打开可交互的 MuJoCo 场景窗口；
`show_camera_views:=true` 打开并排的 RGB 和深度图像窗口。权威仿真仍然使用
NVIDIA EGL GPU 渲染，场景查看器只在独立进程中镜像 ROS 状态，不改变
20 Hz 仿真和数据发布时序。

ROS launch 默认注册 `640x480` RGB-D。固定 `front` 相机的 world 坐标位置为
`(1.25, 0.0, 0.80) m`，位于机械臂正前方并朝向工作台中心线。状态/控制
频率可在 10 Hz 与 20 Hz 之间切换；数据采集统一使用 20 Hz。GPU RGB-D
渲染、压缩和 ROS 发布在工作线程中运行，不通过阻塞主控制循环来降低频率。

在当前 NVIDIA GeForce RTX 4060 Laptop GPU（驱动 `595.71.05`）上，已测得注册的
`640x480` 压缩 RGB、深度流和关节状态流均可稳定达到约 `20.0 Hz`。
项目自带的串行性能测试结果为：RGB-D 渲染约 `2.04 ms`、RGB JPEG 编码约
`0.70 ms`、深度 PNG 编码约 `3.98 ms`，理论串行流水线约 `149 Hz`。

启动正式采集前，必须在同一终端环境中运行性能测试：

```bash
uv run --offline python scripts/benchmark_camera.py
```

`rgbd_render_ms` 应明显小于 `50 ms`。如果接近 `100 ms`或只有约 `9 Hz`，
通常表示已落到 Mesa `llvmpipe`，该环境不应用于数据采集。

仿真启动后，在其他已加载 ROS 2 环境的终端中分别验证 RGB、深度和状态频率：

```bash
ros2 topic hz /mujoco/camera/front/color/image_raw/compressed
ros2 topic hz /mujoco/camera/front/depth/image_raw/compressedDepth
ros2 topic hz /mujoco/joint_states
```

三者都应接近 `20 Hz`。未达到目标时应终止当前采集并检查 GPU/EGL 环境。

ROS launch 默认启用盒子和目标盘随机化。MuJoCo 原生查看器只是 ROS 状态镜像，因此其
`Reset` 按钮和 `R`/Backspace 键会向 `/mujoco/reset` 发布请求，由 bridge 重置权威仿真状态；
VR 节点收到同一请求后会清除旧的手柄对齐，等待新的末端状态再重新对齐。
如需固定场景，在仿真启动后修改以下参数；新设置从下一次 Reset 开始生效：

```bash
ros2 param set /mujoco/pick_place_bridge randomize_object false
ros2 param set /mujoco/pick_place_bridge randomize_goal false
```

### 控制话题

除 `/clock` 外，所有话题均位于 `/mujoco` 命名空间下。

| 话题 | 类型 | 约定 |
|---|---|---|
| `joint_position_command` | `sensor_msgs/msg/JointState` | 七个机械臂绝对关节目标，单位为弧度。可提供 `joint1` 到 `joint7` 名称，或省略名称并使用标准顺序。 |
| `ee_pose_command` | `geometry_msgs/msg/PoseStamped` | world 坐标系下的末端目标位姿。Pinocchio 将其求解为七关节目标；不可达位姿会被拒绝。 |
| `gripper_command` | `std_msgs/msg/Float64` | 两指总开度，单位为米，截断到 `[0.0, 0.08]`。 |
| `update_rate_command` | `std_msgs/msg/UInt8` | 运行时频率切换，只接受 `10` 和 `20`。 |
| `reset` | `std_msgs/msg/Empty` | 重置仿真、机械臂命令和夹爪命令。 |
| `vr/input_raw` | `sensor_msgs/msg/Joy` | 带时间戳的原始 VR 位姿、trigger、grip 和对齐状态。 |
| `collection/command` | `std_msgs/msg/String` | 采集控制：`task`、`start`、`save`、`discard`、`finish`。 |
| `collection/status` | `std_msgs/msg/String` | 采集器当前状态和落盘结果。 |

### 控制示例

以下命令都要求桥接节点正在运行，且已加载 ROS 2 环境：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
```

#### 关节空间机械臂控制

发布七个绝对关节目标，单位为弧度。推荐使用具名形式，因为关节映射明确；关节名称必须为
`joint1` 到 `joint7`。

```bash
ros2 topic pub --once /mujoco/joint_position_command sensor_msgs/msg/JointState \
  "{name: [joint1, joint2, joint3, joint4, joint5, joint6, joint7], position: [0.30, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398]}"
```

也可以省略 `name` 字段；此时桥接节点按标准 `joint1` 到 `joint7` 顺序解释数值：

```bash
ros2 topic pub --once /mujoco/joint_position_command sensor_msgs/msg/JointState \
  "{position: [0.30, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398]}"
```

#### 夹爪控制

命令为两根手指之间的总距离，单位为米。`0.0` 为闭合，`0.08` 为完全张开：

```bash
# 闭合两根手指。
ros2 topic pub --once /mujoco/gripper_command std_msgs/msg/Float64 "{data: 0.0}"

# 将两指总开度调整为 6 cm。
ros2 topic pub --once /mujoco/gripper_command std_msgs/msg/Float64 "{data: 0.06}"
```

#### 笛卡尔末端控制

发布 `world` 坐标系下的目标位姿。桥接节点会验证四元数，并用 Pinocchio IK 求解七关节
目标；不可达目标会被拒绝。ROS 四元数顺序为 `x, y, z, w`：

```bash
# 保持 home 姿态，将夹爪向上移动约 2 cm。
ros2 topic pub --once /mujoco/ee_pose_command geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: world}, pose: {position: {x: 0.30689, y: 0.0, z: 0.51028}, orientation: {x: 1.0, y: 0.0, z: 0.0, w: 0.0}}}"
```

可以先读取当前末端位姿，将其作为下一条笛卡尔命令的起点：

```bash
ros2 topic echo /mujoco/ee_pose geometry_msgs/msg/PoseStamped --once
```

更新频率模式可独立于控制接口切换：

```bash
ros2 topic pub --once /mujoco/update_rate_command std_msgs/msg/UInt8 "{data: 10}"
ros2 topic pub --once /mujoco/update_rate_command std_msgs/msg/UInt8 "{data: 20}"
```

Pinocchio 模型由 `example-robot-data` 中的 Panda URDF 构建。固定标定坐标系将
`panda_hand_tcp` 对齐到 MuJoCo `gripper` site，因此 FK、IK、`/mujoco/ee_pose` 和
MuJoCo 使用相同的末端定义。

### 坐标系

`/mujoco/ee_pose`、`/mujoco/object/pose` 和 `/mujoco/goal/pose` 都是表示在同一个
MuJoCo `world` 坐标系下的 `geometry_msgs/msg/PoseStamped` 消息。
`/mujoco/object/twist` 也表示在该坐标系下。world 为右手坐标系，单位为米，`+Z` 向上，
重力为 `(0, 0, -9.81)`。`/mujoco/ee_pose_command` 只接受 `world` 坐标系下的位姿
（或空的 `frame_id`，空值会按 `world` 处理）。

RGB、深度、压缩图像和 `CameraInfo` 消息使用独立的
`front_camera_optical_frame`。它在 `world` 中的固定原点为 `(1.25, 0.0, 0.80) m`；
图像坐标遵循 ROS optical 约定：`+x` 向右、`+y` 向下、`+z` 沿相机前方。相机内参和
MuJoCo 相机外参可通过 `env.get_camera_calibration("front")` 获取。

桥接节点当前尚未发布 `/tf` 或 `/tf_static`，因此 ROS `tf2` 还不能在 `world` 与
`front_camera_optical_frame` 之间变换。状态 pose 话题彼此一致，但深度像素反投影或
world/图像融合需要后续添加静态
`world -> front_camera_optical_frame` 变换。该变换需要使用 `diag(1, -1, -1)` 将
MuJoCo 相机轴转换为 ROS optical 相机轴。

也可以通过节点参数修改频率：

```bash
ros2 param set /mujoco/pick_place_bridge update_rate_hz 20
```

### 状态与相机话题

| 话题 | 类型 | 内容 |
|---|---|---|
| `joint_states` | `sensor_msgs/msg/JointState` | 七个机械臂关节和两个物理手指关节的位置/速度。 |
| `gripper/state` | `sensor_msgs/msg/JointState` | 两个手指关节的位置/速度。 |
| `gripper/opening` | `std_msgs/msg/Float64` | 两指总开度，单位为米。 |
| `ee_pose`、`object/pose`、`goal/pose` | `geometry_msgs/msg/PoseStamped` | world 坐标系下的位姿。 |
| `object/twist` | `geometry_msgs/msg/TwistStamped` | world 坐标系下的物体速度。 |
| `object/grasped` | `std_msgs/msg/Bool` | 当前的双侧接触抓取启发式状态。 |
| `active_update_rate` | `std_msgs/msg/UInt8` | 当前 10/20 Hz 模式。 |
| `/clock` | `rosgraph_msgs/msg/Clock` | MuJoCo 仿真时间。 |
| `camera/front/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | JPEG 压缩且已注册的 RGB 图像。 |
| `camera/front/depth/image_raw/compressedDepth` | `sensor_msgs/msg/CompressedImage` | 无损 PNG 压缩且已注册的深度图，`16UC1` 毫米。 |
| `camera/front/{color,depth}/camera_info` | `sensor_msgs/msg/CameraInfo` | 零畸变针孔相机标定。 |

默认在 `640x480` 下仅发布压缩图像：

- `publish_raw_images:=false`
- `publish_compressed:=true`

当还需要未压缩的 `rgb8` 与米制 `32FC1` 数据流时，设置
`publish_raw_images:=true`。标准压缩深度负载以 `compressedDepth` transport header
开头，解码后得到单位为毫米的 `16UC1`；使用者必须除以 1000 才能得到米。

压缩在工作线程中进行，以保证原始状态和 RGB-D 发布维持选定更新频率。当前相机是理想的、
已注册的 MuJoCo 针孔 RGB-D 相机，没有镜头畸变、传感器噪声、rolling shutter 或缺失深度
模型。后续可增加匹配硬件的配置，以模拟 RealSense D435i/D455 或其他选定传感器，而无需
改变话题名称。

## 验证

```bash
uv run python scripts/test_env.py
uv run python scripts/test_joint_control.py
uv run python scripts/test_camera.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

相机测试产物写入 `outputs/front_rgb.png`、`outputs/front_depth.npy` 和
`outputs/front_depth.png`。

## 当前明确未实现的部分

当前阶段尚未实现在线 Diffusion 辅助、学习型阶段分类器以及
failure injection/recovery 状态机。HDF5 同步示教采集、规则阶段标签、数据校验和
RSS2023 离线训练已实现。

## 使用 HDF5 示教训练 RSS2023 Diffusion 模型

训练模块使用每帧完整低维观测：七关节位置、末端位姿、夹爪开度、物体位姿和目标位姿，
共 29 维；专家动作使用 VR 的末端位姿和夹爪命令，共 8 维。模型因此在 29 维状态硬条件下
学习 8 维笛卡尔动作扩散，总输入维度为 37。首版训练不读取 RGB-D。

安装训练依赖：

```bash
cd /home/ray/diffusion-for-shared-autonomy/mujoco_shared_control
uv sync --extra dev --extra rss2023
```

使用 `datasets/pick_box` 训练：

```bash
./.venv/bin/python scripts/train_rss2023.py \
  --dataset-dir datasets/pick_box \
  --output-dir outputs/rss2023/pick_box \
  --steps 30000 \
  --batch-size 512 \
  --device cuda
```

训练器按 Episode 固定随机划分为 80%/10%/10%，将每条轨迹截断到第一次任务成功，过滤
无效用户命令，并只用训练集计算逐维归一化统计量。四元数按 Episode 消除正负二义性。
`outputs/rss2023/pick_box/` 中会生成：

```text
dataset_manifest.json  数据划分、字段和帧数
best.pt                验证损失最低的 checkpoint
step_XXXXXXXX.pt       定期 checkpoint
final.pt               最终 checkpoint
```

checkpoint 包含模型、EMA、优化器、扩散配置、数据划分以及观测/动作归一化参数。实时 ROS
推理插件仍未实现；当前训练产物可用于下面的离线动作评估和 MuJoCo 闭环回放评估。

训练后的 NumPy 推理接口位于 `mujoco_shared_control.rss2023.inference`：

```python
from mujoco_shared_control.rss2023.inference import RSS2023Predictor

predictor = RSS2023Predictor.from_checkpoint(
    "outputs/rss2023/pick_box/best.pt",
    device_name="cuda",
)
assisted_action_8 = predictor.predict(
    observation_29,
    human_action_8,
    gamma=0.04,
)
```

输入依次为当前 29 维观测与用户 8 维笛卡尔命令，输出为辅助后的
`[x, y, z, qw, qx, qy, qz, gripper]`。接口默认加载与验证损失对应的原始模型参数、应用
checkpoint 中的归一化统计量，并对输出四元数和夹爪范围做最终处理；需要比较 EMA 时可向
`from_checkpoint` 传入 `use_ema=True`。

### 第一阶段：离线动作修正评估

该阶段在 checkpoint 固定的数据划分上构造 RSS2023 风格的代理用户：`noisy` 以概率
`p` 将示教动作替换为训练集动作池中的随机动作，`laggy` 以概率 `p` 重复上一条命令。
这里使用示教动作池，是因为本项目输出绝对笛卡尔位姿，直接在无界空间均匀采样没有物理意义。

先在验证集扫描扩散强度：

```bash
./.venv/bin/python scripts/eval_rss2023_offline.py \
  --checkpoint outputs/rss2023/pick_box/best.pt \
  --output-dir outputs/rss2023/pick_box/offline_validation_raw \
  --split validation --pilots clean noisy laggy \
  --probabilities 0.3 0.6 \
  --gammas 0.0 0.02 0.04 0.06 0.08 0.1 0.2 0.4 0.6 0.8 1.0 \
  --seeds 0 1 2 3 4 --weights model --device cpu
```

再在测试集报告固定候选值：

```bash
./.venv/bin/python scripts/eval_rss2023_offline.py \
  --checkpoint outputs/rss2023/pick_box/best.pt \
  --output-dir outputs/rss2023/pick_box/offline_test_raw \
  --split test --pilots clean noisy laggy --probabilities 0.3 0.6 \
  --gammas 0.0 0.04 1.0 --seeds 0 1 2 3 4 5 6 7 8 9 \
  --weights model --device cpu
```

输出的 `summary.csv` 包含位置、姿态、夹爪和归一化动作误差。`offline_best_gamma`
只表示代理动作误差最小，不能作为部署参数；它必须通过第二阶段闭环成功率和干净动作安全性。

### 第二阶段：MuJoCo 闭环回放评估

该阶段从每个留出 Episode 的物体/目标初始位置重置环境，以 20 Hz 重放示教笛卡尔命令，
每条命令执行 5 个 0.01 s 物理步。模型每帧读取当前仿真状态并输出修正位姿，再通过
Pinocchio IK 执行。它是当前数据可完成的闭环代理实验，不等同于论文中的 SAC 代理用户实验。

```bash
./.venv/bin/python scripts/eval_rss2023_closed_loop.py \
  --checkpoint outputs/rss2023/pick_box/best.pt \
  --output-dir outputs/rss2023/pick_box/closed_loop_test_raw \
  --split test --pilots clean noisy laggy --probabilities 0.3 0.6 \
  --gammas 0.0 0.04 --seeds 0 1 2 3 4 5 6 7 8 9 \
  --weights model --device cpu
```

当前 `best.pt` 的测试结果表明模型尚不适合接入实时控制：干净代理用户在 `gamma=0`
时成功率为 100%，在 `gamma=0.04` 时降为 33.3%；延迟代理用户也下降。随机错误动作有
少量恢复，但不足以抵消对正常命令的破坏。应先增加覆盖更多初始状态的成功示教、重新训练，
并加入工作空间/IK 可达性保护，再重新通过两阶段测试。`gamma=0` 应作为当前安全默认行为。

## 启动 XRobotToolkit VR 遥操作

`mujoco_pick_place_ros.vr_teleop_node` 从 XRobotToolkit SDK 读取右手柄，并直接向已有的
MuJoCo 控制话题发布消息；不修改仿真桥接节点。节点订阅 `/mujoco/ee_pose` 作为初始末端位姿，
在右手 `grip` 按下时才记录当前手柄 pose 与末端 pose 并完成对齐。随后、且仅在持续按住
`grip` 时，手柄的相对平移和相对旋转会转换为
`/mujoco/ee_pose_command` 的 world-frame 目标。

XRT pose 按 `[x, y, z, qx, qy, qz, qw]` 解释。默认使用 XRoboToolkit 官方示例的
headset-to-world 轴映射：

```text
x_mujoco = -z_xrt
y_mujoco = -x_xrt
z_mujoco =  y_xrt
```

当前默认 `control_orientation:=true`，末端会同时跟随手柄的相对平移和相对旋转。
如果只需要三轴平移，可显式将 `control_orientation` 设为 `false`。若设备或场景
坐标系改变，可通过 `vr_to_world_axes` 参数传入一个正交的 3×3 矩阵覆盖默认值。

为抑制手柄追踪抖动和目标突变，节点会对位置与旋转分别平滑，并限制笛卡尔速度。默认参数为：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `update_rate_hz` | `50.0` | VR 输入与命令发布频率，单位 Hz。 |
| `translation_scale` | `1.0` | 手柄相对平移到机械臂末端平移的比例。 |
| `control_orientation` | `true` | 是否让末端跟随手柄相对旋转。 |
| `position_smoothing_time_constant` | `0.08` | 位置指数平滑时间常数，单位秒。越大越平滑，但延迟越明显。 |
| `orientation_smoothing_time_constant` | `0.10` | 旋转 SLERP 平滑时间常数，单位秒。 |
| `max_linear_speed` | `0.5` | 最大末端线速度，单位 m/s。 |
| `max_angular_speed` | `2.0` | 最大末端角速度，单位 rad/s。 |

仿真 bridge 只在选定的 10/20 Hz 仿真更新周期中求解最新一条末端目标；旧目标会被新目标覆盖。
IK 使用上一帧关节命令作为初值，以降低连续控制时的关节解跳变。

右手扳机值会被截断到 `[0, 1]`，并映射为两指总开度：

```text
gripper_opening = 0.08 * (1 - right_trigger)
```

所以扳机松开时夹爪全开（`0.08 m`），按到底时夹爪闭合（`0.0 m`）。SDK 读取失败或 pose 无效时，
节点停止发布末端目标。松开 `grip` 也会停止末端控制并清除对齐；下一次按下时，以当时的末端 pose
重新对齐。扳机始终独立于 `grip` 控制夹爪。默认按键阈值为 `0.5`，可通过 `grip_threshold` 参数调整。

先启动仿真桥接节点，再启动 VR 遥操节点。完整且唯一的 VR 启动命令见页首
「快速启动」的第三步。`xrobotoolkit_sdk` 由 `xrt` Conda 环境通过
`PYTHONPATH` 提供，其他 NumPy/MuJoCo 依赖仍使用项目虚拟环境。

该节点默认以 `50 Hz` 读取手柄并发布命令。如需调整平移比例、姿态跟随、
平滑时间常数或速度上限，使用上表中的 ROS 参数；不再使用另一套 VR 启动流程。

## 同步 RGB-D 示教采集

采集以 bridge 的 `20 Hz` 权威仿真边界为唯一时钟。每一个 HDF5 step 都是一条
完整的同步帧：

```text
D_t = (RGB_t, Depth_t, Observation_t, VR_t, UserCommand_t,
       ExecutedAction_t, Stage_t, Events_t)
```

`Observation_t`、RGB 和深度来自动作执行前的同一份 MuJoCo 场景快照；
`ExecutedAction_t` 是随后在 `[t, t+0.05s)` 内实际采用的七维关节目标和
一维夹爪目标。RGB 为 `uint8[480,640,3]` RGB 顺序，深度为
`float32[480,640]` 米制数据。两者通过同一次 `render_rgbd()` 获取。

26 维策略状态的顺序为：

```text
q(7), dq(7), ee_xyz(3), gripper_opening(1),
object_xyz(3), goal_xyz(3), object_grasped(1), object_goal_distance(1)
```

除 26 维状态外，HDF5 还保存完整观测、42 维现有策略观测、VR 原始输入、
处理后的末端/夹爪命令、IK 后的实际 action、MuJoCo `ctrl`、阶段、事件、时间戳、
相机标定和图像同步误差。

采集控制台的启动命令、即时快捷键和推荐操作顺序见页首「快速启动」的
第四步。

采集中的文件名为 `episode_*.inprogress.h5`。`save` 会等待所有已排队的
RGB-D 帧完成，校验通过后再改为正式 `.h5`；校验失败的 Episode 会移入
`invalid/`。数据默认保存在 launch 参数 `dataset_dir` 指定的目录。

离线校验单个 Episode 或整个数据集：

```bash
uv run --offline python scripts/validate_dataset.py datasets/pick_place
```

校验报告包含实际平均频率、平均/最大间隔、RGB/深度帧数、缺帧率、连续重复帧、
时间戳单调性和图像—状态同步误差。

## 可插拔共享控制接口

项目保留两套控制方式：

```text
direct: VR -> ee_pose_command/gripper_command -> MuJoCo bridge
shared: VR -> human_command -> SharedPolicy -> ROS适配器 -> MuJoCo bridge
```

`direct` 是默认模式，与原来的 VR 操作一致。`shared` 模式下 VR 不再直接向机器人
发送控制指令，只发布处理后的人类意图；`shared_control_node` 是机器人控制话题的
唯一发布者，避免直接控制和模型控制同时竞争。

### 策略插件协议

模型代码不需要导入 ROS，只需要实现 `input_spec`、`reset()` 和 `predict()`：

```python
import numpy as np

from mujoco_shared_control.shared_control import (
    CommandSpace,
    ModelInput,
    ModelInputSpec,
    ModelOutput,
)


class MySharedPolicy:
    input_spec = ModelInputSpec(
        history_length=8,
        use_state_26=True,
        use_human_action=True,
        use_executed_action=True,
        use_rgb=True,
        use_depth=True,
        image_history_length=2,
        cameras=("front",),
    )

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint

    def reset(self, task_name: str) -> None:
        pass

    def predict(self, model_input: ModelInput) -> ModelOutput:
        # 在这里调用自己的模型。下面仅演示返回接口。
        human_command = model_input.latest_human_action.astype(np.float64)
        return ModelOutput(
            timestamp=model_input.timestamp,
            command=human_command,
            command_space=CommandSpace.CARTESIAN_POSE,
            valid=True,
            control_active=bool(model_input.human_control_active[-1]),
            confidence=1.0,
            policy_name=type(self).__name__,
        )
```

策略通过 `ModelInputSpec` 自己声明所需数据。未启用的字段不会复制大数组；图像策略
还可以独立设置图像历史长度和相机列表。主要张量约定为：

```text
state_history           float32[T, 26]
human_action_history    float32[T, 8]   # xyz + quaternion_wxyz + gripper
executed_action_history float32[T, 8]   # joint1..joint7 + gripper
rgb[camera]             uint8[K, H, W, 3]
depth[camera]           float32[K, H, W]，单位为米
history_valid           bool[T]
human_action_valid      bool[T]
human_control_active    bool[T]
human_action_timestamps float64[T]
human_action_age_ms     float32[T]
```

策略输出支持两种命令空间：

```text
CommandSpace.CARTESIAN_POSE  xyz + quaternion_wxyz + gripper
CommandSpace.JOINT_POSITION  joint1..joint7 + gripper
```

使用外部插件时传入 `模块路径:类名` 和构造参数 JSON：

```bash
ros2 launch mujoco_pick_place_ros shared_control.launch.py \
  policy_plugin:=my_package.my_policy:MySharedPolicy \
  policy_config_json:='{"checkpoint":"checkpoints/model.pt"}'
```

同步输入由 bridge 在同一20 Hz仿真边界发布。状态、实际执行动作、RGB和深度使用
相同时间戳；节点按时间戳组帧后再提交后台推理，不阻塞仿真控制循环。

### 错误和超时行为

模型抛出异常、返回无效结果、产生 NaN、夹爪越界或超过 `inference_timeout_ms` 时：

- 终端输出醒目的 `SHARED CONTROL ERROR`；
- `/mujoco/shared_control/status` 发布带原因和帧编号的 JSON 错误状态；
- 不发布错误结果，并保持机器人最后一次有效指令；
- 不会静默切换回 VR 直接控制。

有效策略输出同时发布到 `/mujoco/shared_control/policy_output`，并写入 HDF5 的
`actions/policy_output`。数据中仍分别保留 `actions/user_command` 和
`actions/executed`，可以复现人类输入、策略输出和机器人实际执行的完整链路。
# Rule Expert state-only collection

The existing ROS teleoperation recorder remains schema 1.1 RGB-D data.  The
automatic path writes transition-aligned schema 2.0 episodes without creating
camera datasets.  Its canonical physical policy action is
`[dx, dy, dz, drx, dry, drz, gripper]`; the IK joint target and MuJoCo actuator
control are recorded separately.

Run a small smoke campaign from this directory:

```bash
.venv/bin/python scripts/collect_rule_expert.py \
  --output datasets/pick_box/expert_rule \
  --nominal-success-target 2 \
  --perturbed-episodes 2 \
  --seed 1000 \
  --max-attempts 20
```

The campaign keeps every failure, targets the requested number of clean
successes, and executes exactly the requested number of perturbed episodes.
Files are first written with an `.inprogress.h5` suffix and are atomically moved
to `success/`, `recovered/`, `failure/`, or `invalid/` after validation.

The formal 1300-episode collection is selected only through
`manifests/rule_expert_v1_formal.json`. Rebuild it with:

```bash
.venv/bin/python scripts/build_rule_expert_manifest.py
```

Training code must use `ManifestActorDataset` or `ManifestCriticDataset` rather
than recursively discovering HDF5 files. Both loaders accept `train` or
`validation` and verify the manifest and per-file SHA-256 checksums by default.
