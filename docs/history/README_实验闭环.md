> 历史说明：当前交接入口为 [README.md](README.md)，状态以 [STATUS](docs/STATUS.md) 为准。以下早期方案保留供追溯。

# Dummy 最小实验闭环：Ubuntu → 仿真 → 本地推理 → 真机

## Windows 实机上位机（2026-09-16 新增）

双击根目录 **`MuJoCo实机同步.cmd`**。启动读取当前实机角度，点击“开启实时跟随”后用关节滑块控制实体；停止按钮或空格停止跟随。左右分别显示反馈姿态与目标预览。最终版以原生 USB 读反馈、串口发送目标，已通过窗口完成 J1 小范围往返和停止验证。六轴外观映射与其他轴仍需校验，不能据此认为任意路径均已验证。

完整操作与范围见 [MuJoCo实机同步使用说明](docs/history/MuJoCo实机同步使用说明.md)。以下早期 Ubuntu/训练方案保留，状态以近期实机诊断记录为准。

这份工程把“只能用 DummyStudio 手动控制”拆成可逐步替换的软件链路。第一版选择 **Python Robot API + USB + MuJoCo**。ROS 2/MoveIt、D435f 示教、ACT/VLA 在同一接口上逐步加入，不要求先把所有框架都安装好。

**当前已跑通的是参考模型上的软件闭环。Ubuntu 真机连接与运动尚未验证；不是已经完成整机部署。**

## 1. 先理解每个软件负责什么

```text
                     策略模型（本地推理）
                读当前状态 → 预测下一动作
                              ↓
                动作检查 / 时效 / 执行调度
                              ↓
                  统一 Robot API（rad）
               ┌──────────────┴──────────────┐
          SimRobot                       SerialRobot
     MuJoCo 仿真积分                USB 文本协议（degree）
               ↓                            ↓
        仿真关节反馈                 REF → CAN → 六轴
               └──────────────┬──────────────┘
                         状态与日志
```

DummyStudio 是上位机，不是必须保留的算法执行器。本地文件已确认它使用 VID 0x1209、115200；V3.0.7 手册要求 CRLF。独立 Python 程序可以实现相同通信入口。但具体响应、动作模式和版本仍要在实机核验。

MuJoCo 是参考仿真器：它可以在没有机械臂时运行模型与策略。RViz 是可视化工具；MoveIt 是规划工具；它们并不能自动替代硬件驱动或自动得到与真机一致的模型。

## 2. 现在提供的功能

| 模块/命令 | 状态与用途 |
|---|---|
| `ports` | 枚举串口，显示 VID/PID；不连接、不运动 |
| `probe` | 显式端口，只发 GETJPOS，记录原始 RX/TX，严格解析六轴反馈 |
| `jog` | 现场监督的单轴小步入口；默认配置阻止运动；需要实际标定和停止验证 |
| `models/dummy_reference.xml` | 来自本地 auk URDF 的六轴参考模型，保留上游许可和来源哈希 |
| `collect-sim` | 生成参考仿真关节到点示教数据 |
| `train` | 监督训练线性 BC 模型，按 episode 留出验证，保存权重 |
| `run-sim` | 加载模型，连续读取仿真状态、预测、执行并记录闭环 |
| `shadow` | 读取真机状态并计算策略输出，不下发动作；需要已校准的坐标映射 |
| `tools/camera_probe.py` | D435f 可选探测入口；未在你的相机实测 |
| `tools/doctor.py` | 查看 OS、Python、依赖和 GPU 型号/显存 |

这个小模型从合成教师学习关节目标跟踪，目的是验证训练与部署的数据契约。**它不是 VLA，不看图像，不会抓取，也不能直接迁移实机。** 将来换 ACT/VLA 时保留观测、执行、安全和日志层，替换 policy adapter 和数据输入。

## 3. 你现有三台主机怎么安排

| 主机 | 当前角色 |
|---|---|
| Windows / 4080 SUPER | 代码与文档、现有 DummyStudio 对照、参考仿真；本机已跑通 |
| Ubuntu / 5070 | 未来现场设备主机：接机械臂 USB 和 D435f、采集、执行、本地推理 |
| 远端 4×RTX 6000 | 后续训练；通过现有 KIT 登录器的 VS Code 入口操作 |

5070 暂未配置 SSH，不影响直接在 Ubuntu 桌面终端操作。先把下载包通过 U 盘/已有文件共享复制到 Ubuntu。远端服务器不是第一版必须使用的依赖，参考小模型用 CPU 几秒内就能训练。

## 4. Ubuntu 第一次运行：先不接机械臂

将 `dummy-loop-ubuntu.zip` 解压到你的用户目录，进入 `dummy-loop` 文件夹。推荐原生 Ubuntu 24.04 / Python 3.12；本次尚未在该主机实测。

如果 `python3 -m venv` 不可用，在 Ubuntu 上安装：

```bash
sudo apt update
sudo apt install python3-venv python3-pip libgl1 libglfw3
```

在解压后的工程根目录运行：

```bash
bash tools/ubuntu_setup.sh
source .venv/bin/activate
python tools/doctor.py
```

脚本只创建本工程 `.venv`、安装固定版本依赖并跑测试。无需 ROS、CUDA、PyTorch 或远程服务器即可验证这一版。

然后按顺序执行：

```bash
python -m dummy_loop collect-sim
python -m dummy_loop train
python -m dummy_loop run-sim --viewer
```

窗口中的参考臂会由加载的模型自动控制，不需要鼠标逐关节拖动。默认运行 240 步，仿真控制周期 0.05 s，约 12 秒。无桌面时移除 `--viewer`。

产物：

```text
outputs/teacher.npz               合成示教数据
outputs/policy.npz                实际训练保存的权重和契约
outputs/rollout.jsonl             每步观测、预测、下发目标与下一状态
outputs/rollout.summary.json      闭环误差与通过状态
```

可以换一个目标检查：

```bash
python -m dummy_loop run-sim --viewer --goal -0.17 0.13 0.11 -0.08 0.16 -0.19
```

这里的六个数是**参考仿真关节空间的弧度**，不是可以复制给实机的命令。

## 5. Ubuntu 连接真机：第一轮只读

退出占用同一串口的 DummyStudio/其他工具。按你已验证的上电姿态和接线操作，把 USB 接到 Ubuntu，先运行：

```bash
source .venv/bin/activate
python -m dummy_loop ports
```

根据输出选择实际端口。若显示 `/dev/ttyACM0`，则：

```bash
python -m dummy_loop probe --port /dev/ttyACM0 --samples 20 --hz 2
```

不要默认端口永远是 ttyACM0；可用 `/dev/serial/by-id/` 的稳定路径。若没有端口，先检查 Type-C 朝向、数据线、板卡枚举和 `dmesg`；不要先刷固件。

若系统报告串口权限不足，在确认设备属组后将当前用户加入对应串口组（常见为 dialout），退出登录后重新登录：

```bash
sudo usermod -aG dialout "$USER"
```

该命令不在安装脚本中自动执行，也不需要 `chmod 777`。

成功时终端应显示六个固件关节角和请求耗时；原始日志在 `outputs/usb_probe.jsonl`。**此命令不发送 START、HOME、RESET 或动作。** 串口打开时底层 modem 信号是否导致板卡复位仍要现场确认。

程序容忍队列数字、普通 ok、启动日志等非关节帧，只有精确六个合法数字的 GETJPOS 响应才接受。超时即退出本轮，不盲目重试混淆新旧帧。协议查询读的是固件缓存，返回时间不能冒充每轴传感器采样时间。

第一轮将 `doctor.py` 输出、`ports` 输出和 20 次查询日志作为后续接入依据即可，不需要提供密码。

## 6. 从只读到程序控制：单轴调试门槛

本工程带 `configs/hardware.unverified.json`，默认参数为空且验证标志为 false。**不能为了让命令跑起来随手把 false 改成 true。** 它们应对应实际测量和可复核记录。

要完成：

- 实际 REF/六轴固件与源码的对应；USB 回包与模式 2 语义一致。
- 六轴正方向、零位、实际软限位；记录固件空间→规范空间的映射。
- 实机上软件停止、物理停止、断连和断使能行为，尤其是重力下垂。
- 速度字段的实际含义和上限，不把源码中的数值直接当度/秒。

复制 profile 到新文件并填入标定数据后，才使用 `jog`。它每次只做一次有限幅度单轴动作；先检查当前反馈，再显式使能、设置已验证模式、刷新状态、下发目标并监视。结束尝试软件停止，不自动 HOME，也不无条件 DISABLE。

查看参数：

```bash
python -m dummy_loop jog --help
```

未提供直接可复制的真机运动示例值，因为你的有效 profile 尚不存在。收到真实响应和标定信息后应据此生成首个测试配置，而不是从仿真范围推导。

`jog` 是现场有人监督的调试入口，不是认证安全控制器。停止指令可能在 USB 断开后发不出去，现场独立停止机制必须存在。回包代表缓存状态，仍无法从当前协议证明各轴在线；因此 `SerialRobot.send_action()` 明确阻止无人监督的自动模型控制。

## 7. 真机自动推理还需补的接口

自动模型运行比手动单步多了持续执行和积压风险。下一个实现目标应是：

1. REF 状态消息暴露每轴最近更新时间/序号、有效位、故障状态；在 CAN 失联时明确标错。仅比较关节值是否变化不可靠。
2. 校验并提供板端通信超时/watchdog：主机退出或 USB 断开后按已验证方式停止，不能只依靠 Python 发 STOP。
3. 固定命令的入队、拒绝、执行与完成语义；限制待执行队列，过期动作丢弃。
4. 针对实际固件修复/确认重复目标和软件停止中的零位移数值边界；此前历史源码发现可能 0/0，不能假定当前固件已修复。
5. 在 `SerialRobot` 后端接入这些验证结果，才开放与 `SimRobot.send_action()` 一致的自动接口。

这不是要求重做全部固件。先从当前烧录版本与可用遥测确认起：已有能力就复用，缺少的能力再做最小补充。不要将旧归档固件直接覆盖到已能工作的机器。

已确认规范映射后，可以用 `shadow` 对真实状态运行推理、只记录输出。参考仿真模型的策略输出仍属诊断值，不能作为真机任务策略。

## 8. 仿真模型的来源与边界

模型来自本地 `dummy-v1-auk` 的六轴 URDF 和 7 个 STL，已保留许可证、修改说明、来源路径与哈希；它提供可运行几何与关节链。没有声称它与任同学 V2 每个尺寸、零位和减速器完全一致。

为跑通控制软件，模型使用：

- 显式位置执行器并通过 `mj_step` 积分；每步执行不使用 `resetJointState` 瞬移。
- 理想重力补偿、示例阻尼和执行器限值。
- CAD 的零惯量数值正则。
- 关闭接触和碰撞响应。

所以它能检验关节状态→策略→执行的程序闭环，**不能检验抓取接触、碰撞安全或真实动力学泛化**。下一版应按实机标定修正关节原点/方向/零位和 TCP，加入简单碰撞体、经过验证的质量惯量、夹爪、桌面与物体。不要把未经校准的精细 STL 碰撞效果当真实接触。

先做关节一致性，随后做独立末端观测的 FK 对比，再做动力学辨识；不需要第一天追求完整数字孪生。

## 9. D435f、夹爪与学习算法怎么接上

D435f 固定在外部观察位置，后续 `Observation` 增加 RGB、深度、各自时间戳和标定版本。先查看设备支持的流，再运行有限帧探测：

```bash
# 仅在按设备/内核匹配的官方 librealsense 安装完成后：
python tools/camera_probe.py
python tools/camera_probe.py --frames 30
```

当前依赖清单没有强行安装 pyrealsense2；D435f 仍需在实际 Ubuntu 的内核/USB 环境验证。固定相机到基座的外参应独立标定，不等同于 RGB-depth 对齐。

夹爪要单独确认型号与协议。历史固件有 `hand` 对象，但 ASCII 六轴命令第七位是速度，不能拼夹爪值进去。夹爪通道调通之后，再扩展动作和实测/命令状态；初版不伪造夹爪反馈。

第一项任务推荐“一个轻质物体抓取放入固定区域”。采 5–10 条示教先验证同步，再扩大到能覆盖初始位置变化的数据。用 LeRobot ACT 做第一个视觉模型；远端训练，下载权重与归一化配置到本地，再进入影子运行与受限闭环。VLA 则在 ACT 基线稳定后替换策略模块。

## 10. KIT GPU 登录器怎么复用

已读 `E:\project\kit-gpu-launcher` 的 README、launcher.py 和 desktop.sh，没有读取密码配置、改动连接器或建立远端会话。

- 继续使用现有启动器的 **VS Code 开发**入口，在远端用户目录放置独立实验工程与数据；保持启动器运行以维持临时认证。
- VNC 入口通过 SSH 隧道工作。桌面脚本显式设置 `LIBGL_ALWAYS_SOFTWARE=true` 和 Mesa，因此“桌面能打开”不能证明 GPU 渲染可用。CUDA 计算与桌面渲染是不同路径。
- 连接器是认证和开发入口，不负责分配四张 GPU，也没有在已读逻辑中看到训练调度。先确认实验室是否要求 Slurm/PBS、可用卡与配额，再启动训练。
- 训练长任务应使用服务器允许的调度器或持久终端。关闭 VS Code/VNC 不是可靠的训练任务管理方式。
- 5070 不需要通过中央服务器中转控制机械臂。未来配置 SSH 后，可从当前 Windows 同时开发 Ubuntu 控制端和远端训练端。

服务器首轮只读检查建议：`nvidia-smi`、`python --version`、`command -v sbatch`、`command -v qsub`。不要根据 RTX 6000 名称推定显存或架构，也不要默认四张卡全部属于当前账号。

## 11. 5070 的 SSH：可后置

在 Ubuntu 本机完成串口探测后再配置也可以。若希望远程开发，在 Ubuntu 本机安装并启动 OpenSSH Server，查看局域网 IP，由当前 Windows 连接 `ssh 用户名@地址`。确认主机指纹后再认证，密码只在终端输入。

```bash
sudo apt install openssh-server
sudo systemctl enable --now ssh
hostname -I
systemctl status ssh --no-pager
```

防火墙如已启用，只给所需局域网/主机开放访问；不需要对公网暴露端口。本次没有替你执行这些系统修改。

## 12. 已验证与未验证

本机 Windows / Python 3.12.14 实测：14 个测试通过；MuJoCo 可加载、动力学积分可运行；40 个合成 episode / 2400 帧完成训练；权重重新加载后闭环目标误差从 0.30822 rad 降为 0.002226 rad（六轴向量范数）；参考模型图像已渲染检查。

未验证：Ubuntu 本机安装、USB 真实回包、真机运动、D435f、夹爪、远端登录/训练、ROS 2、碰撞与抓取、实机模型策略。测试中串口使用 fake transport，不把模拟收发当真实通信验收。

建议现在按次序完成两个可见结果：**Ubuntu 运行 MuJoCo 模型闭环窗口；Ubuntu 成功读取真机六轴角度。** 得到实际日志后再打通监督单步和自动执行。

官方依据：[MuJoCo Python/viewer](https://mujoco.readthedocs.io/en/stable/python.html)、[pySerial](https://pyserial.readthedocs.io/en/latest/shortintro.html)、[LeRobot 自定义硬件](https://huggingface.co/docs/lerobot/en/integrate_hardware)。
