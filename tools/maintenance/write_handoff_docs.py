raise SystemExit("Historical one-time documentation generator; edit current Markdown files directly")
from pathlib import Path
R=Path.cwd()
def put(p,s):
 q=R/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s.strip()+'\n',encoding='utf-8')
put('README.md', '''# Dummy 机械臂实验工程

交接基线：2026-09-18。以本文件和 `docs/STATUS.md` 为当前说明；`docs/history/` 是保留的历史资料，早期结论不代表最新状态。

## 当前结果

- Windows 实机位置指令与原生 USB 角度读取已跑通。用户确认实体 J1 运动和折叠姿态。
- 上位机具备双视图、单轴目标、主机跟随速度、范围内连续往复、直立与折叠按钮。
- 完整预设复测：直立 `[0,0,90,0,0,0]`；折叠 `[0,-75,180,0,0,0]`；记录的最大反馈角度差约 0.006°，不是实体末端空间精度。
- 软件链路已有合成示范→线性行为克隆→参考 MuJoCo 闭环；不是视觉/VLA 或 ACDC 策略。
- Ubuntu 实机、D435 采集、夹爪、力控制、ACDC、世界模型、多卡训练尚未完成部署验证。

## 从哪里开始

1. [安装和不驱动实机的验证](docs/SETUP.md)
2. [当前状态与结果证据](docs/STATUS.md)
3. [源码结构与接口](docs/ARCHITECTURE.md)
4. [实验记录和复现规范](docs/EXPERIMENTS.md)
5. [实机操作边界](docs/HARDWARE.md)
6. [跨电脑交接、归档与许可](docs/HANDOFF.md)
7. [服务器条件和后续研究](docs/ROADMAP.md)

## 目录

```text
dummy_loop/              核心 Python 包，保持现有导入路径
configs/                 硬件配置（现有 profile 仍为未标定）
models/                  MJCF、URDF、网格、模型来源及许可
vendor/native_client/    实机必需的本地 Fibre 客户端及已修复超时
requirements/            核心、上位机、模型提取依赖和环境快照
tools/gui/               MuJoCo 上位机
tools/simulation/        离线可视化和渲染
tools/modeling/          模型构建和导入
tools/hardware/          实机/相机探测和有界调试脚本
tools/diagnostics/       Studio 协议和结构取证脚本
tools/environment/       环境检查和 Ubuntu 安装脚本
tools/maintenance/       归档、打包、交接校验工具
tools/*.py,*.ps1         老路径兼容入口；修改分类目录中的实现
experiments/             已归档的原始实验结果及 SHA-256 清单
outputs/                 程序当前输出；不是唯一历史结果来源
docs/                    当前交接文档，history/ 为原始研究和诊断资料
archive/                 整理前源码、旧交付包、上游原始工程快照
.diagnostics/            历史固件摘录、Studio诊断副本及Cecil；非日常入口
releases/                新生成的迁移包和校验文件（不提交Git）
```

## 快速验证（在项目根目录）

```powershell
.\\.venv-loop\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv-loop\\Scripts\\python.exe tools/live_mujoco.py --smoke-test
.\\.venv-loop\\Scripts\\python.exe tools/live_mujoco.py --offline
```

以上命令不连接机械臂。根目录两个 `.cmd` 保留，实机同步入口启动会尝试只读连接，开启跟随后才运动。旧脚本中存在主动调试入口，不能批量执行 tools 目录所有文件。

## 可移植性原则

运行环境不随源码迁移；目标电脑重新建 venv。核心代码、模型和 USB 客户端随包交付。历史取证脚本仍可能引用 `D:/VLA`，只能在提供对应上游资料后重用；这些不是正常运行的依赖。
''')
put('requirements/core.txt','numpy==2.5.3\nmujoco==3.13.0\npyserial==3.5')
put('requirements/gui.txt','-r core.txt\npillow==12.3.0\npyusb==1.3.1\nlibusb-package==1.0.30.0')
put('requirements/modeling.txt','-r gui.txt\nUnityPy==1.25.3')
put('requirements/README.md','''# 依赖分层

core.txt 是参考仿真/线性策略环境；gui.txt 加入 GUI 和原生 USB；modeling.txt 加入 Unity 资源提取。
Windows 已用 Python 3.12.14。完整已安装版本见 windows-observed-freeze.txt（取证快照，不是 Linux 通用锁文件）。Tk 为 Python/系统组件，不由 pip 安装。Linux 可能需要管理员提供 python3-tk、OpenGL/EGL 和设备权限。
相机需要额外的 pyrealsense2/librealsense；当前相机未实测，不在可复现基础依赖中锁定未经验证版本。
UnityPy 及其解码依赖只用于模型取证，正常 GUI 不需要。某版本在目标平台无 wheel 时应记录安装失败并建立单独环境，不能默默升级全部依赖后称为原版本复现。
''')
put('docs/SETUP.md','''# 新电脑安装与验证

## Windows

安装64位 Python 3.12（包含 Tk）。在项目根目录：

```powershell
py -3.12 -m venv .venv-loop
.\\.venv-loop\\Scripts\\python.exe -m pip install -r requirements/gui.txt
.\\.venv-loop\\Scripts\\python.exe tools/doctor.py
.\\.venv-loop\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv-loop\\Scripts\\python.exe tools/live_mujoco.py --smoke-test
.\\.venv-loop\\Scripts\\python.exe tools/live_mujoco.py --offline
```

离线双视图确认后再参考 HARDWARE.md 连接。Windows原生USB接口可能需要WINUSB接口注册，不能在新电脑自动套用旧注册表实例路径/备份。

## Ubuntu / 其他 Linux

Ubuntu 是拟验证的目标环境，不是已经完成实机验收的平台。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/gui.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/live_mujoco.py --offline
```

Linux主机需可用Tk和图形驱动。没有桌面的服务器先仅安装core.txt运行单元测试和不渲染的参考实验；虚拟相机仍需渲染环境。不要复制Windows venv。
USB串口名称、udev/用户组权限、libusb访问、原生接口发现需要单独验证。现有原生客户端和设备身份检查仍绑定本实验机械臂，换另一台机械臂不能直接认为等价。

## 参考策略流程

先运行 `python -m dummy_loop --help` 及各子命令 `--help` 查参数。现有 CLI 包括 collect-sim、train、run-sim、shadow、ports 等。归档policy.npz只适用于参考仿真。新增数据必须输出到新的实验目录，不覆盖baseline。

## 验证范围

本次交接执行的软件测试和迁移路径测试记录在 experiments/2026-09-18_handoff/。未发送实机指令；未连接Ubuntu或服务器；安装命令是目标环境操作说明，不冒充已实测。
''')
put('docs/ARCHITECTURE.md','''# 源码结构和数据契约

## 核心包

| 文件 | 职责 |
|---|---|
| dummy_loop/core.py | Observation、角度/形状检查、Guard和硬件profile |
| dummy_loop/serial_backend.py | 串口协议、原始收发日志、调试命令 |
| dummy_loop/live_control.py | 单后台线程、原生角度反馈、目标限速、停止与预设状态机 |
| dummy_loop/sim_backend.py | MuJoCo参考模型和统一仿真接口 |
| dummy_loop/policy.py | 合成教师与线性行为克隆；非VLA |
| dummy_loop/__main__.py | 采集、训练、仿真运行和只读影子模式CLI |
| tools/gui/live_mujoco.py | Tk双视图界面与显示刷新 |
| vendor/native_client/fibre/ | 原生USB传输、接口发现；有本项目超时修复 |

## 坐标和单位

核心Robot API及参考策略使用弧度；实机上位机目标/反馈为固件角度（度）。Studio视觉模型显示采用 `qpos = deg2rad(firmware_deg - [0,0,90,0,0,0])`。原生joint.angle需加偏置 `[0,-75,180,0,0,0]`；初始化与一次串口读数交叉核对。
不要把MuJoCo qpos直接当成实机角度。参考模型和Studio模型不是同一套已标定动力学。

## 控制流

UI目标 → LiveController → 角度/目标速度/跟随误差检查 → ASCII串口目标 → 控制板。
原生USB缓存角度 → LiveController → UI反馈姿态和日志。
GUI的mj_forward仅计算显示姿态，未进行训练用的接触动力学验证。

关键参数：主机目标速度1～20°/s；最大目标领先2°；跟随误差超过3°停止；反馈过期会请求取消动作。固件缓存反馈的接收时间不等于传感器采样时间。STOP应答不等于独立实体停止测量。

## 模型

models/dummy_reference.xml 来自auk参考模型，支持早期软件闭环；models/dummy_studio_visual.xml用于上位机和六轴可视化。Studio关节层级通过实际Transform及桥接代码核对，相关来源在 models/studio_source/；质量、惯量、力矩、接触及机械零点仍不能视为实机标定值。

## 工具分类与旧路径

分类实现在tools各子目录；旧tools同名入口转发以保留启动脚本和历史命令。完整映射见tool_migration.json。仅支持从根目录运行，或先将项目根目录加入PYTHONPATH；本工程不宣称可独立pip安装核心包后自动携带外部模型。
''')
put('docs/STATUS.md','''# 当前状态及证据（2026-09-18）

| 内容 | 结论 | 证据 |
|---|---|---|
| 参考仿真线性BC | 软件闭环通过；不是抓取/VLA | baseline/raw/reference_learning/rollout.summary.json，unseen_rollout.summary.json |
| J1实际运动 | 用户确认实体运动；两接口试验留档 | baseline/raw/commissioning_and_failures/native_j1_1789546152444180900.json及1789546220803937900.json |
| 实体折叠 | 用户确认目标姿态正确 | baseline/raw/verified_poses/correct_fold_capture.json |
| 直立→折叠完整执行 | 固件反馈到位；最大差约0.006° | baseline/raw/verified_poses/preset_recheck_1789550272034353800.report.json |
| J5视觉层级 | 长前臂/电机归属修正并进行数学对照 | tests/test_studio_articulation.py，docs/history/2026-09-16_J5可视化层级修复.md |
| USB反馈中断 | 已修复无限等待风险；重启后复测成功，未证明唯一根因 | docs/history/2026-09-16_折叠中断与反馈超时修复.md |
| 六轴自动往复和速度UI | 已实现，有软件测试；不代表所有组合姿态实机验收 | dummy_loop/live_control.py，tests/test_live_control.py |
| 直立末端外观/坐标 | 曾被用户质疑；独立空间标定尚未完成 | 不以折叠确认替代直立验收 |
| Ubuntu实机 / D435 / 夹爪 | 未完成实际闭环验证 | 待办 |
| ACDC / 世界模型 / FSDP | 讨论与研究计划；未安装、未训练、无复现结果 | ROADMAP.md |

上表baseline路径相对于experiments/2026-09-16_baseline/（省略baseline前缀时同义）。原始反馈是控制器读数，不是外部相机或末端测量系统给出的精度。

参考仿真240步：初始误差0.30822 rad→0.002226 rad；另一目标0.35496 rad→0.002546 rad。均为六维关节误差范数，不是每轴误差。

保留失败和中止日志，不能把有日志的测试都算成成功。详见实验目录索引。
'''.replace('baseline/raw/','../experiments/2026-09-16_baseline/raw/'))
put('docs/HARDWARE.md','''# 实机使用和移植边界

已验证设备：VID 1209 / PID 0D32，序列号325F368F3135；原机端口COM6，目标电脑必须重新枚举。原生客户端运行依赖已放在vendor/native_client，不需要安装外部同名fibre包。

确认预设（度）：HOME=[0,0,90,0,0,0]；FOLD=[0,-75,180,0,0,0]。
GUI当前软件范围：J1[-20,20]，J2[-75,20]，J3[90,180]，J4/J5/J6[-15,15]。这是调试范围，不是机械全限位标定结果。折叠坐标已由用户摆到正确姿态后只读核验。

原版Studio和本上位机不要同时占用串口。上位机默认读取；开启跟随、姿态按钮和自动往复会驱动实体。SPACE/Esc发软件停止；协议停止失败时界面会告知，不能替代硬件停止。

ASCII目标是六个角度、逗号分隔、末尾逗号、LF；START/STOP及应答解析见serial_backend。返回ok代表命令应答/入队，不能独立证明到位。

历史commission_*脚本属于监督调试；直接电机映射未验证，不应作为策略执行API。注册表修复PS脚本包含旧电脑MI_02实例，只保留取证和受控修复用途，不要在另一台电脑自动执行。旧注册表备份不能用于新设备回滚。

要上模型控制，仍需动作/状态时效、坐标变换、速度和误差限制、故障处理与独立验收。当前整理不改变控制参数、不刷固件、不执行实机测试。
''')
put('docs/EXPERIMENTS.md','''# 实验管理规范

## 已有基线

experiments/2026-09-16_baseline/raw/ 保存整理时 outputs/ 全部117个普通文件的原样副本，包括成功、失败、截图、模型与数据。manifest.json包含原路径、大小、SHA-256；原outputs保留以兼容旧命令。重复副本用于冻结证据，不代表新增实验。

- reference_learning：合成教师数据、线性policy和rollout。
- verified_poses：有用户确认或完整反馈报告的姿态相关记录；同类失败报告仍保留，需读result字段。
- commissioning_and_failures：串口、原生USB试验及中止记录，不能整体视为成功。
- studio_protocol：Studio抓取、IL、Transform和连接失败记录。
- visual_validation：模型图片与GIF（不是实机录像）。
- machine_specific_usb：旧机器注册表备份与修复结果；仅本机历史证据。

## 新实验

从experiments/_template复制到YYYY-MM-DD_short-name，填写run.json和README。每次运行使用唯一目录，保留命令、环境、配置、随机种子、数据划分、模型哈希、训练/推理耗时、显存、结果和失败原因。不要覆盖baseline或同名权重。

真实数据至少说明单位、坐标系、控制频率、时间戳来源、实际使用硬件和操作是否有监督。用户肉眼确认、固件反馈到位、独立测量精度分别记录。

## 校验

`python tools/maintenance/verify_evidence.py` 只校验归档文件，不打开硬件。
后续如需论文复现，必须单独定义指标、测试集和重复次数；本项目的冒烟测试不等于科研性能评估。
''')
put('experiments/_template/run.json','''{
  "experiment_id": "YYYY-MM-DD_task", "status": "not_started",
  "purpose": "", "source_version": "", "command": [],
  "environment_file": "", "config_file": "", "seed": null,
  "data_split": "", "hardware": {}, "units": "",
  "metrics": {}, "artifacts": [], "limitations": [], "hardware_motion": false
}''')
put('experiments/_template/README.md','''# 实验名称

目的：
输入数据与版本：
执行命令和环境：
结果与证据：
失败与局限：
下一步：
''')
put('docs/ROADMAP.md','''# 后续实验与计算条件

## 下一步按顺序推进

1. 独立核对Dummy关节零位、端点坐标、限位和模型；完成夹爪及D435采集。
2. 在仿真跑通一个可重复任务，记录状态/图像/动作；训练并重新加载Dummy专用策略。
3. 只读策略影子运行，再进行有界实机验证。
4. ACDC先复现官方环境和checkpoint，再适配Dummy；它是场景生成与策略学习流程，不是任意机器人通用控制模型。
5. 世界模型和施工场景泛化属于后续研究；定义基线、指标和实机证据后再做资源扩展。

## 用户提供的硬件快照，未由本次整理重新连接验证

本地Windows：4080 Super；另一台Ubuntu：5070（SSH部署状态需重新确认）；现场有D435深度相机、夹爪待调。
中央服务器：4×Quadro RTX 6000 Turing，每卡24GiB；用户允许使用全部可用资源；CPU报告40核80线程、256GB RAM；RHEL8.10，GNOME3.32.2，VNC为llvmpipe；不能变更系统级配置。驱动610.57.04、nvidia-smi CUDA UMD13.3均为用户提供历史快照，不证明Toolkit/PyTorch可用。
/work约2.8TB，个人路径和额度仍需确认。不得使用其他用户目录。实际账户配置不写入源码/交接包。

优先将其用于兼容训练、特征提取、多组实验；FSDP/ZeRO-3可切模型训练状态，但四卡不自动合并为96GiB。Turing对BF16/新注意力算子有兼容限制，RHEL基础库和GPU渲染需单独检查。未部署ACDC/Isaac/多卡训练，不能把建议算作结果。
''')
put('vendor/README.md','''# 本地第三方运行依赖

native_client/fibre来自此前用户提供的原生USB客户端副本。保留原文件注释；本项目修改包含控制台兼容及USB有限超时。运行使用此副本，不使用pip同名包。
原始诊断副本在.diagnostics/native_client，完整上游项目归档和来源见archive/upstream/。本整理没有重新审定第三方许可证；未提供明确许可的材料不可默认公开再分发。
''')
put('archive/README.md','''# 历史与上游归档

layout_before_2026-09-18：分类前本项目源码和入口备份，不作为新的开发入口。
legacy_deliveries：早期Ubuntu交付包及测试解压，缺少后续实机依赖，不再推荐使用。
upstream/original_sources.zip：用户指定D:/VLA和服务器助手的上游资料快照；manifest.json逐文件记录原路径及哈希。排除虚拟机磁盘、旧虚拟机压缩包、VirtualBox安装器、缓存、Git元数据和服务器个人配置。原目录未修改。

.diagnostics是历史Studio二进制副本、Cecil和固件摘录。部署包中只收必要native_client；完整历史包另行保留其余文件。固件源码的存在不说明与实机已刷固件逐字一致。
''')
put('docs/HANDOFF.md','''# 交接与迁移

## 交付物

- releases/dummy-experiment-runtime-2026-09-18.zip：当前源码、全部模型、USB客户端、依赖、文档、分类实验结果、兼容入口。排除venv、Git元数据、外部账户配置、历史Studio程序和旧交付包。
- releases/dummy-experiment-history-2026-09-18.zip：整理前源码、旧交付包、历史诊断副本和原始上游归档。作为取证，不作为执行目录。
- 每个zip有SHA-256文件和内容清单。解压运行包后按SETUP重新创建环境，不复制原电脑venv。

执行 `python tools/maintenance/package_release.py` 重新打包；已有同名文件时拒绝覆盖，改用 --tag 新标签。运行包不自动下载外部资料、不写系统注册表、不发送实机命令。

## 完整性和范围

本项目目录中的源码、网格、配置和实验记录均有归档。原始D:/VLA工程的源码、固件、硬件资料、Studio资产和原始压缩包在独立上游zip中；大体积虚拟机镜像及安装器仅列清单，原件仍在原路径。服务器助手仅归档源码和依赖工具，账户配置、known_hosts、preferences和快捷方式排除，目标电脑重新配置。

models/UPSTREAM_LICENSE仅适用于对应来源，不能套用于其他上游。归档包适用于内部交接，公开发布前核对各来源许可和日志中的设备/路径标识。没有把整个混合工程擅自改为统一开源许可。

## 接手者检查

1. 校验zip哈希、解压、建立环境。
2. 运行verify_evidence和单元测试。
3. 离线渲染和GUI预览。
4. 阅读STATUS、HARDWARE，再检查本机设备身份/权限。
5. 在新实验目录继续，不修改既有基线。

旧README_实验闭环.md及docs/history原文保留，可能包含已过时状态；新README优先。2026-09-18整理只验证软件和迁移包，不宣称Linux实机或服务器训练已完成。
''')
put('tools/README.md','''# 工具入口

实现位于gui、simulation、modeling、hardware、diagnostics、environment、maintenance子目录。根目录同名脚本为旧命令兼容入口。
日常：tools/live_mujoco.py --offline；tools/view_simulation.py --studio；tools/doctor.py。
--smoke-test只生成预览。硬件脚本不要批量运行，commission_*含真实运动入口，repair_*涉及Windows设备注册。
历史取证脚本含原始D:/VLA路径，应先配置输入。maintenance/organize_20260918.py是一次性迁移记录，检测到备份时拒绝重复执行。
''')
put('outputs/README.md','''# 临时/运行输出

应用继续输出到此目录以保持兼容。2026-09-18整理时已有文件冻结在experiments/2026-09-16_baseline/raw，含完整清单。新的正式实验应使用唯一实验目录；不要据文件名推断成功。
''')
put('experiments/2026-09-16_baseline/README.md','''# 既有实验基线

采集与调试主要发生于2026-09-16；2026-09-18冻结。原始文件未修改，manifest.json记录SHA-256及原outputs路径。目录按用途分类而不是按成功与否分类，失败/中断也保留。
证据解释见../../docs/STATUS.md及../../docs/EXPERIMENTS.md。policy.npz是参考线性BC模型，不是Dummy实机可直接部署策略。
''')
print('Wrote handoff documentation and requirement layers')
