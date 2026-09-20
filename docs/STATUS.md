# 当前状态及证据（2026-09-18）

| 内容 | 结论 | 证据 | 等级 |
|---|---|---|---|
| 参考仿真线性BC | 软件闭环通过；不是抓取/VLA | ../experiments/2026-09-16_baseline/raw/reference_learning/rollout.summary.json，unseen_rollout.summary.json | L1 |
| J1实际运动 | 用户确认实体运动；两接口试验留档 | ../experiments/2026-09-16_baseline/raw/commissioning_and_failures/native_j1_1789546152444180900.json及1789546220803937900.json | L3 |
| 实体折叠 | 用户确认目标姿态正确 | ../experiments/2026-09-16_baseline/raw/verified_poses/correct_fold_capture.json | L3 |
| 直立→折叠完整执行 | 固件反馈到位；最大差约0.006° | ../experiments/2026-09-16_baseline/raw/verified_poses/preset_recheck_1789550272034353800.report.json | L2 |
| J5视觉层级 | 长前臂/电机归属修正并进行数学对照 | tests/test_studio_articulation.py，docs/history/2026-09-16_J5可视化层级修复.md | L1 |
| USB反馈中断 | 已修复无限等待风险；重启后复测成功，未证明唯一根因 | docs/history/2026-09-16_折叠中断与反馈超时修复.md | L2 |
| 六轴自动往复和速度UI | 已实现，有软件测试；不代表所有组合姿态实机验收 | dummy_loop/live_control.py，tests/test_live_control.py | L1 |
| 直立末端外观/坐标 | 曾被用户质疑；独立空间标定尚未完成 | 不以折叠确认替代直立验收 | — |
| 新开发机环境与L1复现 | 钉死依赖装成；49测试通过；参考仿真终值与冻结基线逐位相同 | ../experiments/2026-09-19_windows_env/ | L1 |
| 墙面抹涂仿真链路 | 场景、末端控制、任务、示教录制与回放、误差与补偿均跑通；参数为假设 | ../experiments/2026-09-20_wall_sim_chain/，docs/SIMULATION.md | L1 |
| 仿真切换到 Dummy V2 | 运动学与 V2 固件 SolveFK 逐位一致；V2 限位、传动、裸轴末端入模；质量与电机参数为估计值，全部待 M3 实测 | ../experiments/2026-09-20_v2_model/，docs/hardware/DUMMY_V2.md | L1 |
| Ubuntu实机 / D435 / 夹爪 | 未完成实际闭环验证 | 待办 | — |
| ACDC / 世界模型 / FSDP | 讨论与研究计划；未安装、未训练、无复现结果 | ROADMAP.md | — |

实验结果位于 experiments/2026-09-16_baseline/raw/；源码路径相对于项目根目录。原始反馈是控制器读数，不是外部相机或末端测量系统给出的精度。

参考仿真240步：初始误差0.30822 rad→0.002226 rad；另一目标0.35496 rad→0.002546 rad。均为六维关节误差范数，不是每轴误差。

保留失败和中止日志，不能把有日志的测试都算成成功。详见实验目录索引。

证据等级定义见 AGENTS.md：L1 软件测试、L2 控制器反馈到位、L3 实体验收。
标「—」的条目尚无任何级别的证据。L2 不能冒充 L3：固件回报到位不等于机器真的在那个位置。

## 2026-09-18 工作区重构

主线工程收敛到 `D:\VLA\dummy_arm`，建立 git 基线并启用 Git LFS 管理网格资产。
223 个源文件经 SHA-256 核对与原目录一致；35 个单元测试通过；参考仿真闭环复现
（0.30822 → 0.0022264 rad），与归档基线一致。证据等级 L1。

Windows 侧 Tk 界面与原生 USB 枚举未在本次重构中复核，需在目标机器上单独验证。
本次未连接机械臂，未发送任何运动指令。记录见
`experiments/2026-09-18_m0_baseline/log.md`。

## 2026-09-19 新开发机落地（desktop-m51oshe）

仓库克隆到 `D:\project\VLA\dummy-arm-lab`，在该机器上用 `requirements/dev.txt`
钉死的版本重建 `.venv-loop`：Python 3.12.7、numpy 2.5.3、mujoco 3.13.0，无版本回退。
49 个单元测试通过（3 个依赖 DummyStudio 数据的用例按设计跳过），参考仿真闭环
0.3082207001484489 → 0.0022264043008919554 rad，终值与 2026-09-16 冻结基线**逐位相同**。
证据等级 L1，记录见 `experiments/2026-09-19_windows_env/`。
本次未连接机械臂，未打开串口，未发送任何运动指令。

**仍未关闭**：上面 2026-09-18 条目提出的「Windows 侧 Tk 界面与原生 USB 枚举需在目标
机器上单独验证」。本次只验证了依赖安装、单元测试与仿真复现，既未启动上位机界面，
也未枚举原生 USB。不要把本次结果当成 GUI 或 USB 通路可用的证据。

**更正**：2026-09-18 条目中「启用 Git LFS 管理网格资产」与仓库实际状态不符。
`.gitattributes` 明确写明本仓库当前不使用 LFS，克隆核对确认 263 个文件全部是普通
git 对象，未安装 git-lfs 也能完整检出。原条目按「原始记录不覆盖」保留，以本条为准。

**新增**：`tools/environment/windows_setup.ps1` 与 `windows_setup.cmd`，是
`ubuntu_setup.sh` 的 Windows 对应物，把上述五步固化成可重复执行的脚本，
输出直接落到 `experiments/<日期>_windows_env/raw/`。脚本本身不碰硬件。

## 2026-09-20 墙面抹涂仿真链路（停电期间，软件会话）

新增 `dummy_loop/wall/`（场景、末端动作空间与逆解、覆盖任务、脚本示教、误差模型、探触与压缩量补偿）
与通用录制格式 `dummy_loop/episode.py`；约定写在 `docs/ACTION_SPACE.md`，设计发现写在 `docs/SIMULATION.md`。
测试 49 → 66 个，全部通过（3 个按设计跳过）。证据等级 L1，记录见 `experiments/2026-09-20_wall_sim_chain/`。

对实机工作有直接影响的三条（均为仿真设计结论，待实机验证）：
1. 墙板距 J1 轴约 40 cm、工作区中心高约 20 cm、刀面垂直于 J6 轴时条件最好；
   早先按可达面积得出的 20–30 cm 会使手腕贴近奇异。依据的关节范围是假设的 ±90°，M3 后重算。
2. 覆盖率收益主要来自开工前探触墙面；转接件上加一个微动开关即可支持。
3. 最敏感的两项是关节伺服刚度与摩擦，二者目前都未测。

同时修改：`dummy_loop/sim_backend.py` 对超出 ctrlrange 的目标显式报错（以前被 MuJoCo 静默截断），
参考仿真闭环数值不变；`tools/environment/doctor.py` 增加 GPU 算力等级、bf16/TF32/FA2 支持与 torch/CUDA 信息。
本次未连接机械臂，未发送任何运动指令。

## 2026-09-20 修复：两个 .cmd 启动器在新机器上打不开

原因：二者都加载 `models/dummy_studio_visual.xml`，其 21 个 Studio 网格不随仓库分发，新机器上不存在，
启动即 FileNotFoundError。不是本次仓库改动引起的，是换机器后的首次暴露。
处理：缺网格时两者改用参考模型显示（实机同步上位机按逐轴比对得到的符号换算转向，并在窗口中注明），
`SimRobot` 缺网格时给出可操作的报错；新增 `tools/maintenance/verify_studio_meshes.py` 与
`tests/test_studio_fallback.py`（测试 66 → 70）。完整外观需从原开发机复制 `models/studio_meshes/`。
L1；本次未连接机械臂。

## 2026-09-20 仿真切换到 Dummy V2 模型

用户实机是 Dummy V2、末端为 J6 裸轴。新增 `models/dummy_v2.xml`（由 V2 资料包装配体 STEP 生成，
生成脚本 `tools/modeling/build_dummy_v2.py`，参数单一来源 `dummy_loop/v2.py`）。
运动学严格等于 V2 固件 DH，模型零位 = 固件 HOME，关节角 = 固件角 − HOME；
墙面仿真、上位机画面、`六轴同时往复仿真.cmd` 均改用它。BC 冒烟闭环仍用参考模型，历史数字不变。
测试 70 → 78，全部通过（3 个按设计跳过）。L1，记录见 `experiments/2026-09-20_v2_model/`。

对实机工作的直接影响（均为资料与仿真结论，待实机验证）：
1. **J6 减速比资料冲突**：V2 固件截图写 5，装配体是直驱。M3 从 J6 开始，≤5° 小角度测实际转角。
2. 墙面布局重算后默认改为墙距 40 cm、工作区中心高 15 cm；探触仍是覆盖率收益的主要来源。
3. J6 直驱可用扭矩约 0.07 N·m：抹刀必须与 J6 同轴；J2 在压 10 N 时超过减速器额定转矩，压力先按 5–10 N 设计。
4. 最便宜的下一步测量：称整机与小臂重量、看六个电机铭牌。

所有 V2 参数都是候选值，**未写入 `configs/`**，也没有改动任何 `*_verified` 字段。
2026-09-20 之前用参考模型录的墙面 episode 不能在当前代码上重放（关节坐标约定不同），原记录保留不改。
本次未连接机械臂，未发送任何运动指令。
