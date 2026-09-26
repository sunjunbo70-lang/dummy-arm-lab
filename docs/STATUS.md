# 当前状态及证据（2026-09-23）

## 2026-09-23 wall-cycle v0.6 P2-A（软件会话）

修复了装料姿态、带料搬运朝下和接触前回缩三项结构问题。v0.6 删除策略声明的
`carry_face_up` 动作维，改用 MuJoCo 实际工具法向计算姿态；装料、朝上搬运、近墙旋转、
单调接近、涂抹、离墙和扫描返回保持同一连续仿真状态。30 个固定种子、每个 25 次循环的
结构门禁中，材料面朝下帧和接触前回缩均为 0，最差搬运倾角约 5.83°，质量守恒最大误差
`2.17e-19 m³`。150 项测试通过，3 项因未分发的 DummyStudio 网格按设计跳过。

P2-A 完成 200 回合教师数据、4×40 回合 DAgger 和 50×2048=102,400 个 PPO 决策步，
总教师/DAgger 样本 26,407。固定 60 场景表测试中，PPO 相对 DAgger 只改善浪费率
（0.1525 对 0.3620），覆盖率与 RMSE 均更差（0.1775 对 0.2666；1.915 对 1.827 mm），
成功率均为 0。20 场景 MuJoCo 复测中，PPO 为覆盖 0.3262、RMSE 2.056 mm、浪费 0.5212，
材料面朝下帧与接触前回缩仍为 0，但平均 24.05 次投影、1.6 次不可达。因此 P2-A 未通过
晋级门槛，未启动 P2-B。装料仍是由实际朝上姿态门控的降阶代理，不是颗粒/流体接触。
证据：`experiments/v0.6/p0_gates/records/2026-09-23_wall_cycle_v06_gates/`、
`experiments/v0.6/p2a/records/2026-09-23_wall_cycle_v06_p2a/`；等级 L1；未连接硬件。

## 2026-09-22 wall-cycle v0.5 P1 训练（软件会话）

接入实验室末端工具 profile、守恒料台装料、朝上搬运时间积分、细分技能阶段和 14 维动作，
完成 120 回合示教、3 轮 DAgger、60×1024 步的 256×256 MLP PPO 训练，并交付原生 MuJoCo
回放脚本 `新一轮抹墙训练MuJoCo回放.cmd`。训练过程中发现并修复搬运奖励零分母，以及 MuJoCo
执行器 `q_last` 和 IK RNG 跨回合泄漏；旧联合仿真统计保留但已由确定性复核取代。

最终固定 100 场景表测试中，最佳 PPO 覆盖 0.221、RMSE 2.033 mm、浪费 0.251、成功率 0；
确定性 30 场景 MuJoCo 复核中，PPO 为 0.202 / 2.209 mm / 0.289 / 0，教师为
0.208 / 1.991 mm / 0.153 / 0。P1 学会朝上搬运和多方向动作，但未完成整墙，也未超过教师。
证据：`experiments/v0.5/p1/records/2026-09-22_wall_cycle_v05/`；等级 L1；未连接硬件。

## 2026-09-22 整片墙面连续强化学习（软件会话）

新增独立的 `dummy_loop/wall_cycle/`：墙面与抹刀均为守恒空间材料场，D435 代理观测隔离真实厚度，
11 维动作可输出任意起终点、刀角、曲率、力和速度；一次决策完整展开为装料/复用、接近、接触、施工、
离墙、回位和重扫。训练采用行为克隆、三轮 DAgger 与 20 轮 PPO（10,240 个 PPO 决策步）。
PPO 在 100 个混合初态上的平均覆盖率 0.924、厚度 RMSE 0.382 mm、浪费率 0.228，显式完成成功率 0.30；
最终保守检查点在另一组 100 个随机种子上为 0.916 / 0.401 mm / 0.225、成功率 0.35。
成功 PPO 回放 seed 2219 为 30 循环、3 次装料、覆盖率 0.9949、RMSE 0.155 mm；MuJoCo 逆解全部成功，
阶段端点之间另插入限幅关节轨迹帧，避免回位与再接近时视觉跳变。
双击 `整片连续强化学习可视化.cmd` 可自由拖动三维相机并检查墙面/刀上积料和多维数据。
证据：`experiments/v0.2/r0/records/2026-09-22_wall_cycle_rl/`；等级 L1；未连接硬件。材料与 D435 参数均未实测标定。

## 2026-09-21 PPO 动作可视化（软件会话）

合入本地 `plaster_update` 中已完成的 RL 更新与两份权重，新增实际 PPO 推理的离线播放器。
双击 `强化学习动作可视化.cmd` 可看机械臂全景、抹刀近景与逐步形成的砂浆高度场，
支持暂停、逐帧、慢放和策略切换。说明见 `docs/RL_VISUALIZATION.md`。
手法预热整片覆盖 0.9796、厚度 RMS 0.371 mm；平刀预热 0.9796、0.819 mm。
渲染轨迹与独立无渲染重跑的关节/厚度状态逐元素完全一致；102 测试完成（3 跳过）。
证据：`experiments/v0.1/r0/records/2026-09-21_ppo_visualization/`，等级 L1。换条带仍是明确标注的重置，
材料仍为未标定代理模型；未连接硬件。

| 内容 | 结论 | 证据 | 等级 |
|---|---|---|---|
| 参考仿真线性BC | 软件闭环通过；不是抓取/VLA | ../experiments/00_initial_debug/records/2026-09-16_baseline/raw/reference_learning/rollout.summary.json，unseen_rollout.summary.json | L1 |
| J1实际运动 | 用户确认实体运动；两接口试验留档 | ../experiments/00_initial_debug/records/2026-09-16_baseline/raw/commissioning_and_failures/native_j1_1789546152444180900.json及1789546220803937900.json | L3 |
| 实体折叠 | 用户确认目标姿态正确 | ../experiments/00_initial_debug/records/2026-09-16_baseline/raw/verified_poses/correct_fold_capture.json | L3 |
| 直立→折叠完整执行 | 固件反馈到位；最大差约0.006° | ../experiments/00_initial_debug/records/2026-09-16_baseline/raw/verified_poses/preset_recheck_1789550272034353800.report.json | L2 |
| J5视觉层级 | 长前臂/电机归属修正并进行数学对照 | tests/test_studio_articulation.py，docs/history/2026-09-16_J5可视化层级修复.md | L1 |
| USB反馈中断 | 已修复无限等待风险；重启后复测成功，未证明唯一根因 | docs/history/2026-09-16_折叠中断与反馈超时修复.md | L2 |
| 六轴自动往复和速度UI | 已实现，有软件测试；不代表所有组合姿态实机验收 | dummy_loop/live_control.py，tests/test_live_control.py | L1 |
| 直立末端外观/坐标 | 曾被用户质疑；独立空间标定尚未完成 | 不以折叠确认替代直立验收 | — |
| 新开发机环境与L1复现 | 钉死依赖装成；49测试通过；参考仿真终值与冻结基线逐位相同 | ../experiments/00_initial_debug/records/2026-09-19_windows_env/ | L1 |
| 墙面抹涂仿真链路 | 场景、末端控制、任务、示教录制与回放、误差与补偿均跑通；参数为假设 | ../experiments/00_initial_debug/records/2026-09-20_wall_sim_chain/，docs/SIMULATION.md | L1 |
| 抹涂手法强化学习闭环 | 单刀环境 + 降阶材料模型 + 纯 numpy PPO 跑通；PPO 稳定超过手写脚本；奖励权重决定学到什么 | ../experiments/2026-09-21_stroke_rl/，docs/RL.md | L1 |
| 抹涂手法学习 + 多刀整片 | 加入「刀上兜不住料就掉」的物理后，PPO 从 −6.3/−10.5 学到 +8.2/+8.3，覆盖率 1.0、掉料 0；单刀策略直接迁移到 196 cm² 整片（覆盖 0.98、RMS 0.37 mm）。但奖励只唯一确定「后缘间隙 = 目标厚度」，不唯一确定俯仰手法 | ../experiments/v0.1/r0/records/2026-09-21_plaster_session_rl/，docs/RL.md | L1 |
| 整片连续强化学习 v0.2 | 材料守恒、D435 代理观测、连续空间动作与完整回位/装料/重扫闭环；PPO 100 组平均覆盖 0.924、RMSE 0.382 mm；自由视角回放可用。复核：PPO 比 DAgger 差（回报 36.2→24.5，成功率 0.40→0.30），作业区未按可达计算，材料物理有误，见 v0.3 变更文档 | ../experiments/v0.2/r0/records/2026-09-22_wall_cycle_rl/，docs/WALL_CYCLE_RL.md | L1 |
| 整片连续强化学习 v0.3 | 砂浆改为基于物理的屈服应力模型（无角度规则）；作业区按用户规则从真实可达范围算出（33 cm，18.5×18.5 cm）；每刀在 Dummy V2 MuJoCo 上规划与联合仿真；只差俯仰的两个教师，立起的浪费 16.5% 对平刀 32.5%；PPO 从平刀起步把俯仰推高到 5°，但未超过教师，整片覆盖 0.22–0.27 | ../experiments/v0.3/r0/records/2026-09-22_wall_cycle_v03/，experiments/v0.3/r0/design/2026-09-22_wall_cycle_v0.3.md | L1 |
| 末端改为 J6 减速器 + 尖头抹刀 | 仿真工具链与实际方案一致（刚性，无弹簧；尺寸按照片估计）；结论：必须有抹刀座力传感器，否则覆盖率约 0.38、撞墙力可达 141 N | ../experiments/00_initial_debug/records/2026-09-20_rigid_tool/，docs/SIMULATION.md | L1 |
| 仿真切换到 Dummy V2 | 运动学与 V2 固件 SolveFK 逐位一致；V2 限位、传动、裸轴末端入模；质量与电机参数为估计值，全部待 M3 实测 | ../experiments/00_initial_debug/records/2026-09-20_v2_model/，docs/hardware/DUMMY_V2.md | L1 |
| Ubuntu实机 / D435 / 夹爪 | 未完成实际闭环验证 | 待办 | — |
| ACDC / 世界模型 / FSDP | 讨论与研究计划；未安装、未训练、无复现结果 | ROADMAP.md | — |

实验结果位于 experiments/00_initial_debug/records/2026-09-16_baseline/raw/；源码路径相对于项目根目录。原始反馈是控制器读数，不是外部相机或末端测量系统给出的精度。

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
`experiments/00_initial_debug/records/2026-09-18_m0_baseline/log.md`。

## 2026-09-19 新开发机落地（desktop-m51oshe）

仓库克隆到 `D:\project\VLA\dummy-arm-lab`，在该机器上用 `requirements/dev.txt`
钉死的版本重建 `.venv-loop`：Python 3.12.7、numpy 2.5.3、mujoco 3.13.0，无版本回退。
49 个单元测试通过（3 个依赖 DummyStudio 数据的用例按设计跳过），参考仿真闭环
0.3082207001484489 → 0.0022264043008919554 rad，终值与 2026-09-16 冻结基线**逐位相同**。
证据等级 L1，记录见 `experiments/00_initial_debug/records/2026-09-19_windows_env/`。
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
测试 49 → 66 个，全部通过（3 个按设计跳过）。证据等级 L1，记录见 `experiments/00_initial_debug/records/2026-09-20_wall_sim_chain/`。

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
测试 70 → 78，全部通过（3 个按设计跳过）。L1，记录见 `experiments/00_initial_debug/records/2026-09-20_v2_model/`。

对实机工作的直接影响（均为资料与仿真结论，待实机验证）：
1. **J6 减速比资料冲突**：V2 固件截图写 5，装配体是直驱。M3 从 J6 开始，≤5° 小角度测实际转角。
2. 墙面布局重算后默认改为墙距 40 cm、工作区中心高 15 cm；探触仍是覆盖率收益的主要来源。
3. J6 直驱可用扭矩约 0.07 N·m：抹刀必须与 J6 同轴；J2 在压 10 N 时超过减速器额定转矩，压力先按 5–10 N 设计。
4. 最便宜的下一步测量：称整机与小臂重量、看六个电机铭牌。

所有 V2 参数都是候选值，**未写入 `configs/`**，也没有改动任何 `*_verified` 字段。
2026-09-20 之前用参考模型录的墙面 episode 不能在当前代码上重放（关节坐标约定不同），原记录保留不改。
本次未连接机械臂，未发送任何运动指令。

## 2026-09-20 修复：实机同步上位机全屏后布局与缩放

原窗口按固定像素排版（三维画面固定 540×430，滑块固定长度），全屏后内容挤在左上角、不放大，高分屏上字很小。
改为：进程声明 DPI 感知并按屏幕 DPI 设字号；三维画面随窗口大小重建渲染器（上限 1920×1440）；
滑块与速度条随宽度伸展；说明文字自动换行；Ctrl+滚轮 / Ctrl+= / Ctrl+- 调整字号；`--fullscreen` 启动即最大化。
同时修复重建渲染器的释放顺序（旧 MjrContext 会删掉新渲染器的帧缓冲导致黑屏）。
只改显示层（`tools/gui/live_mujoco.py`），串口与控制逻辑未动。L1：在 1150×880 与 1900×1060 窗口下截图核对；未连接机械臂。

## 2026-09-20 末端工具改为 J6 减速器 + 3D 打印抹刀座 + 尖头抹刀

用户说明实际方案：去掉现有「J6 减速器 + 夹爪电机 + 夹爪」中的夹爪电机和夹爪，只保留减速器，
在输出法兰上装 3D 打印抹刀座卡住日式尖头抹刀的木柄；整个末端刚性，没有伸缩杆。仿真默认工具改为该结构
（减速器壳体固定在 J6 电机座上，法兰随 J6 转；抹刀 120 × 27.5 mm，按照片估计），弹簧滑轨方案保留为 `--tool spring` 仅作对比。
压力传感改为法兰与抹刀座之间的力传感器。重跑 30 组随机误差：无补偿平均覆盖率 0.378、最大撞墙力 141 N；
探触 + 力闭环 0.893、最大 47 N。测试 78 → 81。L1，记录见 `experiments/00_initial_debug/records/2026-09-20_rigid_tool/`。未连接机械臂。
待确认：J6 减速器型号与减速比（目前 J6 仍按直驱建模）；抹刀、木柄、抹刀座的实际尺寸与质量。

## 2026-09-21 抹涂手法强化学习闭环

新增降阶材料模型（墙面高度场 + 刀上带料量）、单刀强化学习环境（动作：竖直移动 / 法向压入 / 刀面俯仰，
观测 7 维全部实机可得）、纯 numpy 的 PPO 与训练闭环（脚本示教 → 行为克隆 → 价值预热 → PPO 微调），
命令行 `rl-train` / `rl-eval`，启动器 `抹涂手法强化学习.cmd`。测试 81 → 94。
L1，记录见 `experiments/2026-09-21_stroke_rl/`，说明见 `docs/RL.md`。未连接机械臂。

结论：**闭环通了**。PPO 稳定超过手写脚本（默认权重 2.09 → 2.67；调权重后 −2.49 → −1.36，
厚度 RMS 偏差 1.424 → 1.341 mm）。三点经验：从零探索学不动，必须先示教预热；
微调容易把示教手法打坏，需要价值预热 + 冻结观测归一化 + 小噪声小学习率；
奖励权重决定学到什么（默认权重下策略选择把料全倒出去来降低浪费项，厚度反而更差）。
材料是代理模型、系数是假设值，仿真里的最优手法不等于真实砂浆下的最优手法。

## 2026-09-21 抹涂手法强化学习：完整闭环（单刀手法 + 多刀整片）

按用户要求把「回到立起待命位 → 下缘先贴墙 → 逐渐放平抹上去」做成可学的目标，并扩展到整片作业。
三处新增：材料模型加入**刀上带料容量与掉料规则**（楔形容量随俯仰增大，所以放平就兜不住料）；
`dummy_loop/wall/session.py` 多刀整片（逐条带试逆解筛出够得到的、每刀上一份料、共用高度场）；
命令行 `rl-session` 与对比工具 `tools/simulation/rl_compare.py`，`rl-train` 增加 `--script-style`。
测试 94 → 101。L1，记录见 `experiments/v0.1/r0/records/2026-09-21_plaster_session_rl/`，说明见 `docs/RL.md`。未连接机械臂。

结论：**流程跑通，而且确实学到了东西**。加入掉料规则后手写脚本只有 0.64 覆盖（不及格），
PPO 从 −6.26 学到 **+8.20**（覆盖 1.0、厚度 RMS 0.356 mm、掉料 0）；
用「不用手法」的平刀示教预热做对照，同样从 −10.52 学到 **+8.32**，说明学到的不是复述示教。
单刀策略**未经整片训练直接迁移**：196 cm² 三条带，覆盖 0.98、RMS 0.371 mm，优于脚本的 0.898/0.879 mm。

但有一条必须记住的限定：**奖励唯一逼出来的是「把后缘间隙锁在目标厚度上、随剩料放平」，
不是「俯仰必须从 30° 缓慢放平」**——平刀预热的策略只保留 2°~4° 小俯仰也拿到同样回报。
要让它必然学出工人的大角度手法，得先标定带料/掉料物理，或直接对姿态加约束。

已知缺陷与待办：起刀处堆料过厚、条带之间有台阶、换条带的过渡轨迹不稳
（锁住刀面 roll 后相邻条带落在不同腕部支解上，J6 直驱转不动，`simulate_transit` 默认关闭）。
材料系数仍全是假设值，标定刮板实验是下一步收益最大的一件事。

## 2026-09-22 整片连续强化学习 v0.3：物理化砂浆 + 真实机械臂

复核 v0.2（提交 9b5716f）后按用户意见迭代，每项改动的原因与「不要改回去」的理由见
`experiments/v0.3/r0/design/2026-09-22_wall_cycle_v0.3.md`（C1–C16），结果见 `experiments/v0.3/r0/records/2026-09-22_wall_cycle_v03/`。
v0.2 用 `CycleConfig(physics='v0.2')` 逐位复现（有测试）。测试 107 → 125（3 跳过）。L1，未连接机械臂。

- 作业区：按「最大圆 → 内接正方形 → 边长 ×0.8」从 Dummy V2 + 抹刀的真实可达范围算出，墙距 33 cm，18.5×18.5 cm（v0.2 为未经推导的 28×7 cm）。
- 砂浆：屈服应力 + 挤压力平衡 + 薄膜挤出 + 重力排流 + 坍落，不含任何角度规则；倾斜是否有利由物理算出（250 Pa 时单刀最佳 15°）。
- 机械臂：每刀在 MuJoCo 上规划（逆解、限位、离墙离桌）并做动力学联合仿真；安全层把不可执行的刀投影到最近的可执行刀。
- 结果：只差俯仰的两个教师，立起浪费 16.5% 对平刀 32.5%；PPO 从平刀起步把俯仰从 3.8° 推到 5.0°、浪费下降，但未超过教师；覆盖 0.22–0.27，任务远未解决。
- 发现：J5 在该墙距接近 −100° 限位，底部向上抹时 20° 以上的下缘先贴墙不可执行；斜向刀次在 5 mm 网格上有混叠（下一步先修）。
- 可视化：`整片连续强化学习可视化.cmd` 改为 MuJoCo 窗口中的真实模型；v0.2 网页回放改名为 `整片连续强化学习可视化_v0.2记录.cmd`。

## 2026-09-22 实验室照片支持的末端工具模型

新增命名配置 `models/tools/lab_20260922.json`：刀片124 mm、尾宽42 mm、肩宽35 mm，渐变椭圆木柄，
按“8-30”刻字使用MINIF08-30候选规格；27 mm净距解释获用户暂时认可，其余未测项明确保留估计。
新增双抱箍安装件仿真设计、独立入口 `实测末端工具仿真.cmd`，基础wall命令支持显式工具配置。
TCP比旧配置外移40.5 mm（含暂定安装高度）；新配置修正J6关节/执行器两层限幅和固定壳体质量归属。
完整回归131个：128通过、3跳过；6个几何/动力学测试通过。L1，无硬件操作。
照片、模型、预览和限制见 `experiments/00_initial_debug/records/2026-09-22_lab_tool/README.md`。
旧wall_cycle/RL/可达表保留原配置；新工具的完整施工训练、材料轮廓一致性和可达性重算尚未完成。

## 2026-09-22 末端 D435 支架（软件会话）

lab 工具变体新增一体延伸支架、45 mm 中心距双 M3 安装接口和末端虚拟相机。
3.2 mm 通孔经过射线验证；相机随动且 TCP 不变；133 测试中 130 通过、3 跳过。
相机/支架质量为估计，D435f 实物适配、碰撞、线缆、手眼标定和打印强度待验证。
证据与说明：[D435 支架实验](../experiments/00_initial_debug/records/2026-09-22_d435_mount/README.md)，L1。

## 2026-09-22 减速器长度修正（软件会话）

用户粗测总安装高度约 28 mm，替换先前 50 mm 估计（均包含顶部转接板）。
当前 lab 变体的抹刀与相机一起回移 22 mm；零位 TCP 为 (310.91,0,324.5) mm。
8 项相关测试及两处位置差验证通过，L1；未进行实体标定。
先前记录中“TCP 比旧配置外移 40.5 mm”仅对应历史 50 mm 配置；当前差值为 18.5 mm。
证据：`experiments/00_initial_debug/records/2026-09-22_reducer_28mm/`。

## 2026-09-22 MINIF8 图纸基准更新（软件会话）

采用240914A图纸：安装面至壳顶26.5 mm，输出凸台1 mm，暂估转接板6 mm，总长33.5 mm。
取代28 mm临时配置；与用户粗测总长仍存在差异，待核对转接板嵌套结构。
9项相关回归通过，L1；TCP零位(316.41,0,324.5) mm，质量与转矩未实测。
证据：`experiments/00_initial_debug/records/2026-09-22_reducer_drawing/`。

## 2026-09-22 整片连续强化学习 v0.5 P1（软件会话）

按照审定方案完成 P1：有限装料板、工具/墙面界面差异、搬运保料状态、横/斜/竖三类刀路、
14 维分层动作、实验室末端工具可达表、行为克隆/DAgger/PPO 训练和原生 MuJoCo 回放。
正式训练含 120 个教师回合、3 轮 DAgger 和 61,440 个 PPO 决策步；最终选用 `ppo_update_50`。

固定 100 场降阶评估中，教师覆盖率 0.253、浪费率 0.138；所选 PPO 覆盖率 0.221、浪费率 0.251。
固定 30 场 MuJoCo 联合仿真复测中，教师覆盖率 0.208、浪费率 0.153；PPO 覆盖率 0.202、浪费率 0.289，
且所有策略成功率均为 0。因此本轮结论是**训练与回放链路完整跑通，但策略训练未达到任务成功阈值**；
不能据此进入实机控制，也不能宣称 PPO 优于教师。原生回放种子 20005 是较易观察动作的失败样本。

修复了新动作维度的行为克隆权重、奖励零分母、MuJoCo 执行器跨回合状态与随机数泄漏。
最终回归 143 项通过、3 项跳过；未连接串口、未发送实机指令。结果见
`experiments/v0.5/p1/records/2026-09-22_wall_cycle_v05/RESULTS.md`，设计变更见
`experiments/v0.5/p1/design/2026-09-22_wall_cycle_v0.5_p1.md`，启动器为 `新一轮抹墙训练MuJoCo回放.cmd`。

## 2026-09-24 J6 末端夹爪连接盖板源文件检索

从用户补充的完整源文件包中确认，照片中的 J6 输出端银色连接件是 35 mm 夹爪电机后盖，
源模型为 `motor35-br-top v164 v1.SLDPRT/STEP`，并有对应 STL。夹爪说明书明确要求拆下该后盖，
用 M3×6 或 M3×8 安装到 J6 减速器输出端；CAD 外形和照片中的四角凸台、非对称开孔一致。
单实体 STEP 包络约 35.026×35.129×8.250 mm，四个主孔为 26×26 mm 中心距。
模型、哈希、预览、排除项与制造前复核限制见 `experiments/00_initial_debug/records/2026-09-24_end_flange_source_search/README.md`。
本项为源文件检索与 CAD 几何核对，无机械臂控制、无打印件实物装配验证。

## 2026-09-24 J6 连接盖板版本更正

用户补充确认当前实物盖板为中部三螺钉孔的改良版。复查归档中的两个 `motor35-br-top` STEP、
SLDPRT、STL、J6 总成和加高法兰后，确认现有可编辑模型均基于早期 v164 v1 孔型；BACK STEP
也没有三孔末端法兰阵列。因此上一条记录中“照片对应源模型”的表述降级为“旧版外形近似且四角接口可参考”。
当前改良版 CAD 未包含在用户提供的源文件中，必须测量三孔接口后重建。证据见
`experiments/00_initial_debug/records/2026-09-24_end_flange_version_audit/README.md`。L1 CAD 审查，无硬件操作。

## 2026-09-24 末端抹刀与D435转接头V1

新增 experiments/00_initial_debug/design/tool_adapter_v1：平底四角26×26参考接口、M3热熔盲孔、双可拆抱箍、一体D435侧支架，含独立STL、STEP装配、参数源、试孔件和中文装配说明。4个CAD有效单实体；STL闭合/非流形检查通过；理想木柄与相机包络无干涉。L1，未验证打印装配或承载，未修改仿真和实机标定。证据：experiments/00_initial_debug/records/2026-09-24_tool_adapter_v1/README.md。

## 2026-09-24 转接头V1.1紧固工具可达性修正

设计位于experiments/00_initial_debug/design/tool_adapter_v1_1。抱箍螺钉各侧外移10 mm、紧固耳加固；4处Ø10直杆/Ø30高位手柄包络检查通过，4份STL流形检查通过。L1，尚未实物试装/承载验证。证据见experiments/00_initial_debug/records/2026-09-24_tool_adapter_v1_1/README.md。

## 2026-09-24 v0.8实验方案审查（分析会话）

核对v0.7训练/回放产物与当前代码，支持局部教师修复方向，但建议先处理评分mask、掉料奖励归因、带余料过渡、教师禁忌可观测性、独立测试与选模门槛，再全量重训。只做静态审查和已有回放数组复算，未启动训练或更改实验源码。详细结论见experiments/v0.8/r0/records/2026-09-24_v08_plan_review/REVIEW.md；证据L1。

## 2026-09-24 wall-cycle v0.8 代码实施（软件会话）

按v0.8方案并吸收同日审查意见实施：局部教师（局部缺料/过厚评分、向外刮平、失败避让、平台期收尾）、教师记忆进入观测（838→1116维）、掉料三本账（搬运/其他掉落/出网格，outside按1/4计，用户决定）、长预算与收尾扣分-2、装料过渡带料物理与FEED_TRANSIT标签、评分掩码显式为40×40格即200mm（与v0.7实际一致）、新验证/测试种子40000/50000、选模需覆盖与边缘覆盖不低于BC的0.9倍、多进程并行（parallel.py，--workers 0为逻辑核数减1）。v0.7配方可逐项复现，旧训练启动器已加--recipe v0.7。单元测试与缩小规模全流程（2进程）通过，2进程评估吞吐1.67倍且逐局一致。尚未正式训练。变更见experiments/v0.8/r0/design/2026-09-24_wall_cycle_v0.8.md，启动器experiments/v0.8/r0/scripts/training/整片抹墙v0.8一键训练(并行+回放).cmd；证据L1。

## 2026-09-24 v0.8计算路径审查（分析会话）
确认当前BC/PPO为纯NumPy CPU实现，正式记录23个worker、总耗时3027.3秒，完整MuJoCo测试1204.3秒。细分网络/采样耗时尚无独立测量，不宣称GPU提速倍数。见experiments/v0.8/r0/records/2026-09-24_v08_compute_audit/README.md，L1。

## 2026-09-24 v0.8边缘精修建议（分析会话）
核对终止记录，联合仿真选中策略20局均由finish/stall终止，未耗尽预算；建议持久缓冲墙面、局部精修技能、早退恢复与分阶段长PPO实验。未修改算法或启动训练。见experiments/v0.8/r0/records/2026-09-24_v08_edge_refinement_review/README.md。

## 2026-09-24 v0.9完整实验方案（待审查）
方案见experiments/v0.9/r0/design/wall_cycle_v0.9_边缘精修完整实验方案.md，包含持久越程区、缺陷精修技能、有限恢复、长PPO及GPU后端独立验收。未实施未训练。交付记录experiments/v0.9/r0/records/2026-09-24_v09_experiment_plan/README.md。

## 2026-09-24 v0.9修订1.1（待审查）
新增experiments/v0.9/r1.1/design/wall_cycle_v0.9_r1.1_教师初始化与自主强化学习.md。教师仅初始化，取消90%教师成绩准入，加入自由参数轨迹与纯/衰减模仿PPO对照，同预算验证超越教师。旧稿保留，未实施未训练。

## 2026-09-24 观测来源核对
当前墙面输入为D435Proxy对高度场加噪声再池化，非渲染深度；actor另含余料/损失真值与真实质量改善派生量。v0.9实施需补观测契约和去真值泄漏验证。证据experiments/v0.8/r0/records/2026-09-24_observation_source_audit/README.md，静态L1。

## 2026-09-24 教师示教可视化检查
启动v0.8教师已有轨迹1倍速原生回放，保存关键帧；发现第35决策最高合格覆盖57.375%回落至43.75%，以及192个带余料倾斜FEED_TRANSIT帧。单局L1，不是新训练。见experiments/v0.8/r0/records/2026-09-24_teacher_visual_review/README.md。

## 2026-09-24 教师闪现检查
已有回放179处相邻帧关节差>10度，最大48.34度；记录终点无真实时间、固定50ms播放且SCAN使用名义姿态。当前BC/PPO非视频训练，显示跳帧不直接入训练，但材料段末姿态积分等物理近似仍需核验。见experiments/v0.8/r0/records/2026-09-24_teacher_replay_discontinuity/README.md。

## 2026-09-24 v0.9修订1.2完整规划（待审查）
新增experiments/v0.9/r1.2/design/wall_cycle_v0.9_r1.2_完整实验规划与通俗说明.md，25节说明训练分工、观测契约、真实时间记录与材料积分、A0/A1动作和自主PPO实验。替代旧稿实施安排，旧文档保留。未实施未训练。交付记录experiments/v0.9/r1.2/records/2026-09-24_v09_r12_plan/README.md。

## 2026-09-25 v0.9 r1.2 实施中（软件会话，L1）
已添加隔离的观测桥、连续时间执行器、资源监测、CUDA混合策略骨架、A0后端pilot和时间戳回放。CUDA实际计算与3项回归通过；A0仅完成4096步后端验收，完整A1与正式多种子实验尚未完成。长程P0验证进行中。状态与待办见 experiments/v0.9/r1.2/records/2026-09-25_v09_implementation/README.md；不得将pilot视为完成的v0.9实验。

v0.9同次补充：60决策P0比较与4096步A0后端pilot均已结束；正式A1尚未启动。完整实验任务每30分钟自动跟进，继续按门槛推进。


### 2026-09-25 v0.9完整实验持续执行（尚未完成）

A1统一CONTACT、测量观测、400示教、40有效DAgger、混合策略与GPU PPO已实现。BC1执行率88.83%通过；R1/R2累计100轮和后续三种子/对照由`experiments/v0.9/r1.2/runs/v09_r12/campaign_001`持续管理。A0初始化尚需验收；最终独立测试和结论未完成。证据见`experiments/v0.9/r1.2/records/2026-09-25_v09_full_campaign/README.md`。仅L1仿真。

补充：A0初始化BC0_seed11_dagger_003已通过，216次接触请求中可执行率92.59%。RL1首种子累计100轮已完成，后续种子由campaign继续运行；不代表最终质量验收通过。

### 2026-09-25 v0.9 r1.2 可用组交付（L1）
- 主训练8组×400轮、2项消融×100轮；1980场验证/独立测试和5份原生回放已生成并核验。
- 质量未达标，straight消融门槛失败缺失；不是完整矩阵成功，无稳定超越教师证据。
- 结论：experiments/v0.9/r1.2/runs/v09_r12/campaign_005_a0_log_recovery/DELIVERY.md。
- 回放：launchers/replay/v09_final_*.cmd；审计：delivery_audit.json。

2026-09-25：v0.9原生播放器修复逐帧整段NPZ解压，加入仅显示的关节插值。五份回放性能证据见experiments/v0.9/r1.2/records/2026-09-25_v09_replay_performance；原训练记录不变。

### 2026-09-25 实验目录统一整理
所有版本按experiments/vX.Y/修订分组，初期调试在00_initial_debug；根launchers撤销。13299原文件迁移核验、13模型哈希一致，入口与回放回归通过。导航experiments/README.md；当前experiments/v0.9/r1.2/README.md。迁移证据experiments/_organization/2026-09-25。

### 2026-09-25 v0.10 r0实验设计稿
装料成本/余料观测、长刀与连续接触、自主RL及GPU等价迁移的分阶段方案已写入experiments/v0.10/r0/design。当前仅设计，未实现或启动训练；详见experiments/v0.10/r0/README.md。

### 2026-09-25 v0.10 r0 实施进展（L1）
已完成8496个名义分支与43个MuJoCo候选检查；P0未通过，不启动PPO。GPU压力求解核数值验证通过，256批量CUDA Graph约1.005ms，尚非全环境/训练加速。详见experiments/v0.10/r0/records/P0_AND_GPU_INITIAL_RESULTS.md。无后台训练、无实体指令。

### 2026-09-26 v0.10 r1 多步自主修整方案（待审查）
完整方案：experiments/v0.10/r1/design/v0.10_r1_多步自主修整与GPU批量训练完整方案.md。取消单刀质量准入，保留数值/执行契约；开发128、正式512更新/种子，多步终态回报与连续接触，端到端GPU效率验收。本次仅文档，未启动r1训练。

### 2026-09-26 v0.10 r1 实施开始（L1，未启动PPO）
共享CNN/七操作/时间GAE与多步奖励12测试通过，候选材料CPU编译加速12直线+30旋转等价。扩大100单段+50条100动作的材料门槛仍失败，需要修旋转网格覆盖；不因单刀质量差拦RL。记录experiments/v0.10/r1/records/IMPLEMENTATION_PROGRESS.md，v0-10-r1自动跟进继续实现，当前无后台训练。

- v0.10 r1 软件续做：新增旋转多边形面积重映射底层模块，16项离线测试通过（L1）；尚未接入材料完整G0，不代表训练已开始。证据 experiments/v0.10/r1/records/OVERLAP_GEOMETRY_PROGRESS.md。

- v0.10 r1 持续库存候选（L1）：20项测试通过，12路径局部收敛诊断完成，未通过完整G0；无PPO。记录 experiments/v0.10/r1/records/CONTACT_INVENTORY_PROGRESS.md。

- v0.10 r1 压力库存耦合候选（L1）：24项测试通过，6条三档分辨率诊断完成；完整G0未通过，PPO未启动。记录 experiments/v0.10/r1/records/PRESSURE_INVENTORY_PROGRESS.md。

- v0.10 r1（L1）：分阶段诊断完成，新增PPO更新原语，27测试通过；非正式训练。记录 experiments/v0.10/r1/records/STAGE_ERROR_AND_PPO_PROGRESS.md。

- v0.10 r1（L1）：出入区更新顺序修正候选，29测试通过；最终数值收敛仍失败，料堆沉积误差较大，无PPO。记录 experiments/v0.10/r1/records/ORDERED_INVENTORY_PROGRESS.md。

- v0.10 r1（L1）：六路径细分辨率显示近一阶收敛，但均未达局部厚度门槛，PPO未启动。记录 experiments/v0.10/r1/records/FINE_RESOLUTION_PROGRESS.md。

- v0.10 r1（L1）：守恒外推候选未取得高阶/效率收益，31测试通过但G0未过，无PPO。记录 experiments/v0.10/r1/records/EXTRAPOLATION_PROGRESS.md。

- v0.10 r1（L1）：通量分步v4六路径局部厚度收敛通过，33测试通过；待扩大G0与完整G1，无PPO。记录 experiments/v0.10/r1/records/SPLIT_INVENTORY_PROGRESS.md。

- v0.10 r1（L1）：v4扩大100单段+50条100操作材料审计运行中，12workers+5秒资源监控。记录 experiments/v0.10/r1/records/EXPANDED_SPLIT_AUDIT_STARTED.md；不是PPO。

- v0.10 r1：扩大材料审计单段69/100通过，长序列继续；新增独立PPO rollout调度，38组件测试通过。G0未过、PPO未启动。证据 experiments/v0.10/r1/records/ROLLOUT_INTERFACE_PROGRESS.md。

- v0.10 r1（L1）：最大失败单段已定位抬刀沉积放大工具库存误差，独立重放逐元素一致。记录 experiments/v0.10/r1/records/SPLIT_FAILURE_STAGE_DIAGNOSIS.md；扩大审计继续，无PPO训练。

- v0.10 r1（L1）：独立v5库存自适应积分候选41组件测试通过，case13有界诊断PID6180运行，旧审计33596继续。证据 ADAPTIVE_INVENTORY_STARTED.md；无正式训练。

- v0.10 r1（L1）：v5两档有界诊断失败；v6中点交换单例最大厚度差0.061914mm未改善，44组件测试通过。记录 MIDPOINT_GEOMETRY_RESULT.md；无正式训练。

- v0.10 r1（L1）：组件隔离确认几何交换误差；v7同例最大差0.04716mm仍未过门槛，46组件测试通过。记录 COUPLED_MIDPOINT_RESULT.md；旧审计继续，无PPO。

- v0.10 r1（L1）：v7三分辨率和运动分解均显示几何交换仍近一阶，最细差0.024336mm未达门槛。证据 V7_GEOMETRY_ORDER_DIAGNOSIS.md；未启动PPO。

- v0.10 r1（L1）：解析确认端点扫过区域漏算；独立平移边界几何核49组件测试通过，未接旋转和材料交换。证据 TRANSLATION_BOUNDARY_RESULT.md；无PPO。

- v0.10 r1（L1）：旋转边界流量与局部闭合诊断完成，52组件测试通过；尚未接材料库存，不代表G0或训练完成。记录 ROTATING_BOUNDARY_RESULT.md。

- v0.10 r1 L1进展：冻结边界库存交换56组件测试/32独立冻结姿态通过，移动容量及完整材料未接入，G0未过、PPO未开始。证据 experiments/v0.10/r1/records/BOUNDARY_EXCHANGE_COMPONENT.md。

- v0.10 r1（L1）：移动面积解析交换62组件测试通过，连续零间隙case13细化差0.002198mm；尚非完整物理/G0，无PPO。证据 MOVING_AREA_PICKUP_RESULT.md。

- v0.10 r1（L1）：有限工具库存联合交换67组件测试通过；轴向/斜向诊断完成，旋转新生接触支持仍失败。记录COUPLED_BOUNDARY_RESULT.md，无PPO。

- v0.10 r1 L1：自适应积分修复简单旋转诊断，70组件测试通过；原组合轨迹PID40724在测，仍无完整G0/PPO。记录ADAPTIVE_BOUNDARY_PROGRESS.md。

- v0.10 r1 L1：全局速度重测组合轨迹仍失败；v8压力接口空墙初测通过、有料墙非负校验失败。73组件测试，无G0/PPO。记录GLOBAL_RATES_AND_PRESSURE_INTEGRATION.md。

- v0.10 r1 L1：初始拾料修复，短压力路径完成；越界事件预分段通过原失败区间，完整组合PID40456在测。76组件测试，仍无完整G0/PPO。记录INITIAL_PICKUP_AND_VERTEX_EVENTS.md。

- v0.10 r1 L1：CPU边界局部加速已对照；完整压力粗步长case13差0.06330mm。原50/25μm对照PID31416在跑。77组件测试，无G0/PPO。记录COMPILED_BOUNDARY_AND_CASE13_PRESSURE.md。


2026-09-26 L1: case13 boundary v8.1 passes fixed-spacing single-case numerical criteria; expanded_boundary_001 (100+50x100) running PID32992. Formal RL not started. See v0.10/r1 records/CASE13_FIXED_SPACING_AND_EXPANDED_BOUNDARY.md.

L1 v0.10r1: expanded boundary audit has zero-area inventory exceptions (cases1/18/21); independent case21 trace PID35368 running. No formal RL. See EXPANDED_BOUNDARY_ZERO_AREA_TRACE.md.

L1 v0.10r1: bounded-edge v8.2 candidate, 3 targeted tests passed; original failure regressions PID31516 running, see BOUNDED_EDGE_CANDIDATE.md. No formal RL.

L1 v0.10r1: edge three-case regression passes; expanded v8.2 PID6544 running. Empty-map dtype candidate unit test passes, sequence trace pending. No formal RL. See EDGE_PASS_AND_EMPTY_MAP_DTYPE.md.

L1 v0.10r1: confirmed empty-map dtype failure, stationary geometry identity tests pass; sequence0 four-action diagnostic PID27260 running. See STATIONARY_CONTINUATION_FIX.md. No formal RL.

L1 v0.10r1:4-step continuation passed, v8.2 singles99/100; v8.4 long-sequence audit40668 and case76 pre/post-lift trace9672 running. No formal RL. See CONTINUATION_EXPANSION_AND_CASE76.md.

L1 v0.10r1:83 component tests passed plus1temporal test; case76 error dominated by lift bead inventory. v8.5 diagnosticPID32884 and v8.4 long sequences40668 running. No formal RL. See CASE76_LIFT_AND_TEMPORAL_PRESSURE.md.

L1 v0.10r1:temporal-control candidate did not fix case76; isolated pressure subcycling PID5688 and v8.4 long sequences40668 running. No formal RL. See PRESSURE_SUBCYCLE_DIAGNOSTIC.md.

L1 v0.10r1:pressure subcycling failed precision; matched-time tracePID2408 and long sequences40668 running. Resource monitoring gap identified, single5sec monitor13564 restored. No formal RL. See SUBCYCLE_FAILURE_AND_DIVERGENCE.md.

L1 v0.10r1:matched-time trace indicates smooth carried-inventory divergence; quarter-control exit-gap candidate2tests pass, case76PID29260 running. Long sequences40668 continue; no formal RL. See QUARTER_GAP_DIAGNOSTIC.md.

L1 v0.10r1:quarter-gap candidate failed precision; bounded whole-contact step-doubling candidate2tests pass, diagnosticPID39808 running, long sequences40668 continue. No formal RL. See CONTACT_ERROR_CONTROL_DIAGNOSTIC.md.

L1 v0.10r1:first4/50 full100-step sequences pass; bounded contact sampler4tests pass, not full executor. Adaptive case76PID39808 pending, long-sequence manager40668 continues. No formal RL. See LONG_SEQUENCE_AND_CONTACT_SAMPLING.md.


2026-09-26 L1 v0.10r1: v8.8 adaptive contact case76 failed depth6 at both resolutions; v8.9 pressure precision diagnostic running PID17240, no tolerance relaxation. Four long sequences passed; G0 incomplete and formal RL not started. See experiments/v0.10/r1/records/PRESSURE_PRECISION_DIAGNOSTIC.md.

L1 v0.10r1: v8.9 coarse case76 fails at same2314; pressure precision hypothesis unsupported as sole cause. Inventory-level rejection tracePID26144 running, instrumentation test passed; fine diagnostic17240/long sequences40668 continue. No formal RL. See CONTACT_REJECTION_TRACE.md.

L1 v0.10r1: local rejection dominated blade/slump ordering; v8.10 continuous wall-cap candidate reduces local discrepancy~146x with ledger closed;2tests pass. Full case76 PID7244 pending, long sequences40668 continue. No formal RL. See WALL_CAP_ORDERING_DIAGNOSTIC.md.

L1 v0.10r1: v8.11 fixed-step wall-cap originalcase76 passes0.00313509mm<0.01mm, ledger/nonnegative pass. Full same-version100+50x100 auditPID22620 running8workers; old4worker audit40668 retained. Formal RL not started. See FIXED_CAP_EXPANDED_AUDIT.md.

L1 v0.10r1: v8.11 audit89singles pass so far, long sequences pending. Offline joint path preflight5synthetic tests pass, no MuJoCo/G1 claim. Active audit22620 and historical40668 unchanged. No formal RL. See JOINT_PATH_PREFLIGHT_COMPONENT.md.


2026-09-26 用户暂停v0.10r1：验证任务及监测已停止，自动跟进PAUSED。正式RL未启动。新版本单段100/100通过，长序列未完成；详见experiments/v0.10/r1/records/PAUSE_AND_RETROSPECTIVE_20260926.md。没有用户恢复指令不得重启。

2026-09-27 用户授权清理：删除10215个中间文件，释放约464.76GiB；22个最终/对照模型哈希核验通过，最终回放和关键指标保留。详见experiments/_organization/2026-09-27_cleanup/README.md。旧中间路径可能失效，训练保持暂停。
