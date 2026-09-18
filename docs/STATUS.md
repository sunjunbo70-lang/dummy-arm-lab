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
