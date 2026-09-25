# v0.9 / r1.2

按以下入口查看方案、原始记录、训练数据与可视化。旧目录名保留日期和语义，失败与中止记录不删除。

## 设计方案

- [original_submissions](design/original_submissions/)
- [wall_cycle_v0.9_r1.2_完整实验规划与通俗说明.md](design/wall_cycle_v0.9_r1.2_完整实验规划与通俗说明.md)

## 实验记录与诊断

- [2026-09-24_v09_r12_plan](records/2026-09-24_v09_r12_plan/)
- [2026-09-25_v09_full_campaign](records/2026-09-25_v09_full_campaign/)
- [2026-09-25_v09_implementation](records/2026-09-25_v09_implementation/)
- [2026-09-25_v09_replay_performance](records/2026-09-25_v09_replay_performance/)

## 训练过程与最终输出

- [v09_r12](runs/v09_r12/)

## 本版本运行入口

- [training](scripts/training/)
- [visualization](scripts/visualization/)

## 本轮最终结果

[最终结论](runs/v09_r12/campaign_005_a0_log_recovery/DELIVERY.md) · [全部指标](runs/v09_r12/campaign_005_a0_log_recovery/REPORT.md) · [原生渲染预览](runs/v09_r12/campaign_005_a0_log_recovery/native_replay_preview.png)

**质量未达标；straight消融缺失。**

[RL回放（初始已涂层，边缘修整）](scripts/visualization/v09_final_RL.cmd) · [教师回放](scripts/visualization/v09_final_Teacher.cmd) · [BC回放](scripts/visualization/v09_final_BC.cmd) · [最差案例](scripts/visualization/v09_final_Worst.cmd)

共享实现：dummy_loop/wall_cycle_v09；原始阶段/恢复目录全部保留，不能把失败目录计作成功。
