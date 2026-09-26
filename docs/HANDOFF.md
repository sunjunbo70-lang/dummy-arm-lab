# 跨电脑开发交接（2026-09-27）

## 当前状态

跨电脑接续分支：main（主分支）。当前不是已完成施工任务的产品。全部仿真实验仅为 L1 软件证据。
v0.9 已执行可用组训练和评估，但 straight 消融未过门槛，矩阵不完整，施工质量未达标。
v0.10 r1 已暂停，正式 RL 未开始；新材料单段100/100通过，长序列验证未完成。不得自动重启训练。
详见 [暂停复盘](../experiments/v0.10/r1/records/PAUSE_AND_RETROSPECTIVE_20260926.md)。

## 文件入口

- [模型索引](MODEL_INDEX.md)：22个正式实验最终/对照模型；[机器可读清单](retained_models.json)含 SHA256。
- [实验目录](../experiments/)：按大版本/修订查找 design、records、runs、scripts。
- [v0.10实现进度](../experiments/v0.10/r1/records/IMPLEMENTATION_PROGRESS.md)：历史过程记录；最后状态以暂停复盘为准。
- [清理记录](../experiments/_organization/2026-09-27_cleanup/README.md)：删除了周期检查点和原始rollout，不能恢复任意历史轮次。
- [项目状态](STATUS.md)、[第三方资源](../THIRD_PARTY.md)。

## 新电脑安装

先安装 Git、Git LFS、Python 3.12。以下命令在仓库根目录执行：

```powershell
git lfs install
git lfs pull
py -3.12 -m venv .venv-loop
.venv-loop\Scripts\python -m pip install -r requirements/core.txt
```

学习模块另需 PyTorch，数值编译需 numba，资源监测需 psutil；依据新电脑 CUDA/驱动安装匹配的 PyTorch。requirements/loop-observed-freeze.txt 是原机实际环境快照，不是跨平台锁文件。不要直接复制旧虚拟环境。

## 查看已有结果

运行 experiments/v0.9/r1.2/scripts/visualization/v09_final_RL.cmd，或在根目录执行：

```powershell
.venv-loop\Scripts\python -m dummy_loop.wall_cycle_v09.replay experiments/v0.9/r1.2/runs/v09_r12/campaign_005_a0_log_recovery/records/RL1_seed33/60090/trajectory.npz
```

这是记录回放，不是重新运行策略，也不代表完整作业达标。最终RL选模为RL1_seed33的0320.pt，详见模型索引。同一records目录含教师、BC、最好/最差等记录。

## 后续开发顺序

1. 阅读暂停复盘和现有失败证据，确认材料数值验证尚未完成的范围。
2. 在新运行目录继续开发/验证，不覆盖旧结果，不凭单段通过启动正式大训练。
3. 完成连续接触执行器、环境/PPO集成与端到端效率验证后，再决定恢复训练。
4. 新实验只长期保留最终选定模型、关键指标及代表性回放；调试检查点需设置保留数量。

## 跨电脑限制

项目内运行入口使用相对路径；历史manifest和日志可能仍记录D:/VLA原机绝对路径，不保证旧campaign可直接恢复。加载最终模型须使用当前仓库根目录重新定位路径。
models/studio_meshes和studio_source遵循THIRD_PARTY.md，不随本仓库发布；依赖它们的特定功能需自行获取资产。基础V2仿真资产随仓库提供。
本次没有在第二台机器实际验证CUDA、GUI和驱动，不把原机检查视作跨平台验收。

## 新增方案与本次检查

[v0.10 r2](../experiments/v0.10/r2/README.md)是待审查设计稿，未实现、未运行。
2026-09-27原机L1检查：最终RL检查点CPU读取通过，2343帧回放读取通过，MuJoCo场景构建通过（6关节）；暂存普通Git对象无超过100MiB项。尚未在新电脑安装及启动GUI验收。
