# Wall-cycle v0.5 P1 最终结果

本目录记录纯软件 L1 实验。未连接串口，未控制实体机械臂。

- 训练前 100 场景门禁：`raw/pretraining_gates_100.json`
- 正式训练：`raw/p1_training/training.json`
- 最终有效 MuJoCo 复核：`raw/cosim_recheck_deterministic.json`
- 原生回放：`raw/native_replay_seed_20005_bare/rollout.npz`
- 关键帧：`reports/native_replay_frames/`
- 变更说明：`docs/changes/2026-09-22_wall_cycle_v0.5_p1.md`

最终结论：P1 学会了朝上搬运和多方向动作，但没有完成整墙任务，也没有在完整 MuJoCo 联合仿真中超过教师。`training.json:test_cosim` 与 `raw/cosim_recheck_after_executor_reset.json` 受缓存 IK 状态影响，保留作失败证据；以 `raw/cosim_recheck_deterministic.json` 为准。
