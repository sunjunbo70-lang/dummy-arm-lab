# v0.8 CPU/GPU计算路径审查

分析会话，L1代码与已有日志审查；未跑性能基准、未重训、未修改算法。

证据：dummy_loop/wall_cycle/train.py导入wall/ppo.py；后者为纯NumPy的MLP前向、手写反向、Adam、PPO；无CUDA训练路径。parallel.py通过spawn多进程并行，training.json记录workers=23。传统MuJoCo调用和自定义NumPy砂浆环境仍在CPU。

outputs/wall_cycle/v08_manual/training.json总耗时3027.3秒。stage_elapsed_s是从启动累计时间，差分后：
- 教师数据275.2秒（4.59分钟）
- BC及DAgger阶段915.0秒（15.25分钟），包含数据采集，不是纯网络训练
- BC验证、价值预热、PPO采样/更新与周期验证415.0秒（6.92分钟）
- 新测试125.9秒（2.10分钟）
- 完整MuJoCo测试1204.3秒（20.07分钟，39.78%）
- 历史测试91.8秒（1.53分钟）

网络隐藏层256×256，PPO minibatch128，epochs4，50×2048=102400决策步。当前没有分别计时采样、网络更新、教师评分、IK和材料计算，不能得出各内核精确占比或GPU加速倍数。CPU高占用不等于低效率；未用GPU首先是实现选择，而不是驱动故障的证据。

建议先分项计时；将BC/PPO迁移PyTorch CUDA并对照现有实现，保留CPU环境并行，避免小批逐步往返GPU。材料/教师搜索向量化并基准测试workers数。若后续需GPU环境，需迁移自定义材料、奖励、观测、重置及执行路径；仅换MuJoCo后端不足以迁移当前table训练环境。检查数值与质量守恒回归。无需先升级显卡。

参考：https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
参考：https://mujoco.readthedocs.io/en/latest/mjx.html
