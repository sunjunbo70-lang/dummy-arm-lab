# v4扩大材料数值审计已启动

L1软件会话。执行器dummy_loop.wall_cycle_v10.r1.expanded_split_audit，源码f127c1b；候选v4物理保持不变。运行目录runs/expanded_split_001。启动器PID35460，实际管理器PID33596（以manifest和实际进程为准，PID可能退出后复用）。12工作进程，100单段+50条100操作序列，采样间距50/25微米；采样数量同时考虑平移距离和刀面半径乘旋转角，纯旋转不被漏掉。

检查逐操作全墙最大格厚差、评分区域RMSE/coverage差、材料账本及非负性。阈值：单段0.01mm、序列0.05mm、RMSE0.02mm、coverage0.005、账本0.01mL，沿用既有严格检查。生成的连续接触段继承上一真实材料姿态和力/倾角；补料前抬刀，调用旧feed模型并记入外部供应。搬运姿态/机器人执行/完整观测尚未接入，因此即使材料数值通过也不是完整环境G0/G1。

已完成入口烟测：1个纯旋转单段通过（厚度差0.000754535mm），1个仅补料序列通过；仅验证入口与账本，不是全预算。旧烟测summary的material_numerical_gate字段可能为true，但只是两个烟测案例，不能解释为完整预算通过。随后执行器已新增required_budget_complete标记，非100/50/100配置不允许宣称完整材料数值门槛通过。

正式审计首个单段已完成且通过，其余进行中。不要重复启动，不在当前运行时修改材料核心。每案例逐操作JSONL可查看长序列进度；异常进入cases.jsonl而不是被忽略。最终summary只有所有任务返回才写入。保留所有失败。单条有质量下降不会停止该审计或构成RL质量门槛。

使用已有wall_cycle_v09.telemetry，仅一个监控进程，5秒resources.jsonl；stdout/stderr在runs/expanded_split_001.stdout.log和.stderr.log。源码快照、hash、git差异均已保存在运行目录。单例JSON/最终NPZ/动作recipe保留；不是PPO训练日志。

下一次先读manifest、cases.jsonl、*_steps.jsonl、stderr、resources和实际管理器/子进程。若仍运行可推进独立的环境/训练器代码，但不要更改本审计加载的物理定义。完成后按实际100+50结果决策，不以已通过六路径代替全矩阵。
