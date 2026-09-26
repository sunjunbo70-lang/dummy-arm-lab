# v0.10 r1 实施进展（2026-09-26）

用户已批准完整实验。当前实际状态：**实施进行中，PPO尚未启动**，不是完整实验交付。L1软件会话，无实体指令。状态机见../runs/implementation_state.json。自动跟进v0-10-r1每30分钟继续未完成工作，重要变化才通知；不是后台训练进程。

## 已做工作

1. 确认没有其他Python训练占用，保留工作树原有用户修改。按本修订建立独立runs，不覆盖r0。
2. 新增r1/actions.py：7操作、连续19维路径、续接起点及首段姿态/力/速度继承，接触状态mask，不按施工质量强制选动作。
3. 新增returns.py：有限时间gamma=1、终态势0、真实成本、按时间的GAE。暂时变差后修复与直接到达同终态的奖励一致；循环破坏不会刷净奖励。rollout截断与真正终止分开。
4. 新增policy.py：共享CNN+标量融合、操作/连续参数/长度分箱策略和critic。原始动作、混合分箱与变换概率保留；未执行BC或PPO。test_r1_contracts/test_policy合计12项通过，包括CPU/GPU前向一致及共享编码器梯度。
5. 新增候选材料数值修复，独立physics_version，不修改历史mortar.py。浮点边界索引snap单独实验没有解决问题；按实际位移缩放刀前料返回与挤出通量，改善轴向路径收敛，但未通过扩大检查，不能当正式物理冻结。
6. 虚拟环境安装Numba0.67.0、llvmlite0.49.0。compiled_material.py编译候选接触循环，未使用fastmath。12轴向和30旋转路径与Python参考逐数组对照通过。预热后1001子步约0.02s，对应Python约0.88–0.91s；这是局部CPU内核加速，不是端到端训练加速。

## 原始证据

|目录|含义|
|---|---|
|resolution_001|旧物理0.625至0.078125mm；12个代表案例最细比较1/12通过|
|resolution_002_index_snap|只消除极小浮点边界抖动；仍1/12|
|resolution_003_displacement|距离通量候选，误差下降但当前采样未通过|
|resolution_004_fine_displacement|进一步9.765625与4.8828125微米比较，12轴向代表案例12/12通过；不是完整G0|
|compiled_001|12案例Python/Numba参考等价与局部耗时，首次编译预热单列|
|compiled_rotation_001|30旋转案例参考等价，30/30通过|
|material_gate_001|100单段+50条100动作的材料数值检查，已全部执行完，门槛失败；见summary.json|
|footprint_001|3案例旋转、固定刀角、轴向对照与完整墙面NPZ，定位剩余误差|

material_gate_001下manifest.json是启动时原始状态，最终以completion.json和summary.json为准；不覆盖原始启动记录。

## 尚未解决的问题及正确解释

扩大检查不是在要求“每刀改善”。它比较同一个动作在两种极细积分间距下的材料结果，不比较施工优劣。多步序列全部未通过数值门槛；没有理由将这些结果称作RL失败，因为还没有RL。

footprint_001示例：案例0旋转路径最大格厚差0.445mm、固定角0.0464mm、轴向0.00362mm；案例17对应约2.000/0.00606/0.00137mm。误差集中于少量网格，部分在目标区内，不能靠平均RMSE较小掩盖。该对照还改变了轴向组的位置，所以是定位线索，不是严格纯旋转因果证明。编译/参考旋转等价通过，因此不是仅由编译移植造成。

需要进一步处理最近邻刀面覆盖、墙格归属和转动期间的保守重映射。不能只删除旋转动作、放宽误差标准或用通过的12条轴向轨迹宣称全G0通过。候选材料通量按位移缩放属于离散动力学变更，不是已校准的新水泥物理；保留为候选，未经完整审计不用于正式训练。

## 后续推进次序

1. 修复旋转/斜向覆盖映射；每个候选单独版本、参考对照、守恒与收敛。
2. 实现实际CONTINUE/LIFT机械臂执行与G1，不能仅靠动作解码器声称跨段运动已实现。
3. 接入完整观测/余料估计/环境/时间终止；冻结CPU版本后移植完整GPU环境。现有GPU只有策略和旧局部核能力。
4. G0/G1通过后立即进入已批准开发预算，不再要求成功单刀；随后正式矩阵、独立评估、完整原生回放。
5. 资源策略：CPU编译批处理加GPU网络，端到端测量后决定CPU/GPU材料后端。不能人为制造占用率；CPU局部加速与GPU整轮加速分开。

目前没有运行中的PPO。已有短测保存逐案例参数、耗时、数值差和必要墙面数组；尚无训练5秒资源日志，因为没有训练。本次数值检查也没有完整5秒资源时序，不能以单案例计时替代资源占用率。后续正式执行器必须接入既有telemetry，避免重复监测。不得在汇报中说本轮训练已跑完或已启动后台训练。

## 续做：旋转面积映射
新增底层守恒覆盖模块，16项合同/几何测试通过。详见 OVERLAP_GEOMETRY_PROGRESS.md 和 runs/overlap_geometry_001；尚未接入完整材料动力学，G0仍未通过，PPO未启动。

## 续做：持续接触库存
新增出入足迹面积及守恒库存候选，20项测试通过；12条路径细化误差均下降，但最大仍0.225317mm，完整G0未通过。参见CONTACT_INVENTORY_PROGRESS.md与runs/inventory_001/implementation_state.json。无后台训练。

## 续做：压力与库存耦合
候选area_pressure_candidate.v1已接压力/挤出/余料/抬刀，24项测试通过；6条路径细化均改善，最大差仍0.228280mm，非完整G0。详见PRESSURE_INVENTORY_PROGRESS.md及runs/pressure_inventory_001/implementation_state.json。无后台训练。

## 续做：分阶段误差与PPO更新原语
同6路径分阶段对照定位到刀上/料堆误差经抬刀沉积传到墙面；新增PPO单minibatch优化，27测试通过。完整G0/G1及完整训练器仍未完成，无后台训练。详见STAGE_ERROR_AND_PPO_PROGRESS.md及runs/stage_error_001/implementation_state.json。

## 续做：出入区顺序修正
候选v2修正同一步退出沉积重吸问题，29测试通过；对照显示料堆沉积误差仍主导，最终最大差0.253357mm（未改善），完整G0/G1仍未过。详见ORDERED_INVENTORY_PROGRESS.md及runs/ordered_inventory_001/implementation_state.json。无后台训练。

## 续做：细分辨率收敛与成本
候选v2六路径801/1601/3201点全部完成，经验阶数0.994–1.000，最细比较0.01359–0.03157mm，六例仍未达0.01mm。单路径3201点13.57–15.77秒，需改积分而非仅缩步。详见FINE_RESOLUTION_PROGRESS.md及runs/fine_resolution_001/implementation_state.json；无后台训练。

## 续做：守恒外推候选失败
候选v3完成六路径对照，4200步仅22次接受外推，仍近一阶且未过门槛；31测试通过不代表G0通过。局部墙格负库存导致回退，未放宽非负约束。详见EXTRAPOLATION_PROGRESS.md、runs/extrapolation_001/implementation_state.json。无后台训练。

## 续做：通量分步候选局部通过
候选v4分步SSPRK2，33测试通过；六路径801/1601点厚度差最大0.00713mm，局部6/6通过，整体仍近一阶，非完整G0/G1。新旧极限估计接近但非证明。详见SPLIT_INVENTORY_PROGRESS.md及runs/split_fine_001/implementation_state.json。无后台训练。

## 当前正在执行：扩大材料审计
已启动expanded_split_audit，100单段+50条100操作，12workers，实际manager PID33596（manifest核对），runs/expanded_split_001。已有5秒资源监控，不重复启动或修改运行物理。详见EXPANDED_SPLIT_AUDIT_STARTED.md及该run的implementation_state.json；PPO仍未启动。

## 续做：扩大审计单段失败与 PPO 调度
100单段69通过31失败，长序列审计继续，当前候选不能通过完整G0。独立rollout/时间GAE/整批归一化/四epoch与KL停止接口补齐；38组件测试通过，不代表训练完成。详见ROLLOUT_INTERFACE_PROGRESS.md及runs/rollout_interface_001。无PPO训练。

## 续做：最大失败案例抬刀归因
独立重放single13与原结果逐元素一致：抬刀前最大差0.00820mm，抬刀后0.06179mm；最大误差格中余料堆沉积贡献0.04837mm、刀面贡献0.01171mm。详见SPLIT_FAILURE_STAGE_DIAGNOSIS.md及runs/split_failure_stage_001。扩大审计继续，G0未过、PPO未开始。

## 续做：v5工具库存自适应积分候选
新增独立v5局部误差控制，含预计抬刀沉积，41组件测试通过。case13两档预算有界诊断已启动，实际PID6180，runs/adaptive_inventory_001；旧v4扩大审计33596继续。详见ADAPTIVE_INVENTORY_STARTED.md。均不是PPO；G0尚未通过。

## 续做：v5失败与v6中点几何交换对照
v5两档预算均耗尽，未完成case13，见ADAPTIVE_INVENTORY_FAILED.md。独立v6局部解析二阶/守恒测试通过，但同例抬刀厚度差0.061914mm，未改善v4的0.061789mm。44组件测试通过不代表G0通过。详见MIDPOINT_GEOMETRY_RESULT.md和runs/midpoint_geometry_001。旧扩大审计继续，无PPO。

## 续做：组件隔离与v7联合中点交换
内部通量规定输入下约二阶；纯几何交换case13抬刀差0.04868mm，证明几何侧也有误差。v7联合中点交换完整同例差0.04716mm，比v4降约24%但未过0.01mm门槛。46组件测试通过；原审计已完成12条长序列（10通过2失败）仍运行。记录COMPONENT_ISOLATION_RESULT.md、COUPLED_MIDPOINT_RESULT.md，runs/component_isolation_001、coupled_midpoint_001。无PPO。

## 续做：v7仍是一阶几何交换
v7纯几何50/25/12.5μm抬刀相邻差0.048677/0.024336mm，仍一阶且未过门槛。固定方向移动/原地旋转/组合三种轨迹200/400/800区间均近一阶，不能只归因旋转或压力。详见V7_GEOMETRY_ORDER_DIAGNOSIS.md，runs/coupled_geometry_isolation_001、geometry_probes_001。两个诊断已结束，旧审计33596继续；无PPO。下一步检验端点覆盖流量分配与运动边界积分。

## 续做：证实端点漏算与平移边界几何核
解析基准证明端点覆盖差漏掉斜向角部步内先入后出的区域，累计漏面积|DxDy|/N。新增平移边界核消除此纯几何误差，4种分段差≤5.21e-18m²，49组件测试通过。仅平移几何，尚未接旋转/库存，不能说完整物理修复。详见TRANSLATION_BOUNDARY_RESULT.md，runs/swept_pickup_001、translation_boundary_001。旧审计33596继续，无PPO。

## 续做：旋转边界流量组件
新增刚体边界法向流量、空间线段裁剪和时间Gauss积分，52组件测试通过。case13逐墙格闭合4096区间最大面积误差1.295e-8m²（仅均匀2mm示意换算0.001036mm），耗时15.08秒；未接材料库存，不是G0通过。详见ROTATING_BOUNDARY_RESULT.md及runs/rotating_boundary_001/002。原审计33596继续（核验时23长序列完成），无PPO。

## 续做：冻结边界库存交换组件
新增 NumPy 非负守恒指数交换，56组件测试通过。真实失败轨迹32个独立冻结姿态通过；首次浮点负面积拒绝记录保留，第二次仅量化并置零负舍入面积。未接移动容量、间隙/压力/抬刀，不能算G0通过。详见BOUNDARY_EXCHANGE_COMPONENT.md及runs/boundary_exchange_001/002。原审计33596继续，24长序列完成；无PPO。

## 续做：移动面积解析库存与连续拾料
新增变面积壁面解析解，62组件测试通过。连续零间隙轴向/斜向/旋转基准改善；原case13轨迹均匀初态1024/2048步差0.002198mm，账本约1e-13mL。显式面积闭合投影有修正量记录，尚无工具逐格分配/有限沉积/压力抬刀，不能算G0。详见MOVING_AREA_PICKUP_RESULT.md和runs/moving_pickup_001/002/003。原审计33596继续，24长序列含4失败；无PPO。

## 续做：工具逐格有限库存联合交换
新增中点间隙出料+解析壁面拾料固定点候选，67组件测试通过，轴向/斜向连续128/256步守恒非负。旋转缺少新生接触片的工具格支持，有限端点补全仍失败；未放宽预算。详见COUPLED_BOUNDARY_RESULT.md及runs/coupled_boundary_001/002/003。下一步几何事件分割，尚无压力/抬刀完整集成，G0未过，无PPO。原审计33596仍运行。

## 续做：自适应边界支持积分
70组件测试通过，旋转128/256步联合库存完整走完，无端点补全，墙面差0.00002595mm。长case13组合128/256步诊断PID40724仍运行，runs/adaptive_coupled_boundary_002，勿重复启动；完成结果查summary.json。记录ADAPTIVE_BOUNDARY_PROGRESS.md。内部压力/抬刀未接，G0未过，无PPO。旧审计33596继续，29长序列含6失败。

组合轨迹更新：case13 128步因自适应depth18耗尽失败（difference3.44e-17m²、closure5.12e-18m²）；256步仍在40724运行，查最终summary，不加预算凑通过。

## 续做：全局速度修正未解决组合失败，v8压力接口初测
全局速度v2保持预算，case13 128/256仍depth失败，均已结束（adaptive_global_rates_001）。新增v8压力/余料堆/抬刀接口，空墙短轨迹通过；有料墙初始拾料残值-4.50e-22m³触发非负拒绝（boundary_pressure_001）。73组件测试通过不等于集成通过。记录GLOBAL_RATES_AND_PRESSURE_INTEGRATION.md。下一步稀疏误差项定位及初始拾料账本修复。旧审计33596继续，35长序列6失败；新诊断无运行进程，无PPO。

## 续做：初始拾料供体限额修复与顶点越界预分段
v8.1有料墙初始拾料非负修复；压力/余料堆/抬刀16/32/64步路径完成，差0.01591/0.007919mm。积分失败追踪到工具格0对墙格4348/4349跨界分配；顶点越界预分段使原失败区间通过。76组件测试。完整case13两档PID40456正在runs/vertex_coupled_boundary_001，勿重复启动，查summary。记录INITIAL_PICKUP_AND_VERTEX_EVENTS.md。完整G0未过、无PPO；旧33596继续，36长序列7失败。

## 续做：编译边界加速与完整压力case13
77组件测试通过，局部CPU边界核6.7倍/原失败积分区间5.15倍（非RL端到端）。纯边界case13 512/1024步差0.002163mm；完整压力/抬刀256/512差0.06330mm未过。已启动原50/25μm间距4836/9672段完整case13对照，PID31416，runs/boundary_pressure_case13_gatepair_001，勿重复启动。记录COMPILED_BOUNDARY_AND_CASE13_PRESSURE.md。仍无完整G0/PPO，旧审计33596继续。


# Case13 fixed-spacing result and expanded boundary audit

L1 numerical evidence only. Formal RL has not started.

Original case13 at frozen 50/25 micrometre spacings (4836/9672 segments) completes. Pre-lift maximum wall difference 0.0000547704 mm; post-lift 0.000550873 mm; post-lift RMSE difference 0.00000159875 mm; coverage difference zero. Ledger residual at most 2.71e-14 mL; no negative volume. This passes the single-case criteria, not the full material gate.

Evidence: runs/boundary_pressure_case13_gatepair_001/{summary.json,quality_comparison.json,prelift_*.npz,final_*.npz}. Coarse 256/512 result remains preserved and fails accuracy.

Started expanded_boundary_001: 100 single cases plus 50 sequences of 100 actions, unchanged spacings/thresholds/scenes/action recipes from expanded_split_audit, BoundaryPressure v8.1. Manager PID 32992 (launcher 40600), four workers to coexist with old 12-worker v4 audit. No duplicate resource monitor. Existing resource stream continues; new audit records per-case elapsed time. Source snapshot includes current dirty source; do not infer snapshot equals git HEAD alone.

Next: check actual processes and cases/summary in expanded_boundary_001; do not restart. Preserve exceptions and failed cases. Passing case13 does not prove broad equivalence. Full continuous robot executor/GPU environment/trainer and training/evaluation/replay remain outstanding.


# Expanded boundary audit: zero-area inventory diagnostic

L1 software session. Expanded v8.1 audit PID32992 continues unchanged. Snapshot: 32 completed cases, 29 passed. Failures: [(1, "ValueError('Material in zero-area initial reservoir')"), (18, "ValueError('Material in zero-area initial reservoir')"), (21, "ValueError('Material in zero-area initial reservoir')")].

Started independent case21 input capture in runs/zero_area_trace_001, actual PID35368, launcher31772. The diagnostic wraps exchange only in its own process and dumps area, target, wall/blade inventories and incoming/outgoing sparse maps before the original solver rejects. No tolerance change, no clearing material, no mutation to running audit modules. Do not duplicate. Inspect zero_area.json, failure_inputs.npz and summary.json when finished. If a nonzero wall inventory occupies snapped zero area, first quantify its magnitude and upstream source (initial pickup, edge deposition, pressure/slump or exchange); do not assume roundoff without evidence.

Full material G0 has not passed and formal RL has not begun. Original expanded_split_001 PID33596 remains preserved. Followups should read this record and live cases rather than the historical root implementation_state.json.
