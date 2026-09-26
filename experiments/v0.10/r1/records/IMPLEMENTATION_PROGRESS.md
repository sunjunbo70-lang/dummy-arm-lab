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


## Trace completed
Case21 reproduces at the second geometry half-move immediately after internal pressure/trailing-edge deposition (split_inventory.py:67). Captured wall cell5153 contains 1.012734565880737e-24 m3 (1.0127e-18 mL) with snapped zero free area. This establishes a tiny residual at the exchange boundary, not a macroscopic material leak. Exact upstream operation still needs attribution; do not globally loosen nonnegative/capacity checks or erase inventory. Next candidate should consistently handle geometric roundoff at deposition and preserve mass. Original audit remains unchanged; trace PID35368 finished, do not wait for it. Evidence runs/zero_area_trace_001/{zero_area.json,failure_inputs.npz,summary.json}. Formal RL not started.


# Bounded edge deposition candidate v8.2

L1 software only. Original v8.1 audit PID32992 remains unchanged (83 completed, 78 passed at this snapshot).

Candidate BoundedEdgePressure excludes edge-deposition overlaps into geometrically fully covered wall cells. The existing 2.5e-17 m2 geometric roundoff budget applies per donor column. Larger overlaps raise, and removed tiny overlaps are redistributed over the same column's remaining wall/outside recipients with normalized area; no material is erased and analytic exchange capacity checks are unchanged. Hypothesis: the captured 1e-24 m3 residual arises from edge deposition's independently clipped overlap. This is not yet causal confirmation; original failed-case regression is running.

New physics_version v10r1.area_pressure_candidate.v8_2_edge_support, separate module. Three targeted tests pass: real short pressure/lift conservation, zero-area recipient excluded with preserved donor amount, macroscopic overlap rejected. This does not constitute G0 or hardware validation.

runs/bounded_edge_001 tests original indices1,18,21 at unchanged50/25micrometre spacings. Actual PID 31516; launcher15000. Do not duplicate or modify its imported code. Read cases/summary when finished. If successful, expand to full numerical suite; original v8.1 failures remain preserved. Complete robot executor/GPU environment/training/evaluation/replay still outstanding; no formal RL started.


# Edge regression passed; expanded v8.2 and empty-map dtype candidate

L1 software. bounded_edge_001 completed all three original failed singles1/18/21: max differences0.00109039/0.0000663825/0.00285366mm, ledger <=1.09e-13mL, all pass. Expanded v8.2 started in expanded_bounded_edge_001, actual PID6544 (launcher15420), 4workers,100singles+50x100sequences, unchanged criteria. Existing v8.1 audit PID32992 and v4 PID33596 preserved.

At snapshot v8.1 had144 completed/95passed: single76 precision0.0163213mm exceeds0.01; multiple zero-area exceptions plus many sequence NumPy divide int64 output errors. Thus edge repair alone cannot claim G0 success.

New isolated typed_boundary_pressure.py v8.3 explicitly allocates floating divide output for empty raw bincount. It inherits v8.2 edge handling; copied move method otherwise unchanged. Mocked empty-boundary zero-motion regression passes. Running audits' sources not modified. This fixes a demonstrable dtype vulnerability; sequence stack trace still pending in sequence_type_trace_001 (session53518). Separate same-pose real contact reproduction session99645 still pending, suggesting geometry zero-motion performance also needs examination. Do not duplicate these jobs; check actual process/captured summary. No formal RL yet.

Next: inspect expanded v8.2 results and completed trace; integrate/test v8.3 in a separate sequence diagnostic before further matrix runs. Single76 numerical failure needs independent investigation. Preserve all old data and do not relax gates.


# Continuous-contact empty-map fix and stationary identity

L1 software session. sequence_type_trace_001/summary.json confirms the original long-sequence failure at boundary_pressure.py:49 (np.divide output integer empty bincount), so v8.3 addresses the observed stack, not just a hypothetical bug.

New independent StationaryBoundaryPressure v8.4 inherits v8.3/v8.2. Exactly equal pose coordinates and angle return from geometry move only; pressure is still recomputed by contact. No approximate-motion tolerance or action removal. This avoids expensive integration of identically zero boundary flux. Three targeted tests passed (stationary inventory unchanged, pressure responds to force, empty-map floating output). Full component suite launched, result pending at recording time (exec session85155); do not claim full-suite pass until completion.

Four-operation sequence0 diagnostic with frozen50/25micrometre spacing is running in runs/stationary_sequence_001, actual PID27260 (launcher37848), using v8.4. This includes load/contact/continuation and does not substitute for100-action sequences. Check summary and step logs; do not duplicate.

Expanded v8.2 remains PID6544, snapshot67/67 singles passing; full audit unfinished. Historical v8.1 ended; prior case76 numerical error must still be checked for v8.2. Old v4 auditPID33596 still alive. No formal training has begun and full G0/G1, GPU environment, PPO campaign and final replays remain outstanding. All original sources/results retained.


# Continuous sequence expansion and remaining single76 precision

L1 software only. stationary_sequence_001 completed4operations, passed: maximum wall difference0.00173261699mm, RMSE difference3.3352e-6mm, coverage difference0, ledger1.9652e-13mL. This establishes short continuation execution, not full G0.

Expanded v8.2 finished100singles:99pass; only single76 fails at0.01632131053mm versus0.01mm threshold. Existing long sequences in v8.2 still inherit dtype bug; failures preserved. v8.4 long sequence-only audit started runs/expanded_stationary_sequences_001, manager40668 (launcher25032),4workers,50sequencesx100actions. Do not duplicate. It intentionally reports required_budget_complete=false because singles=0; do not misreport this as full same-version G0. Any later reuse of v8.2 singles requires explicit equivalence rationale/testing.

Remaining single76 split diagnostic runs/case76_pressure_001, manager9672 (launcher31168), original recipe and make_scene(16) (76 modulo60),3344/6688segments at frozen spacing. Saves prelift/final wall/blade/bead. When finished compare pre/post differences and bead contribution to find remaining precision defect; do not shrink spacing solely to pass or change thresholds. Short sequence and single76 physics v8.4.

Component unittest discover still alive actual23116, launcher35204, session85155, with no output. Result remains pending, not a pass. Diagnose that process if it remains stalled; do not create duplicate full-suite runs. Existing old v4 audit33596 remains alive. Formal RL/GPU environment/robot G1 not complete. Preserve user dirty sources and all historical data.


# Case76 lift error attribution and temporal pressure candidate

L1 software. case76_pressure_001 finished: prelift maxwall0.00540715mm, postlift0.01632812mm fails0.01. At worst cell(51,35), prelift wall difference0.00003746mm, bead deposition0.01344008mm, remaining blade lift0.00285058mm. Evidence lift_error_decomposition.json. Error is dominated by carried inventory released at lift; no acceptance threshold changed.

Independent v8.5 TemporalPressure evaluates nonautonomous SSPRK2 flux stages with start/end interpolated pitch/force rather than identical midpoint controls. Same pressure law, geometry and spacings. This is a numerical candidate, not frozen physics or guaranteed improvement. Single stage-control/conservation regression passes. Original case76 at3344/6688 segments running runs/case76_temporal_001 PID32884 (launcher10972). Read summary and pre/post arrays before further adoption; preserve failures.

Original v8.2 full audit complete99/150 (99/100singles, long sequences affected by known dtype bug); v4 complete111/150. Both fail G0, not active anymore. v8.4 long sequence-only audit40668 continues, initial four sequences around15–21 operations at snapshot; do not duplicate.

Old no-output unittest PID23116 accumulated2818CPU seconds. Stop-Process failed; verified alive, then taskkill succeeded. Parent35204 ended. Diagnostic rerun in component_tests_diagnostic_001 completed83tests with0errors/0failures; tests.log and result.txt preserved. New temporal test was added after discovery and separately passed1test, so total83+1, not one84-test suite. Diagnostic test run is complete.

Next: evaluate temporal candidate; continue long-sequence audit. Formal RL/GPU environment/full robot executor remain incomplete. All work D:/VLA, no hardware instructions or changes to active solver versions.


# Temporal control candidate rejected; pressure subcycle diagnostic

L1 software. v8.5 case76_temporal_001 finished both resolutions; wall difference0.01633718116mm still exceeds0.01mm and does not improve v8.4. Do not adopt it as a repair. Terminal pressure comparison is saved: both gaps0.0009312020588m, support13.889882N vs requested10.141783N; bead inventory4.889994/4.888454mL. Non-smooth pressure/contact transitions remain possible; this does not by itself prove root cause.

New diagnostic v8.6 SubcyclePressure inherits v8.4 and uses four internal SSPRK2 substeps per previous constitutive interval, retaining midpoint controls, external50/25micrometre spacing, geometry, physical laws and acceptance criteria. This is controlled internal solver refinement, not a gate-spacing adjustment or frozen physics. Component check passed:8flux evaluations instead of2, nonnegative wall and conservation. Evidence runs/case76_subcycle_001/component_check.json.

Case76 v8.6 at3344/6688segments running runs/case76_subcycle_001 PID5688 (launcher32952), saves pre/post lift arrays. Evaluate accuracy AND added runtime; do not claim success before summary. Long v8.4 audit40668 remains alive, first four sequences35/30/37/41 operations at snapshot; no complete100-action sequence yet. Do not duplicate or modify these modules while running. Read-only auxiliary stdin diagnostic session6617 remained pending without output; its intended terminal-pressure computation was successfully saved separately, not needed for adoption.

Formal RL has not begun. Full material G0, robot G1, GPU environment/training and final evaluation/replay remain missing. Previous failures and original user modifications preserved.


# Pressure subcycling failed; matched-time divergence trace

L1 software. v8.6 case76_subcycle_001 completed both3344/6688 resolutions, maxwall difference0.01635887253mm vs0.01mm, ledger1.36e-14mL. Fourfold internal SSPRK2 refinement did not improve v8.4; not adopted. Preserve this negative result alongside v8.5. No reduced acceptance threshold or removed actions.

New runs/case76_divergence_001 (actualPID2408,launcher28912) advances both v8.4 states at matching path times; logs gap, per-column bead delta, blade/wall error and largest incremental bead changes. This is diagnostic only, not a new physical model. Read summary/top_jumps and trace.jsonl when done before proposing more numerical changes. Existing long sequence manager40668 remains active; do not duplicate or change imported modules.

Resource telemetry audit found no active telemetry process after old monitored audits ended. Restored one5second monitor PID13564 targeting40668 at expanded_stationary_sequences_001/resources_resumed.jsonl. Previous interval is a monitoring gap, cannot claim continuous coverage. Existing raw resource logs preserved; no duplicate current monitor.

Formal RL has not begun; numerical G0 not passed. Full GPU environment, robot continuous execution G1, PPO campaign, evaluations and native full-job replay still missing. Component suite83 tests and separate temporal1test remain latest pass; no test success asserted for new divergence diagnostic. Work only D:/VLA, no hardware.


# Matched-time evidence and quarter-control exit-gap candidate

L1 software. case76_divergence_001 completed. Largest incremental bead difference at step2777, t=0.8304425837320574: 3.0283812431494347e-05mm per coarse interval. Both terminal gaps at this point equal 0.0008464262676239013m. Nearby intervals show smooth accumulation, not one large jump. Thus the previous temporal-control/internal-subcycling hypotheses were not supported as sufficient repairs.

Independent v8.7 QuarterGapPressure uses interpolated controls at quarter/three-quarter time when solving exit gaps for the two geometry half-steps. It uses current available inventories, not fully predicted quarter-time inventories; therefore do not claim a proven second-order coupled solver. Physical pressure law/material accounting/external50/25micrometre sampling and acceptance criteria unchanged. Two tests passed: constant-control short path equals v8.4, changing-control stage timing and conservation/nonnegative inventory. No full-G0 claim.

Original case76 at3344/6688segments running runs/case76_quarter_gap_001 actualPID29260 (launcher9672). Read summary and saved pre/postlift arrays before adopting or rejecting; don't duplicate. Previous candidates remain preserved. Long-sequence v8.4 audit40668 continues, first four sequences67/66/75/72 operations at snapshot; no completed100-action sequence then. Single5sec telemetry13564 is alive, no new monitor.

Formal RL not started. Complete material/robot gates, GPU environment/PPO/evaluation/native full-job replay remain outstanding. No hardware action, no user-source modifications.


# Quarter-gap failure and bounded whole-contact error-control diagnostic

L1 software. v8.7 quarter-gap case76 finished with maxwall0.01642326642mm, still above0.01mm. Do not adopt; previous failed candidates retained.

New v8.8 ErrorControlPressure performs full-contact step doubling based on v8.4: compare one step with two half-steps for wall/blade/bead inventories (volume converted to equivalent blade-cell thickness). Accept two-half result only when max difference <=1e-4mm per5mm swept path plus1e-12mm arithmetic floor. Otherwise recursively split; depth6 exhaustion raises explicitly. No extrapolation or inventory clipping, no altered physical laws or gate thresholds. Caller state changes only after successful advance. Local estimator is not a proven global error bound, particularly at nonsmooth contact; full same-version gates remain necessary. This solver changes internal time sampling and costs more; accuracy/runtime must both be reported.

Two targeted tests passed: conservative/nonnegative advance and rejected step leaves original inventories unchanged. Original case76 at external3344/6688segments running runs/case76_error_control_001 actualPID39808 (launcher21204). Check summary/failure and boundary_stats contact_error_nodes/depth/error sum before further expansion, do not merely raise budget if it fails. External gate spacing50/25micrometres unchanged.

Long-sequence v8.4 manager40668 continues. First four traces84/88/94/93lines at snapshot, no completed100-action sequence yet (lines include load/final comparisons, not a training count). Telemetry13564 remains the single monitor. Formal RL has not begun, full material and robot gates, GPU environment/training/evaluation/replay incomplete. Only D:/VLA, no hardware.


# First complete long sequences and contact sampling component

L1 software session. v8.4 expanded_stationary_sequences_001 completed sequences2,3,1,0 (100operations each), all passed frozen sequence criteria. Maximumwall differences0.01218819/0.01080028/0.01146380/0.00417386mm; ledger<=1.79e-12mL. These are4/50 long sequences, not full G0; same-version single76 still fails. Manager40668 and single resource monitor13564 continue.

v8.8 case76_error_control_001 PID39808 remains alive with~1703CPU seconds on inspection and latest coarse progress step2304/3344 at~199reported seconds. No completed pair/summary, so error-control effectiveness and cost remain unresolved. Do not duplicate or change active source; inspect progress/actual process and eventual failure/summary.

Independent contact_sampling.py added for future continuous-contact executor: de Casteljau subdivision with control-polygon translation upper bound plus angular tool-radius sweep bound, explicit max depth rejection; endpoint position/angle/pitch/force/speed continuity check; approximate speed quadrature and explicit angular-rate-limited duration for pure rotation. It is a geometric/time sampler only, NOT IK/collision/dynamics or robot G1. Tangent continuity is not enforced; later executable joint trajectory must account for direction changes. No integration into current audit or frozen physics.

Four tests pass: curved path swept bound/endpoints, nonzero pure-rotation duration, inherited continuation and deliberate force discontinuity, explicit depth exhaustion. Test command: python -m unittest dummy_loop.wall_cycle_v10.r1.test_contact_sampling. Sampling angular-rate bound is required caller input, no unverified hardware parameter hardcoded.

Formal RL/GPU env/full robot execution not complete; no training started. All work onlyD:/VLA, hardware untouched, old records and user dirty files preserved.


# Error-control budget failure and pressure precision diagnostic

L1 software. v8.8 case76_error_control_001 failed both resolutions at fixed depth6: 3344 steps failed at2314 (error1.6882930473e-8mm versus tolerance1.5621328488e-8mm); 6688 failed at5463 (7.8121361681e-9 versus7.8111642441e-9mm). No tolerance or depth increase. Failure evidence retained.

v8.9 PreciseGapPressure tests whether the20-iteration pressure bisection precision floor contributes to failed adaptive convergence. Same support law/bracket,40 iterations for subsequent pressure evaluations, same v8.8 local tolerance/depth. Initial-contact inherited solver is unchanged. Support has wet/dry discontinuities, so more iterations do not guarantee a force root or resolve the case. This is a candidate diagnostic, not frozen physics.

Two targeted tests passed (uniform support root precision; conservative nonnegative adaptive short advance/lift). Case76 diagnostic launched with original3344/6688 external segments in runs/case76_precise_gap_001: actualPID17240, launcher33596. Source snapshot includes dirty diff; new files committed separately. Check cases/summary before any expansion.

v8.4 long-sequence manager40668 remains active; four of50 complete and pass, remaining pending. Existing single telemetry13564 retained. Full G0 remains unpassed; full robot/GPU/PPO pipeline and formal RL/evaluation/native full-job replay incomplete. No formal training has started. Old results/user source changes preserved; no hardware actions.


# Pressure precision hypothesis and contact rejection trace

L1 software session. v8.9 coarse3344 case76 failed at2314 with error1.68846224217e-8mm versus1.56213284876e-8mm tolerance, almost identical to v8.8. Increasing gap bisection20 to40 has not removed this failure; pressure precision alone is not supported as its cause. Fine6688 remains running PID17240; preserve eventual result. No acceptance tolerance/depth changed.

Added trace_contact_rejection.py, instrumentation of unchanged v8.9 contact. On rejection it records coarse/two-half-step differences separately for wall/blade/bead, maximum-error cell, signed total differences and pressure gaps at each rejected recursion depth; deepest pre-step state is saved for reproducible local diagnosis. This is not another physics candidate. One controlled rejection component test passed: record/pickle produced, original inventory unchanged (runs/contact_rejection_component_001/result.json).

Original case76 coarse reproduction running in runs/contact_rejection_trace_001 PID26144 (launcher32480); check summary/depth JSON/rejected_state.pkl next. Do not restart or blindly increase recursion budget. Long-sequence manager40668 still active, first4/50 completed and passed; sole telemetry13564 retained. Latest resource snapshot four workers roughly99-100percent of one CPU core each, system33.9percent CPU, GPU6percent; this numerical validation is not a GPU training benchmark.

Full G0 unpassed, full execution/GPU/training/evaluation/replay remain missing. Formal RL never started. No hardware, no original results or unrelated user edits overwritten.


# Wall gravity-cap ordering: local causal diagnostic and candidate v8.10

L1 software. v8.9 completed with both resolutions rejected: coarse step2314, fine step5517, depth6 unchanged. Rejection trace identifies blade[0,3] max1.68846224e-8mm, versus wall1.14815316e-10 and bead5.68359e-12mm. Coarse/fine pressure gaps identical. Local replay from saved pre-rejection state with target fractions1,1/2,1/4,1/8,1/16 yields blade errors1.68846e-8,4.22062e-9,1.05509e-9,2.63764e-10,6.59373e-11mm. Approximately quadratic local error; no precision floor at tested scale. Evidence runs/contact_local_scaling_001/results.json. No gate relaxation.

Hypothesis: end-of-contact-only wall slump projection allows deposition above wall carrying capacity to be picked up during the same geometry step, whereas smaller steps shed it earlier. This introduces operator-order dependence. Candidate v8.10 applies the SAME height bound tau_y/(rho*g) before wall pickup and to wall-directed deposition inside the coupled exchange. Blade depletion still uses original exit density; excess wall-directed deposition is explicitly added to dropped inventory rather than retained or deleted. Outside deposition remains outside. Receiving wall density and initial wall inventory bounded by capacity make the free-area transfer preserve the cap under its assumed constant fluxes. Internal bead extrusion, support law and final lift remain unchanged. This changes within-step physical ordering and is NOT frozen/validated physics; full gates and assessment of lift/contact initialization remain required.

Saved rejected local state replay now blade error1.15658114e-10mm (~146x smaller), wall1.14815316e-10, bead5.61159e-12; ledger1.35525e-14mL and nonnegative. This is a local controlled intervention, not proof all case76/global errors arise here. Evidence runs/saturated_wall_local_001/result.json. Two unit tests passed: uncapped transfer exactly matches original; capped transfer sheds positive excess, bounds wall density, retains nonnegative blade and closes total ledger.

Full original case76 pair3344/6688 running runs/case76_saturated_wall_001 PID7244 (launcher35268), unchanged adaptive tolerance/depth and external gate resolutions. Source snapshot retained. Check cases/summary before expansion, do not duplicate. Long sequence manager40668 and monitor13564 continue original v8.4 unmodified (4/50 completed pass at snapshot). No new resource monitor.

Full numerical G0 and robot G1 incomplete; full GPU environment/PPO/formal RL/evaluation/full-job MuJoCo replay still outstanding. No formal training begun. Only D:/VLA software, user dirty files and all historical failures retained.


# Fixed-step wall-cap case76 passed; same-version expanded audit started

L1 software. Adaptive v8.10 case76 rejected both resolutions near end:3344 at3199,error7.72072689e-8mm;6688 at6398,error3.55459335e-8mm,depth6. Local adaptive estimator was introduced as a diagnostic, not part of the original frozen G0 thresholds. These failures remain recorded and must not be relabelled successful.

v8.11 FixedCapPressure uses v8.10 wall-cap physical ordering with the original StationaryBoundaryPressure fixed-step contact integrator, so the actual prescribed50/25micrometre endpoint comparison can be measured. No global acceptance threshold changed. Removing adaptive rejection does not establish convergence on its own; the global test remains decisive. Physics version distinct, not frozen.

Original case76 pair completed:3344 steps59.41s,6688 steps84.60s, max final wall difference0.00313508934mm (threshold0.01mm), ledger0.0mL at logged precision, minimum0.0. Earlier v8.4 same case difference0.0163281157mm. Candidate passes this pair, not full G0. Evidence runs/case76_fixed_cap_001/cases.jsonl. Four tests passed including100 random positive exchange cases verifying cap/nonnegativity/total ledger, no-cap identity and contact/lift ledger.

Full same-version audit launched runs/expanded_fixed_cap_001, actualPID22620 launcher24180:100singles+50sequences of100operations,8workers, unchanged50/25micrometre spacings and thresholds. Source snapshot includes new candidate files; commit follows. All old results preserved. Original v8.4 long-sequence manager40668 remains active with4workers and monitor13564; no duplicate monitor. Existing resource log provides machine-wide CPU/GPU utilization but per-process tree is only original manager, not new audit: do not claim per-process monitoring of new workers. Total12 CPU workers across both audits; no GPU benchmark claim.

Next: inspect actual jobs/results, if expanded gate fails locate exact case; if material subsystem passes proceed full environment/G1 and GPU integration. Formal RL has not started; full executor/GPU environment/PPO/evaluation/native full-job replay still missing. No hardware commands/user source changes.


# Offline joint-path preflight component while material audit runs

L1 software. v8.11 expanded_fixed_cap_001 actualPID22620 remains active with8workers. Snapshot89 completed single cases, all pass; no completed new-version long sequences yet. Original v8.4 audit40668 continues with4workers and4 completed long sequences. No restarts/duplicate monitoring. Full same-version material gate still incomplete.

Added joint_path.py as an independent offline preflight building block: caller-provided IK/FK/clearance callbacks, all-joint position/rate limits, adjacent joint jump cap, both translational and full SO(3) residual checks, interpolated joint-path collision samples. Solver receives copies of seeds; a rejected path does not return a partially accepted executable plan. Initial pose checked. Rotation orthonormality and finite inputs validated. Caller must provide justified tolerances and limits; no hardware calibration value is invented.

Five synthetic callback tests pass: accepted continuous path with mutating solver callback, interior collision rejection even with clear endpoints, rate-limit rejection, orientation residual rejection, IK branch-jump rejection. Evidence runs/joint_path_component_001/result.json. This is NOT a MuJoCo robot execution test: callbacks still need integration, material tool-frame/pitch conventions need verification, approach/load/lift and dynamic tracking remain missing. Collision sampling is discrete, not a proof of continuous clearance; acceleration/torque constraints also remain outstanding. No G1/full executor claim.

Next: continue same-version100+50x100 material validation and integrate physical robot targets/IK callback into preflight without touching active material audit sources. Formal RL, GPU environment, complete PPO training/evaluation/native full-job replay not yet completed. Hardware untouched; user dirty sources preserved.


# USER PAUSED 2026-09-26
All identified audit processes/descendants and telemetry stopped (22 targets,0 remaining); automation v0-10-r1 PAUSED. Latest v8.11 singles100/100, long sequences0/50 completed (8partial); oldv8.4 sequences7/50 completed. No formal RL started. Do not automatically resume on stale heartbeat. See PAUSE_AND_RETROSPECTIVE_20260926.md and runs/user_pause_001/pause_state.json.
