# wall-cycle v0.7：奖励重塑、工作区面积修正与密集回放

针对 v0.6 P2-A 回放暴露的三个问题做定向修复，物理与动作空间不变
（`physics` 仍为 `v0.6`，只改配置/奖励/几何/记录密度）。

**A. 奖励与晋级（env.py/train.py/config.py）**：`_project()` 投影代价从固定
`-0.5` 改为按偏移量与累计次数分级（`project_base_penalty`+`project_count_penalty`）；
新增按覆盖率增量的势函数式塑形奖励 `coverage_shaping_coef`；受力越限代价从
`-50` 悬崖改为二次分级 `force_over_coef`（封顶 `force_over_penalty_cap`），
超过 `force_terminate_mult×max_force_N` 才终止回合；`waste_coef`/`carry_loss_coef`
暴露为配置项；新增 `MIN_COVERAGE_RATIO=0.90` 晋级硬性否决（在原有"三项指标二胜"
规则之外新增覆盖率不得低于 DAgger-BC 基线 90% 的门槛，见 `train.promotion_ok()`）。

**B. 工作区几何（tools/simulation/v2_trowel_reach.py/area.py/material.py/config.py/
work_area.json）**：`work_square()` 的 80% 安全系数修正为作用在可达圆**面积**上
（`r_scored = r*sqrt(area_safety)`），而非直接作用在内接正方形边长上，此前的实现
等效于边长打 8 折、面积只剩 64%，是安全工作区偏小的根因。工作区从单一正方形拆分为
两个同心正方形：`sim_square`（更大，物理/仿真范围——墙面网格、动作解码裁剪边界）与
`score_square`（更小，评分范围——覆盖率/RMSE/收尾判定/教师瞄准只看这一层）。
两层之间的余量允许材料越界且不计入浪费，教师内缩边距同步从 0.02m 收窄到 0.003m。
`work_area.json` 已基于现有已验证的 `circle_radius_m=0.165` 重新推导（未重新扫描，
半径本身未变，只是下游公式变了）：`score_square.side_m=0.205`，`sim_square.side_m=0.230`。

**C. 回放记录密度（arm.py）**：定位到"闪现"根因——`FEED_ALIGN_UP`/`SCAN_RETURN`
两段过渡此前各只调用一次 `_run()`+一次 `mark()`，MuJoCo 内部确实做了多步仿真，
但只有最终姿态被记入 `trace`，而 `view.py` 的回放插值又专门跳过了对 `physics=='v0.6'`
的大关节跳变补帧（假设 v0.6 轨迹已经足够密，这对 `CARRY_FACE_UP`/`ROTATE_TO_WALL`
成立但对这两段过渡不成立）。修复：新增 `ArmExecutor._dense_path()` 做关节空间线性
插值并逐点做碰撞检查，`plan()`/`execute()` 对这两段过渡改为多点 `_run()`+多点 `mark()`，
与既有的 `q_carry`/`q_rotate` 密集记录方式一致。

**前置条件（必须由用户在实机所在机器上执行，本次未运行）**：`sim_square` 边长比
v0.6 的旧工作区更大，`ReachTableExecutor` 对 `width_m`/`height_m` 做严格相等校验，
训练前必须重建 `reach_table_lab_v05.npz`：
`python -m dummy_loop.wall_cycle.reach_table --step 0.02`。

新增启动脚本：`experiments/v0.7/r0/scripts/training/整片抹墙v0.7训练.cmd`（与 v0.6 同参数，便于同口径对比，
输出到 `experiments/v0.7/r0/runs/v07_p2_manual`）、`experiments/v0.7/r0/scripts/visualization/整片抹墙v0.7_MuJoCo回放.cmd`
（对训练出的 policy 生成新的原生回放并播放）、
`experiments/v0.7/r0/scripts/visualization/整片抹墙v0.7教师回放(免训练).cmd`（教师规则直接生成回放，无需训练，
用于快速肉眼验证闪现/边缘覆盖两处修复）。均为纯软件仿真，无串口、无 `--enable`。
