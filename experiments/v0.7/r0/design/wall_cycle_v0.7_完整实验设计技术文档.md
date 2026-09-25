# wall_cycle v0.7 完整实验设计技术文档

日期：2026-09-23
证据等级：L1（纯 MuJoCo 仿真，全程未连接实体机械臂，未执行任何 `--enable`/驱动指令）
状态：**设计稿，交付审查，不自动执行**
合并自：《wall_cycle_v0.6结构拆解与v0.7迭代设计.md》（奖励/晋级门槛部分）+
《v0.7设计_第二部分_回放闪现与工作区修正.md》（回放/工作区部分），并补充完整参数表、
风险分析与实施顺序，是这一轮唯一需要照着做的文档。

---

## 一、这一轮要解决的问题（承接 v0.2→v0.6 的问题史）

| 版本 | 已解决 | 遗留到 v0.7 的问题 |
| --- | --- | --- |
| v0.2 | 建立行程级 POMDP、教师/BC/PPO 三方同分布对比框架 | 无俯仰、无搬运物理、工作区未按可达性推导 |
| v0.3 | 物理化砂浆模型（`mortar.py`）、俯仰作为自由动作、工作区按可达圆推导 | 俯仰倾角有效性未被证实；装料/搬运仍是瞬时补满，无姿态物理 |
| v0.5 | 装料/搬运/朝天分层技能、256×256 网络、可原生回放 | 朝天程度由策略"自称"而非实测；PPO 全程没有跑赢教师；cosim 复测比表执行器评测更差 |
| v0.6 | 朝天姿态改为执行器实测约束、搬运损耗用实测姿态驱动、跨回合 IK 状态清零 | **P2-A 奖励塑形诱导策略走"低投入低风险"捷径而非真实覆盖**（本文档第三节 A 组修复）；**回放中装料对齐/回扫描位两段运动只记录单帧，呈现瞬移**（第三节 C 组）；**工作区面积规则用错（边长×0.8 而非面积×0.8 再取内接正方形），比预期小约 25% 面积**（第三节 B 组）；**越界即扣分导致策略主动放弃边缘覆盖**（第三节 B 组） |

v0.7 的目标：**在同一轮训练里同时修完这四类问题**，用同一套固定测试集把 教师 / DAgger-BC / PPO 三方重新对比一次，
判断是否能让 PPO 首次在"至少 2/3 指标不劣于 DAgger，且覆盖率不倒退超过 10%"这条新门槛下通过晋级。

---

## 二、整体架构（不变的部分，先说清楚基线）

```
CycleConfig (config.py)                        单一配置对象，贯穿全部模块
        │
        ├─ MortarSystem (mortar.py, 继承 MaterialSystem/material.py)
        │       物理：挤压力平衡→间隙、Couette 铺展、挤出分配（gap³）、重力排液、
        │       墙面塌陷；volume 严格守恒。不含任何"倾斜更省料"这类行为规则。
        │
        ├─ ArmExecutor (arm.py)
        │       plan()：运动学可行性检查（IK + 关节限位 + 离墙/离桌净距）
        │       execute()：MuJoCo 动力学协同仿真（位置伺服 + 力导纳 + 材料反作用力）
        │
        ├─ D435Proxy (sensor.py)：仿真深度相机噪声代理，策略永远看不到真实场地
        │
        └─ WallCycleEnv (env.py)
                行程级决策；decode()/encode() 做动作↔物理量映射；
                teacher_action() 是不知道物理模型的规则式教师；
                step() 的奖励函数是本轮修改的核心。

train.py：教师(200回合) → BC → 4×DAgger → PPO(102,400决策步)
          → 教师/DAgger-BC/PPO-last/PPO-best-val 四方在固定测试集(seed 20000+)对比
          → 独立验证集(seed 30000+)选检查点 → 合格则用 cosim 执行器复测30回合
```

这套骨架本身不改；v0.7 的改动全部是**参数、奖励系数、工作区几何、以及运动记录密度**层面的调整，
不新增 `physics` 版本字符串——所有改动仍然挂在既有的 `cfg.physics == 'v0.6'` 分支下（因为动作空间和物理方程都不变，
只是配置数值和记录方式变了），这样能把改动面压缩到最小，不需要把 `'v0.6'` 字符串判断到处改成 `'v0.7'`。
**`physics` 字段本轮维持 `'v0.6'`；`'v0.7'` 只是这一轮实验和文档的名字，不是代码里的新分支。**

---

## 三、四组修复的完整技术细节

### A 组：奖励函数与晋级门槛（解决"策略靠避险刷分而非真实覆盖"）

**A1. 投影代价改为按偏离程度 + 回合内次数定价**（`env.py:_project()` / `step()`）

`_project()` 现在只返回 `(cand, plan)`；改为额外返回本次采用的 `(f_tilt, shrink)`，`step()` 里替换：

```python
proj = self._project(d)
if proj is not None:
    d, plan, f_tilt, shrink = proj
    self.projected += 1
    reward -= c.project_base_penalty * (1 + (1 - f_tilt) + 2 * shrink) + c.project_count_penalty * self.projected
```

新增 `CycleConfig` 字段：`project_base_penalty: float = 0.5`（原固定惩罚的基数，语义不变，只是现在是基数不是全部）、
`project_count_penalty: float = 0.1`（每回合内每多一次投影，额外线性叠加的代价）。

**A2. 覆盖率增量奖励（势函数塑形，不破坏最优策略不变性）**（`env.py:step()`）

```python
coverage_before = self.material.metrics()['coverage']     # step() 顶部，before 附近
...
coverage_after = self.material.metrics()['coverage']       # improvement 计算之后
reward += c.coverage_shaping_coef * (coverage_after - coverage_before)
```

新增字段：`coverage_shaping_coef: float = 8.0`（初始值，需要在 P0 小规模消融里和 `waste`/`improvement` 系数一起扫，
见第五节）。

**A3. 力超限惩罚从悬崖式改为渐进二次惩罚**（`env.py:step()`）

```python
over = max(0.0, (stats.peak_force_N or 0.0) - c.max_force_N)
if over > 0:
    reward -= min(c.force_over_penalty_cap, c.force_over_coef * (over / c.max_force_N) ** 2)
if (stats.peak_force_N or 0.0) > c.force_terminate_mult * c.max_force_N:
    self.done = True
```

新增字段：`force_over_coef: float = 8.0`、`force_over_penalty_cap: float = 50.0`（沿用原悬崖惩罚的量级作为封顶）、
`force_terminate_mult: float = 1.5`（原来是超过 `max_force_N` 就终止，现在放宽到 1.5 倍才终止整回合）。

**A4. `carry_loss`/`waste` 系数显式暴露为配置**（原本硬编码在 `step()` 里的 `4`）：

新增字段：`waste_coef: float = 4.0`、`carry_loss_coef: float = 4.0`，`step()` 里的 `4*waste/...`、`4*carry_loss/...`
分别替换成 `c.waste_coef*...`、`c.carry_loss_coef*...`——不改变 v0.6 的默认行为，只是让这两个此前"隐藏"在代码里的
关键系数可以在 P0 消融里直接调，不用改代码。

**A5. 可达动作参数化（本轮标记为可选二阶段，见第五节执行顺序）**

用已有的 `reach_table_lab_v05.npz`（第三节 B 组重建之后的新版本）在 `decode()` 里对 `(start_u, start_v)` 做一次
"钳制到最近可行格"的预处理，减少提议后被 `_project()`/不可达惩罚事后纠偏的比例。这一步改动面较大（涉及
`decode()` 读表的接口），**本轮先不做**，等 A1-A4 + B/C 组跑完一轮看晋级结果再决定是否需要。

**A6. 晋级门槛追加硬约束**（`train.py`，`selection_score` 旁新增判定，不参与打分只做布尔否决）：

```python
MIN_COVERAGE_RATIO = 0.90   # 新增常量，train.py 顶部，紧邻 TEST_SEED/VAL_SEED

def promotion_ok(ppo_test, dagger_test):
    two_of_three = sum([ppo_test['coverage_mean'] >= dagger_test['coverage_mean'],
                        ppo_test['rmse_mm_mean'] <= dagger_test['rmse_mm_mean'],
                        ppo_test['waste_frac_mean'] <= dagger_test['waste_frac_mean']]) >= 2
    no_coverage_regression = ppo_test['coverage_mean'] >= MIN_COVERAGE_RATIO * dagger_test['coverage_mean']
    return two_of_three and no_coverage_regression
```

`train()` 末尾原有的"是否选 PPO 检查点"逻辑不变（仍由 `selection_score`/`best_val` 决定部署哪个检查点），
`promotion_ok()` 只用于 P2-B 是否解锁的判定，写入 `training.json` 的 `result['promotion']` 字段，供实验记录留档。

### B 组：工作区几何与越界惩罚（解决"面积偏小"和"边缘被放弃"）

**B1. 面积安全系数改为作用在圆面积，不是边长**（`tools/simulation/v2_trowel_reach.py:work_square()`）：

```python
def work_square(us, vs, ok, area_safety=0.8):
    ...                                              # r 的推导不变
    r_scored = r * np.sqrt(area_safety)              # 面积先留 80%，同心缩圆
    side_scored = np.sqrt(2) * r_scored                # 缩圆后的内接正方形 = 评分区边长
    side_sim = np.sqrt(2) * r                            # 原始内接正方形 = 仿真/物理区边长（不再打折）
    ...
    return {..., 'score_square_side_m': round(side_scored, 4), 'sim_square_side_m': round(side_sim, 4), ...}
```

用当前已知 `r = 0.165 m` 算出的新旧对比（不需要重新扫描就能核对数量级）：

| 量 | 现状（边长×0.8） | 新算法（面积×0.8） |
| --- | ---: | ---: |
| 评分区边长 | 0.1867 m | 0.2087 m（+11.8%）|
| 评分区面积 | 0.0349 m² | 0.0435 m²（+24.9%）|
| 仿真/物理区边长（=原未打折内接正方形） | — | 0.2333 m |

重新扫描命令（纯 IK/碰撞几何分析，不含硬件驱动指令）：

```
.venv-loop\Scripts\python.exe tools\simulation\v2_trowel_reach.py --pitches 0 --distances 0.28 0.30 0.33 0.35 0.38 --step 0.01 --out experiments\v0.1\r0\runs\reach
```

**B2. `config.py`/`area.py`/`work_area.json` 拆成两个同心正方形**：

新增字段 `CycleConfig.score_width_m/score_height_m`（评分区，教师瞄准、覆盖率/RMSE/收尾门槛只看这里）；
`width_m/height_m` 保留字段名，语义变为"仿真/物理区"（更大，同心）。`area.py:load_work_area()` 同时写入两组字段。

**B3. `material.py` 的 `metrics()` 只在评分区窗口内统计**：

`MaterialSystem.__init__` 新增一次性的评分区行列切片（`self._score_rows`/`self._score_cols`，居中裁剪），
`metrics()` 改成 `h = self.wall[self._score_rows, self._score_cols]` 再计算，其余逻辑不变；`quality_cost()`
不用改（它只是包了一层 `metrics()`）；`waste_frac`/`dropped_m3`/`outside_m3` 的定义和分母维持不变，
不按评分区裁剪（掉在仿真区内、评分区外的材料仍然是"留在墙上"，不是真的浪费）。

**B4. `env.py:decode()` 的越界 clip 不用改代码**——它引用的就是 `c.width_m/height_m`，这两个字段语义变大后自动生效。

**B5. `env.py:teacher_action()` 两处调整**：
- `under`/`over` demand 权重只在评分区窗口内取（新增一次索引切片，和 B3 用同一套边界）；
- `teacher_edge_margin_m`：`0.02 → 0.003`（约一个物理格子，象征性保留，不再系统性收缩 2 cm）。

**B6. `reach_table.py` 必须重建**（`ReachTableExecutor` 加载时严格比对 `width_m/height_m`，不重建会直接 `ValueError`）：

```
.venv-loop\Scripts\python.exe -m dummy_loop.wall_cycle.reach_table --step 0.02
```

### C 组：回放记录密度（解决"闪现"，不影响训练数值本身）

**C1. 装料对齐路径预计算一次**（`arm.py:ArmExecutor.__init__()` 末尾）：

```python
self.q_feed_path = self._dense_path(self.q_scan, self.q_feed, n=20)   # 新方法，线性插值+逐点 clear_of_wall
```

**C2. 回扫描位路径按刀计算**（`arm.py:plan()`，紧接 `ql` 算完之后）：

```python
q_scan_path = self._dense_path(ql[-1], self.q_scan, n=20)
if q_scan_path is None:
    return self._reject('no dense path back to scan pose')
```

`StrokePlan` 新增字段 `q_scan_path: list`。

**C3. `execute()` 把两处单帧 `_run`+`mark` 换成逐点循环**（对应 `arm.py:403-406`、`:464-466`）：

```python
for q in self.q_feed_path:
    self._run(q, self._move_time(q, joint_speed_deg_s=20.0, settle_s=0.02)); mark('FEED_ALIGN_UP')
...
if c.physics == 'v0.6':
    for q in plan.q_scan_path:
        dt = self._move_time(q, joint_speed_deg_s=30.0, settle_s=0.02)
        self._run(q, dt); mark('SCAN_RETURN')
        mortar.transport(trace[-1]['face_up_score'], dt, stats=stats)
```

`view.py` 不需要改（它对这两个 phase 已经是"来多少条目收多少条目"）。副作用：`face_down_frames` 结构性门槛的
审计范围从"只查最终姿态"变成对这两段全程审计，比 v0.6 更严格。

---

## 四、完整参数表（旧值 → 新值，全部标明理由）

| 参数 | v0.6 值 | v0.7 值 | 所属模块 | 理由 |
| --- | --- | --- | --- | --- |
| `score_width_m`/`score_height_m` | （不存在，即 `width_m/height_m`=0.185/0.185）| 0.209 / 0.209（B1 重新扫描后写入） | config.py（新增） | 面积×0.8 而非边长×0.8 |
| `width_m`/`height_m`（仿真/物理区） | 0.185 / 0.185 | 0.233 / 0.233（B1 重新扫描后写入） | config.py（语义变更） | 恢复未打折的可达内接正方形，给越界留出物理空间 |
| `teacher_edge_margin_m` | 0.02 | 0.003 | config.py | 越界不再扣分，不需要人为再留 2cm |
| `project_base_penalty` | 0.5（硬编码） | 0.5（暴露为配置，公式变了） | config.py（新增） | A1：按偏离程度定价 |
| `project_count_penalty` | 0（不存在） | 0.1 | config.py（新增） | A1：抑制回合内滥用投影 |
| `coverage_shaping_coef` | 0（不存在） | 8.0（P0 消融后可调） | config.py（新增） | A2：显式覆盖率塑形 |
| `force_over_coef` | 0（悬崖式） | 8.0 | config.py（新增） | A3：渐进惩罚 |
| `force_over_penalty_cap` | 50（隐式） | 50 | config.py（新增） | A3：维持原惩罚量级上限 |
| `force_terminate_mult` | 1.0（隐式，即等于 `max_force_N`） | 1.5 | config.py（新增） | A3：降低"一次出错满盘皆输"的方差 |
| `waste_coef` | 4（硬编码） | 4（暴露为配置） | config.py（新增） | A4：可调不改代码 |
| `carry_loss_coef` | 4（硬编码） | 4（暴露为配置） | config.py（新增） | A4：同上 |
| `MIN_COVERAGE_RATIO`（晋级门槛） | 不存在 | 0.90 | train.py（新增常量） | A6：堵住"降覆盖率换分数"的退化解 |
| `lift_wall_fraction`（CLI 默认） | 0.75 | 0.75（不变） | train.py CLI | 沿用，本轮不改材料物理 |
| `stall_window` / `stall_limit` / `min_cycles_before_stall` | 12 / 1 / 25 | 12 / 1 / 25（不变，先观察 A2 塑形后是否需要调） | config.py CLI | 本轮不动，避免同时改太多变量 |
| `updates` × `steps_per_update`（PPO 决策步） | 50×2048=102,400 | 50×2048=102,400（不变） | train.py CLI | RESULTS.md 明确建议"不应直接追加步数"，先修奖励再看 |
| `dagger_rounds` / `dagger_episodes` | 4 / 40 | 4 / 40（不变） | train.py CLI | 不变 |
| `explore` 维度 / `explore_log_std` | (force,pitch_start,pitch_end,start_u,start_v,end_u,end_v,blade_cos,blade_sin) / -1.5 | 不变 | train.py | v0.6 已移除 `carry_face_up`，探索维度不用改 |
| `test_episodes` / `val_episodes` / `cosim_test_episodes` | 100 / 40 / 30 | 100 / 40 / 30（不变） | train.py CLI | 维持可比性 |

---

## 五、实施与验证顺序（严格按此顺序，每步都可独立核查）

1. **B1+B6**：改 `work_square()`，重新扫描生成新 `work_area.json`（人工核对 `score_square_side_m≈0.209`、
   `sim_square_side_m≈0.233`），重建 `reach_table_lab_v05.npz`。**不涉及训练代码**，验证方式：直接看 json 数值
   和 reach table 的 `feasible_fraction` 是否合理（不应比 v0.6 明显降低——仿真区变大，理论上不可达比例还会略降）。
2. **B2-B5**：`config.py` 加字段、`area.py` 对接、`material.py` 加评分窗口裁剪、`env.py` 教师瞄准范围/margin 调整。
   验证方式：写一个独立小脚本，跑 10 个教师回合，检查 `metrics()['coverage']` 的分母（评分区格子数）确实变小、
   `wall.shape`（仿真区格子数）确实变大，且教师轨迹的落点分布里贴近评分区边缘（±1cm 内）的比例明显提升。
3. **C1-C3**：`ArmExecutor` 加 `q_feed_path`，`plan()` 加 `q_scan_path`，`execute()` 改成逐点 `mark`。
   验证方式：用教师策略跑一个回合并录制回放（`view --record`），肉眼确认装料对齐和回扫描位不再是硬跳变，
   同时检查 `feed_path_degraded`（若触发）的比例是否为 0 或很低。
4. **A1-A4**：`env.py:step()`/`_project()` 改奖励公式，字段全部加到 `config.py`。
   **先做 P0 消融**：10~30 回合级别，只跑教师+随机探索（不需要完整 PPO），观察：
   - 投影次数分布是否随 `project_count_penalty` 生效而不再无限堆积；
   - `peak_force_N` 分布是否因为 A3 从"贴着上限试探"变得更分散（说明不再是被悬崖惩罚吓退）；
   - `coverage_shaping_coef` 初始 8.0 是否让"只刷覆盖率不管质量"重新成为新的捷径——**必须同时看 coverage 和
     rmse 两个指标**，只看 coverage 上升不能算通过。
5. **A6**：`train.py` 加 `MIN_COVERAGE_RATIO`/`promotion_ok()`，不需要单独验证（下一步整轮训练里自然验证）。
6. **整轮训练**（P1，同 P2-A 规模，102,400 决策步）：`teacher(200) → BC → DAgger(4×40) → PPO(102,400)`，
   四方在固定测试集对比，`promotion_ok()` 判定是否解锁 P2-B。
7. 视 P1 结果决定：通过 → 可以考虑更大规模的 P2-B 或第三节 A5（可达动作参数化）；不通过 → 回头看 P0 消融数据
   定位是哪个系数需要重新扫参，**不直接加 PPO 步数**。
8. 每一步都在 `experiments/2026-09-2x_wall_cycle_v07_xxx/` 下新建独立记录（`README.md`+`run.json`），
   不覆盖 v0.5/v0.6 的既有记录。

---

## 六、风险分析：这次改动自己可能引入的新问题

这一节专门回答"这次新实验中可能会出现的问题"——不是重复第一节的历史问题，是**针对 v0.7 这些改动本身**的风险点。

| 风险 | 触发条件 | 监测指标 | 缓解措施 |
| --- | --- | --- | --- |
| 仿真区变大导致更多姿态在物理上勉强可达但实际不稳定（IK 抖动、关节抢限位） | B1/B6 扩大仿真区后，边缘新纳入的区域只做过 pitch=0 的宽松可达性验证 | `stats['ik_not_converged']`、`unreachable_strokes`、`clear_of_wall` 拒绝率按 u/v 分区统计 | `_project()` 的安全层本身就会兜底拒绝/收缩，不会执行不可行动作；如果新边缘区域拒绝率明显偏高，优先考虑把 `score_width_m` 和 `sim_width_m` 之间的余量适度收窄，而不是重新放宽拒绝逻辑 |
| 覆盖率塑形系数（`coverage_shaping_coef=8.0`）和已有的 `30*improvement`、`waste_coef*4` 三者相对大小没有联合扫过参，可能出现新的偏科（比如疯狂做小行程刷覆盖率增量，不管质量） | A2 引入后，三个系数的相对量级是猜的，不是算出来的 | P0 消融阶段同时看 `coverage_mean` 和 `rmse_mm_mean`、`waste_frac_mean` 三者是否同向变好 | 严格按第五节步骤 4 的 P0 消融门槛执行，不满足就先调系数，不进入整轮训练 |
| `q_feed_path`/`q_scan_path` 的密集插值在个别工位插值失败（`_dense_path` 返回 `None`） | C1/C2 新增的逐点碰撞检查比原来的单帧检查更严格，某些之前"矇过关"的单帧移动现在可能被拒绝 | `feed_path_degraded` 计数、`plan()` 的 `_reject('no dense path back to scan pose')` 触发频率 | C1 装料路径是固定的、只需要离线验证一次；C2 回扫描路径按刀算，若某一刀的退墙姿态导致插值失败，正确行为是让这一刀走 `_project()`/不可达分支重新规划，**不应该静默退化成单帧跳转**——本文档 C2 的伪代码里 `_dense_path` 返回 `None` 时直接 `_reject`，这一点在实现时必须保留，不能为了"怕拒绝率升高"而悄悄放宽 |
| 评分区/仿真区拆分后，`obs_dim`（策略网络输入维度）会随仿真区变大而增大（`coarse_shape` 是固定比例的 2×4 池化，不是固定目标分辨率，见 `sensor.py:41-42`） | B2 之后 `wall_shape` 变大 | `probe.obs_dim` 在 `train.py:make_env`/`PPO()` 构造时的实际取值 | 本身不是 bug，`train.py` 已经动态读取 `probe.obs_dim` 构造网络，不存在硬编码维度；**唯一要注意的是 v0.6 已保存的任何检查点（如 `policy_ppo_best_val.npz`）不能直接拿来给 v0.7 warm-start**，v0.7 必须从教师重新收集数据、重新 BC，这本来就是既定流程，不是额外负担 |
| `MIN_COVERAGE_RATIO=0.90` 门槛设得过严，导致即使奖励塑形方向正确，也因为 DAgger-BC 基线本身覆盖率就不稳定（不同 seed 波动较大）而永远卡在门槛外 | A6 | 对比 `dagger_bc` 在测试集 vs 验证集上的 `coverage_mean` 波动幅度 | 如果发现 DAgger-BC 基线覆盖率本身方差就超过 10%，`MIN_COVERAGE_RATIO` 应该按基线的实际波动幅度重新标定（例如改成"不低于验证集观察到的基线下界"），而不是拍一个固定数字；这个判断留到 P1 整轮训练拿到真实数据后再定 |
| 同一轮改动太多（B+C+A 三组一起上），一旦晋级失败，难以归因是哪一组改动的问题 | 第五节的整体策略 | 每组改动在进入 P1 整轮训练前都有独立的验证步骤（第五节步骤 1-4），失败时先查各组的独立验证记录 | 严格按第五节顺序执行并保留每一步的中间产物（`work_area.json` 新旧对比、回放录像、P0 消融日志），不要跳步直接上整轮训练 |
| 教师 `teacher_edge_margin_m` 从 0.02 降到 0.003 后，教师演示可能偶尔在真正的物理边界（仿真区之外）失败率上升 | B5 | `collect_teacher()`/`collect_dagger()` 阶段的教师动作被 `executor.plan()` 拒绝后走 `profiles`/`shrink` 回退（`env.py:teacher_action():277-294`）的触发频率 | 这条回退逻辑本来就存在且是为了应对这种情况设计的，不需要新代码；只需要在 P0/P1 阶段的日志里额外看一眼这个回退触发率有没有异常升高 |

---

## 七、与既有约束的对照（沿用前两份文档的合规声明）

- 全部改动集中在 `dummy_loop/wall_cycle/{config.py, env.py, arm.py, material.py, reach_table.py, area.py, train.py}`
  和 `tools/simulation/v2_trowel_reach.py`，均为仿真/训练代码，**不触碰 `configs/` 下任何已标定文件**。
- 两条需要重新运行的命令（工作区重扫描、reach table 重建）都是纯 MuJoCo 运动学/碰撞几何分析，不含任何
  `--enable`/驱动指令，证据等级维持 L1，产生方式与现有记录完全一致，由你自行在实验室电脑上运行。
- 不修改 `mortar.py` 的材料物理规则本身；"允许越界不扣分"和"覆盖率塑形"都是评分范围和奖励系数层面的调整，
  不是往物理模型里加行为规则。
- 不删除或覆盖任何既有 `experiments/` 记录，包括 v0.5/v0.6 已有的失败记录。
- 本文档为方案设计，不包含任何实机相关指令；训练本身全程在 MuJoCo 仿真中进行，不涉及硬件运动。
