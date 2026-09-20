# 墙面抹涂仿真链路：场景、控制、任务、示教数据、误差与补偿

目的：停电、实验室关门期间，在不接实机的前提下把抹涂任务的仿真链路搭通并量化，
为周一起的实机实验提前做出设计决定（墙板位置、转接件形式、需要加装什么传感器、先测什么）。

会话类型：软件会话。未连接机械臂，未打开串口，未执行任何带 `--enable` 的命令。`hardware_motion: false`。
证据等级 **L1**。所有数值基于参考模型与假设参数，只能指导设计与下一步测量，不能预测实机表现。

输入数据与版本：
- 仓库 `main` @ `285028445081c370dc53779203177e3bcc0522bc` + 本次新增的 `dummy_loop/wall/`、`dummy_loop/episode.py`
- `models/dummy_reference.xml`（未修改；由 MjSpec 在其上组装场景）
- 场景参数见 `raw/scene/wall_trowel_scene.config.json`，每一项的来源（假设 / 占位 / 测量）写在 `provenance` 里
- 关节范围假设 ±90°；伺服 kp/kv 沿用参考模型示例值；抹刀尺寸为占位

执行命令和环境（云端沙箱 Linux、Python 3.11、numpy 2.4.4、mujoco 3.13.0、MUJOCO_GL=egl）：

```
python -m unittest discover -s tests                                   # 66 个测试，3 个按设计跳过
python -m dummy_loop.wall layout  --out experiments/2026-09-20_wall_sim_chain/raw/layout.json
python -m dummy_loop.wall study   --out experiments/2026-09-20_wall_sim_chain/raw --random 30
python tools/simulation/plot_wall_study.py experiments/2026-09-20_wall_sim_chain/raw/sensitivity.json
python -m dummy_loop.wall collect --episodes 4 --out experiments/2026-09-20_wall_sim_chain/raw/episodes --random-scale 1 --probe --servo --seed 100
python -m dummy_loop.wall replay  experiments/2026-09-20_wall_sim_chain/raw/episodes/ep_000N.npz          # N = 0..3
python -m dummy_loop.wall scene   --out experiments/2026-09-20_wall_sim_chain/raw/scene/wall_trowel_scene.xml
python -m dummy_loop.wall render  --out experiments/2026-09-20_wall_sim_chain/raw/camera
```

## 结果与证据

| 项 | 结果 | 证据 |
| --- | --- | --- |
| 测试 | 66 个通过（新增 17 个），3 个按设计跳过 | 上述命令 |
| 参考仿真闭环 | 加 ctrlrange 守卫前后逐位相同（本沙箱 0.002226404300892301；Windows 钉死环境下与冻结基线逐位相同，见 2026-09-19_windows_env） | — |
| 工位布局 | 96 个候选中 20 个能覆盖 12 × 10 cm 工作区；最优：刀面倾角 0°、墙距 40 cm、工作区中心高 20 cm，最小奇异值 0.023 m/rad，离限位 21°，腕扭 ≤ 30° | `raw/layout.json` |
| 无误差示教 | 覆盖率 0.958，接触力全部在 2–15 N 窗口内，无触底、无不可达 | `raw/sensitivity.json` → `baseline_no_error` |
| 30 组随机误差 | 平均覆盖率：无补偿 0.907（最差 0.271）→ 探触 0.966（最差 0.852）；探触+闭环最大力最差 11.8 N（无补偿 16.5 N） | `raw/sensitivity.json` → `randomized`，`raw/sensitivity.png` |
| 单因素敏感性 | 最敏感：伺服偏软 kp×0.5（0.73）、摩擦×1.7（0.81）、J2 零位 +2°（0.90）；墙面偏移主要改变压力而非覆盖率 | `raw/sensitivity.json` → `single_factor` |
| 录制与回放 | 4 条带探触+闭环的随机误差 episode，回放后指标全部逐位相同 | `raw/episodes/`，`raw/replay_check.txt` |
| 场景导出 | MJCF 可独立加载（网格用相对路径） | `raw/scene/` |
| 固定相机 | 640×480 RGB + 深度渲染正常；理想针孔模型 | `raw/camera/` |

设计结论与解释见 `docs/SIMULATION.md`。要点：
1. 刀面正对墙面并沿墙水平移动会逼近腕部奇异；控制器将刀面 roll 设为弱约束、关节代价加权、限速。
2. 可达面积最大的墙距（20–30 cm，见 2026-09-20_wall_workspace）并不是好干活的墙距；按条件数选出 40 cm。
3. 覆盖率收益几乎全部来自开工前的探触标定；压缩量闭环主要稳定压力。探触只需要开关量，
   转接件上加一个微动开关即可拿到大部分收益。

## 失败与局限

- 全部参数为假设或占位：关节范围、伺服刚度、摩擦、弹簧、抹刀尺寸、相机位置。
- 未建模：齿隙回差、连杆长度误差、减速器柔性、墙面不平、材料流动、D435 噪声、连杆与墙的碰撞。
- 覆盖率判据是「刀面在 2–15 N 下经过该格」，不是抹涂质量；力窗口本身是假设。
- 随机误差量级（`Perturbation.sample`）是按标定前的合理范围估计的，不是实测分布。
- 本次开发中修正过两个会影响结论的问题，均在最终数字之前：
  (a) 示教以控制器滞后目标为反馈，有延迟时陷入极限环；改为按自身指令累计规划。
  (b) 由 (a) 引起的「无误差时压缩量闭环把覆盖率从 0.958 提到 1.000」，修正后消失，不再作为结论。
- 本沙箱与 Windows 钉死环境的 numpy 版本不同，参考仿真终值在第 13 位有效数字不同；这不是本次改动引起的
  （加守卫前后本沙箱结果逐位相同）。

## 下一步

- M3 实测关节限位后用 `layout` 重算墙板位置与转接件方向。
- 转接件加弹簧滑轨与微动开关；台架上用秤标定 k 与预紧。
- 用秤实测真实抹涂压力与拖曳，更新力窗口与摩擦系数。
- 在 GPU 机器上用 `collect` 的数据训练 ACT。
