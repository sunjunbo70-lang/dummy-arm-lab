# 换到另一台电脑接着做（2026-09-22 写）

给实验室另一端那台电脑：克隆本仓库之后，照这个顺序走一遍，就能接上当前进度。

## 1. 克隆与环境

```bash
git clone <本仓库的 GitHub 地址>
cd dummy-arm-lab
```

Windows 下双击 `tools\environment\windows_setup.cmd`，会建好 `.venv-loop` 并按 `requirements/` 钉死的版本装依赖。
装完后自动跑一遍 `python -m unittest discover -s tests`，应显示 `Ran 125 tests ... OK (skipped=3)`。

先读这三份，再动手：`README.zh-CN.md`（项目总览）→ `docs/STATUS.md`（当前状态表）→ `docs/EXPERIMENTS.md`（全部实验记录索引，含每条的等级与是否被后续取代）。

## 2. 当前进度在哪

最新一轮是「整片连续强化学习 v0.3」：`experiments/2026-09-22_wall_cycle_v03/`。
它用物理化的砂浆模型（不写死任何倾斜角规则，让模型自己学）+ 真实 Dummy V2 机械臂在 MuJoCo 里联合仿真。
完整的改动原因、和上一版（v0.2，`experiments/2026-09-22_wall_cycle_rl/`）逐项对比，在
`docs/changes/2026-09-22_wall_cycle_v0.3.md`——**这份文件专门是为了让接手的人知道"为什么这么改、后续迭代时哪些不能改回去"**，改代码前务必先看。

当前最好的模型：`experiments/2026-09-22_wall_cycle_v03/raw/technique_teacher/policy.npz`（立起手法预热那组，测试集回报 −35.7，优于平刀预热那组）。

双击仓库根目录的 `整片连续强化学习可视化.cmd`，会在 MuJoCo 窗口里播放这个模型在联合仿真下的一局（真实机械臂 + 墙面模型，不是折线示意图）。

## 3. 还没解决的问题（按 `experiments/2026-09-22_wall_cycle_v03/README.md`「局限」排序）

1. 斜向刀次的网格混叠：刀面不与网格对齐时按最近格映射会漏格，墙面厚度图出现棋盘状斑点，抬高 RMSE。下一步先修（刀面足迹超采样）。
2. 每刀起笔处普遍过厚。
3. 砂浆参数（屈服应力、粘度）未标定，来自 Banfill 2003 的量级估计，不是实测。
4. 伺服增益、刀间移动动力学未辨识。
5. 训练量小（每组 61k 步），行为克隆损失偏大，两组 PPO 都没超过各自教师；覆盖率只有 0.22–0.27，没有一局满足完成判定。
6. J5 在作业区中部已接近固件候选限位（−100°），限制了能立起的角度；**M3 逐轴标定后需要重新计算作业区和可达表**（`tools/simulation/v2_trowel_reach.py`、`dummy_loop/wall_cycle/reach_table.py`）。

## 4. 硬约束（照抄 `AGENTS.md`，新会话必须先读那份原文）

- 不执行任何带 `--enable` 的命令，不发送任何会驱动实体的指令。
- 不把任何 `*_verified` 从 false 改成 true。
- 不修改 `configs/` 下已标定文件；新标定写新文件。
- **不删除或改写 `experiments/` 下任何已有记录**，包括失败与中止日志——新结论另开 `experiments/<日期>_<名称>/`。
- 涉及实体运动的步骤，只产出"命令 + 检查清单 + 回滚方式"，由人执行。

## 5. 复现 v0.3 训练（可选，约 50 分钟单核，纯仿真）

```bash
python tools/simulation/v2_trowel_reach.py --pitches 0 --distances 0.28 0.30 0.33 0.35 0.38 --out outputs/wall/reach
python tools/simulation/mortar_sweep.py --out outputs/wall_cycle/mortar_sweep.json
python -m dummy_loop.wall_cycle.reach_table
python -m dummy_loop.wall_cycle.train --out outputs/wall_cycle/v03_flat --teacher-style flat
python -m dummy_loop.wall_cycle.train --out outputs/wall_cycle/v03_technique --teacher-style technique
python tools/simulation/wall_cycle_report.py --flat outputs/wall_cycle/v03_flat --technique outputs/wall_cycle/v03_technique --out outputs/wall_cycle/v03_report
```

新结果写到新的 `experiments/<日期>_<名称>/`，不要覆盖 `2026-09-22_wall_cycle_v03/`。
