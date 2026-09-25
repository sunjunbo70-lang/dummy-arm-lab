# 强化学习动作可视化

双击仓库根目录的 `强化学习动作可视化.cmd`。已生成结果时直接打开离线播放器；
没有结果时加载仓库中的两份已训练 PPO 权重，运行三刀仿真并渲染，约一分钟。
不需要重新训练，不连接硬件。依赖现有仿真环境与 `requirements/gui.txt` 中的 Pillow。

播放器同时展示机械臂全景、跟随当前条带的抹刀侧面近景、墙面厚度展开图，
以及实际刀面俯仰、后缘间隙、受力与剩料。支持切换模型、暂停、重播、逐帧和 0.25–2 倍速度。
默认 0.5 倍慢放循环。`index.html` 内嵌所有画面，断网也能直接打开。

重新生成：

```powershell
.venv-loop/Scripts/python.exe tools/simulation/open_rl_visualization.py --rebuild
```

自选权重，输出目录必须不存在（避免覆盖证据）：

```powershell
.venv-loop/Scripts/python.exe -m dummy_loop.wall.visualize --policy experiments/v0.1/r0/runs/rl/policy.npz --out experiments/v0.1/r0/runs/my_new_replay
```

每个模型子目录包含：`preview.gif` 动画、`final.png` 最终画面、`rollout.npz` 逐控制步真实状态、
`report.json` 指标、模型 SHA-256 与环境说明。NPZ 包含关节位置、厚度场、物理动作、
动作后的观测、条带编号与墙面网格。每条带第一条记录是 reset 状态（step=0，action 为零占位）；
其余记录都是 PPO 动作执行后的真实状态。视频帧另含标注过渡的停顿和末尾展示停顿，
因此显示帧数大于控制步数；每个常规帧对应 0.05 s。

渲染中的材料几何仅加入 MuJoCo 的可视化场景，不参与碰撞，不更改策略输入或动力学。
材料块的厚度使用模拟高度场原值，没有放大。下方颜色图固定使用 0–4 mm 标尺，
便于观察薄层差别；覆盖率沿用训练定义：厚度达到目标 2 mm 的一半即算覆盖。

所有结果为 **L1 软件仿真**。材料为降阶代理，不是流体仿真；参数未经实測。
每刀抹涂是 PPO 控制，换条带和上料沿用 `simulate_transit=False` 的重置，
画面明确标注切换，未将插值动画冒充动力学过渡。

本次复现：手法预热覆盖率 0.9796、厚度 RMS 0.371 mm；平刀预热 0.9796、0.819 mm。
这些结果与已有训练对比记录一致。
