# VLA（视觉-语言-动作）与机器人模仿学习 · 一周学习计划

> 目标：一周后能理解 VLA 领域全貌，掌握 InternVLA-M1 / OpenPI 微调实验流程，能独立设计实验方案。
> 原则：每天上午看视频，下午读论文/跑代码，晚上整理笔记。

---

## 学习前准备（所有资源链接汇总）

### 📺 视频教程

| 资源 | 链接/搜索关键词 | 语言 | 时长 |
|------|------|:--:|:--:|
| **具身智能VLA教程（清华）** | B站搜索 "具身智能VLA 清华" | 中文 | 72 min |
| **逐篇解析VLA经典论文** | [B站 BV1q6RzYnENi](https://www.bilibili.com/video/BV1q6RzYnENi/) | 中文 | ~40 min |
| **Stanford CS224R 2025 DRL** | [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL) | 英文/中文字幕 | 每讲~75 min |
| **Isaac Sim 仿真全集** | [B站 BV1fMoeBPEwM](https://www.bilibili.com/video/BV1fMoeBPEwM/) | 中英字幕 | 系列 |
| **KUKA工业机器人入门** | [B站 BV1T5411N7zd](https://www.bilibili.com/video/BV1T5411N7zd/) | 中文 | ~3h 系列 |
| **机械臂理论入门** | [B站 BV19z4y197cf](https://www.bilibili.com/video/BV19z4y197cf/) | 中文 | ~2h |
| **LeRobot 环境搭建实战** | [B站 BV11L7JzRExs](c) | 中文 | ~30 min |
| **RSS 2024: Supervised Policy Learning** | [YouTube](https://www.youtube.com/watch?v=jIB_joS7ww8) | 英文 | ~3h |

### 📺 论文中文讲解视频（按热度排序）

| 论文 | 视频 | 链接 | 时长 |
|------|------|------|:--:|
| **π₀** | pi0模型与核心源码分析(上) | [B站 BV1D4L9ztE6B](https://www.bilibili.com/video/BV1D4L9ztE6B/) | 22 min |
| **π₀** | π系列EP01：Flow Matching+VLA详解 | [B站 BV1SEdWBXEqj](https://www.bilibili.com/video/BV1SEdWBXEqj/) | ~20 min |
| **π₀.₅** | π0.5: 开放世界VLA智能系统 | [B站 BV1osLqzREHm](https://www.bilibili.com/video/BV1osLqzREHm/) | ~15 min |
| **π₀→π_RL** | 青稞Talk：从π₀到π_RL的强化学习微调 | [B站 BV1Tt2sBPEix](https://www.bilibili.com/video/BV1Tt2sBPEix/) | ~40 min |
| **π₀.₆** | π*0.6: RL Recap提升VLA性能 | [B站 BV13pyGBwEMH](https://www.bilibili.com/video/BV13pyGBwEMH/) | 33 min |
| **Diffusion Policy** | 精讲Diffusion Policy扩散策略 | [B站 BV1WZHseoEEF](https://www.bilibili.com/video/BV1WZHseoEEF/) | ~30 min |
| **Diffusion Policy** | 最适合入门的Diffusion Policy | [B站 BV1MtXHYUE6M](https://www.bilibili.com/video/BV1MtXHYUE6M/) | ~20 min |
| **VLA多篇串联** | 逐篇解析VLA经典论文 | [B站 BV1q6RzYnENi](https://www.bilibili.com/video/BV1q6RzYnENi/) | 47 min |
| **VLA全局** | 具身智能VLA教程（清华） | [B站搜"具身智能VLA 清华"](https://search.bilibili.com/all?keyword=%E5%85%B7%E8%BA%AB%E6%99%BA%E8%83%BDVLA) | 72 min |

### 📝 论文中文文字解读

| 论文 | 链接 |
|------|------|
| RT-2 论文解读 | [知乎](https://zhuanlan.zhihu.com/p/651670131) |
| Octo 论文解读 | [知乎](https://zhuanlan.zhihu.com/p/717370085) |
| Diffusion Policy 详解 | [知乎](https://zhuanlan.zhihu.com/p/670555655) / [CSDN](https://blog.csdn.net/v_JULY_v/article/details/143651718) |
| InternVLA-M1 解读 | [知乎](https://zhuanlan.zhihu.com/p/1962112316918178822) |
| VLA+RL 融合方向汇总 | [知乎](https://zhuanlan.zhihu.com/p/1943256525368955961) |

> ⚠️ **缺口提示**：InternVLA-M1 目前无独立视频讲解（架构已由 π₀+Diffusion Policy 视频覆盖）；SmoothVLA 太新（2026.03）暂无中文视频。

### 📄 必读论文

| 论文 | 年份 | 重要性 |
|------|:--:|:--:|
| **视觉-语言-动作模型综述（自动化学报）** | 2025 | ⭐⭐⭐ 中文综述，入门必读 |
| RT-2: Vision-Language-Action Models | 2023 | ⭐⭐⭐ VLA 开山之作 |
| Octo: An Open-Source Generalist Robot Policy | 2024 | ⭐⭐ 扩散策略路线代表 |
| OpenVLA: An Open-Source VLA Model | 2024 | ⭐⭐ 开源 VLA 标杆 |
| π₀: A VLA Flow Model for General Robot Control | 2024 | ⭐⭐⭐ Flow Matching 路线 |
| InternVLA-M1 论文 | 2025 | ⭐⭐⭐ 你的目标 Backbone |
| SmoothVLA 论文 | 2026 | ⭐⭐ 平滑性优化方法 |

### 🛠️ 代码仓库

| 仓库 | 用途 |
|------|------|
| [InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1) | 目标 Backbone |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | 备选 Backbone |
| [LeRobot](https://github.com/huggingface/lerobot) | 数据处理 & 策略训练框架 |
| [OpenVLA](https://github.com/openvla/openvla) | 参考（SmoothVLA 也是基于它） |

### 📄 中文综述直达链接

- 自动化学报 VLA 综述：https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c250417

---

## 每日计划

### 第 1 天：VLA 是什么？从零建立全局认知

**上午（2-3h，看视频）：**

1. 🎬 B站「具身智能VLA教程（清华）」— 72 min
   - 搜索关键词 "具身智能VLA 清华"
   - 了解 VLA → 世界模型的演进路径
   - 重点听：VLA 的架构分类（VLM-based vs Diffusion-based vs Flow Matching）

2. 🎬 B站「逐篇解析VLA经典论文」— ~40 min
   - 建立对 RT-2、Octo、π₀ 的直观理解

**下午（2-3h，读论文）：**

3. 📄 《视觉-语言-动作模型综述：从前史到前沿》（自动化学报 2025）
   - 中文，读完前 10 页（发展历史 + 分类框架）
   - 画出「VLA 家族树」：RT-2 → Octo → OpenVLA → π₀ → InternVLA-M1 → π₀.₇

4. 📄 RT-2 论文 — 精读 Abstract + Figure 1-3
   - https://robotics-transformer2.github.io/
   - 辅助阅读：[知乎中文解读](https://zhuanlan.zhihu.com/p/651670131)
   - 理解「把动作当文本 token」这个核心思想

**晚上（30 min）：**

- 整理笔记：用自己的话写一段 200 字的 VLA 定义
- 列出你不懂的 3-5 个术语（留着后面几天解决）

---

### 第 2 天：深入 VLA 三大技术路线

**上午（2h，读论文）：**

1. 🎬 **「π系列EP01：Flow Matching+VLA详解」** [B站 BV1SEdWBXEqj](https://www.bilibili.com/video/BV1SEdWBXEqj/) — ~20 min
   - 理解 Flow Matching 的原理和为什么比离散 token 更平滑
   - 对比：RT-2（离散化） vs π₀（连续 Flow Matching）

2. 🎬 **「精讲Diffusion Policy扩散策略」** [B站 BV1WZHseoEEF](https://www.bilibili.com/video/BV1WZHseoEEF/) — ~30 min
   - 理解了扩散模型怎么生成动作序列，才能理解 Octo 和 InternVLA-M1 的 DiT
   - 如果觉得太难，先看「最适合入门的Diffusion Policy」[B站 BV1MtXHYUE6M](https://www.bilibili.com/video/BV1MtXHYUE6M/)

3. 📄 π₀ 论文 + Octo 论文对照阅读
   - π₀: https://www.pi.website/blog/pi0
   - Octo: https://octo-models.github.io/
   - 重点：两种「去噪」思路的相同与不同

**下午（2-3h，动手）：**

3. 🖥️ 在你的 3090 上验证环境
   ```bash
   conda create -n vla_study python=3.10 -y && conda activate vla_study
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.__version__)"
   ```

4. 🖥️ 下载 InternVLA-M1 权重并测试推理
   ```bash
   git clone https://github.com/InternRobotics/InternVLA-M1.git
   cd InternVLA-M1
   pip install -e .
   huggingface-cli download InternRobotics/InternVLA-M1 --local-dir ./weights
   ```

5. 📄 快速浏览 InternVLA-M1 论文的 Architecture 部分
   - 理解：System 2 (VLM) + System 1 (DiT) 双系统架构

**晚上（30 min）：**

- 画一张图：三种动作生成方式（离散 Token / 扩散 DDPM / Flow Matching）的区别

---

### 第 3 天：模仿学习基础 + 数据处理

**上午（2h，看视频）：**

1. 🎬 Stanford CS224R 2025 — Imitation Learning 讲
   - YouTube 搜索 "CS224R Imitation Learning 2025"
   - 重点：Behavior Cloning、DAgger、分布偏移问题

2. 🎬 RSS 2024: Supervised Policy Learning for Real Robots（前 1h）
   - 重点：Action Chunking、Temporal Ensembling

**下午（2-3h，读 + 动手）：**

3. 📄 理解 RLDS 数据格式（Open X-Embodiment 的标准格式）
   - 你的 MoCap → KUKA 轨迹需要转成这个格式

4. 🖥️ 安装 LeRobot 并跑示例
   ```bash
   git clone https://github.com/huggingface/lerobot.git
   cd lerobot
   pip install -e .
   ```

**晚上（30 min）：**

- 列出你的数据 pipeline：MoCap → 重定向 → IK 验证 → RLDS

---

### 第 4 天：机械臂控制基础 + KUKA 实操

**上午（2h，看视频）：**

1. 🎬 B站「机械臂理论入门」— 2h
   - 运动学基础（DH 参数、正/逆运动学）
   - 末端位姿表示

2. 🎬 B站「KUKA工业机器人入门」— 前 3-4 集（~1h）
   - 理解 KUKA 的动作空间、坐标系

**下午（2-3h，动手）：**

3. 🖥️ PyBullet 仿真验证
   ```bash
   pip install pybullet
   ```
   - 加载 KUKA 模型，手动给定末端位姿 → IK 求解

4. 📄 学习 ROS 基本概念（Topic/Service/Action，MoveIt 流程）

**晚上（30 min）：**

- 整理 KUKA 技术参数：工作空间、关节限位、通信协议

---

### 第 5 天：仿真 + InternVLA-M1 微调实战

**上午（2h，看视频）：**

1. 🎬 B站「Isaac Sim 仿真全集」— 前 3 集
   - 理解仿真在 VLA 训练中的角色
   - Domain Randomization、Sim-to-Real Gap

**下午（3-4h，核心动手环节）：**

2. 🖥️ 尝试 InternVLA-M1 微调（最小规模验证）
   - 先用 5-10 条公开数据验证流程
   - 目标：走通整个微调流程，不是训练出有用的模型
   - 记录：3090 显存占用、训练速度、Loss 趋势

3. 🖥️ 如果 InternVLA-M1 暂时跑不通：
   - 用 LeRobot 训练一个 Diffusion Policy（pusht 或 aloha 数据集）
   - 同样走通全流程

**晚上（30 min）：**

- 记录踩坑日志

---

### 第 6 天：领域全景 + 你的实验方案设计

**上午（2h，读论文）：**

1. 📄 InternVLA-M1 论文精读 — Method + Experiment
   - 辅助阅读：[知乎文字解读](https://zhuanlan.zhihu.com/p/1962112316918178822)
2. 📄 SmoothVLA 论文 — Reward Design（Jerk 计算 + GRPO 流程）
3. 🎬 **「青稞Talk：从π₀到π_RL的强化学习微调」** [B站 BV1Tt2sBPEix](https://www.bilibili.com/video/BV1Tt2sBPEix/) — ~40 min
   - 这是理解 SmoothVLA 的 RL 微调思路的最佳中文视频
   - 虽然讲的是 π₀ 的 RL 微调，但 GRPO + jerk 惩罚的思路是通用的
   - 可选补充：**「π*0.6: RL Recap提升VLA性能」** [B站 BV13pyGBwEMH](https://www.bilibili.com/video/BV13pyGBwEMH/) — 33 min

**下午（2-3h，设计实验）：**

3. ✍️ 撰写实验方案草稿
   - 数据准备：MoCap → KUKA 轨迹重定向 → RLDS
   - Backbone：InternVLA-M1
   - 训练策略：SFT + Jerk 正则化 → （可选）GRPO
   - 评估指标：Jerk、SAM2 覆盖率、轨迹完成率
   - 创新点：SAM2 空间提示 + 人体技能迁移 + Jerk 约束

**晚上（30 min）：**

- 检查方案里有没有明显的技术漏洞

---

### 第 7 天：查漏补缺 + 路线确认

**上午（2h）：**

1. 📄 重读 VLA 综述的后半部分（前沿方向：WAMs、人形机器人、仿真数据生成）
2. 🎬 补看之前跳过的视频章节
   - 想深入理解 π₀ 源码：看「pi0模型与核心源码分析(上)」[B站 BV1D4L9ztE6B](https://www.bilibili.com/video/BV1D4L9ztE6B/)（22 min）
   - 想了解最新进展：看「π0.5: 开放世界VLA」[B站 BV1osLqzREHm](https://www.bilibili.com/video/BV1osLqzREHm/)（15 min）
   - 想了解RL微调实战：重看青稞Talk [B站 BV1Tt2sBPEix](https://www.bilibili.com/video/BV1Tt2sBPEix/)（40 min）

**下午（2-3h）：**

3. ✍️ 最终确定实验路线图
   - 本周能开始的：环境搭建、数据格式转换
   - 下周能开始的：SFT 微调（小规模）
   - 一个月目标：完整的 SFT + KUKA 初步验证

4. 📝 写一份 1 页的「实验可行性报告」
   - 硬件、数据、模型、风险

**晚上（1h）：**

- 整理本周所有笔记
- 标记 10 个「关键理解」+ 5 个「待深入问题」

---

## 必备工具链速查

| 工具 | 用途 | 安装命令 |
|------|------|---------|
| PyTorch 2.x + CUDA | 深度学习框架 | `conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia` |
| LeRobot | 机器人数据 & 训练框架 | `pip install lerobot` |
| HuggingFace Hub | 模型下载 | `pip install huggingface_hub` |
| PyBullet | 轻量物理仿真 | `pip install pybullet` |
| Isaac Sim（可选） | 高保真仿真 | NVIDIA Omniverse Launcher |
| Weights & Biases | 训练监控 | `pip install wandb` |

---

## 论文阅读优先级

```
必读（影响你的实验设计）：
  1. InternVLA-M1（你的 Backbone）
  2. π₀（理解 Flow Matching）
  3. 视觉-语言-动作模型综述（中文，全局认知）

选读（拓宽视野）：
  4. SmoothVLA（Jerk 惩罚方法）
  5. RT-2（VLA 开山之作）
  6. Octo（扩散策略路线）
  7. ForceVLA（力感知 VLA，接触任务相关）

参考（引用用）：
  8. OpenVLA
  9. Diffusion Policy
  10. ACT
```

---

## 一周后的检验标准

一周结束时，你应该能回答以下问题：

1. ✅ VLA 的三种动作生成方式（离散 Token / 扩散 / Flow Matching）各有什么优缺点？
2. ✅ InternVLA-M1 的双系统架构中，System 2 和 System 1 分别做什么？
3. ✅ 你的 MoCap 数据要经过哪些步骤才能喂给 InternVLA-M1？
4. ✅ 为什么 SFT 之后的 RL 微调能让轨迹更平滑？
5. ✅ 3090 上跑 InternVLA-M1 微调，显存够不够？（用实际测试回答）
6. ✅ 你的项目中，SAM2 分割掩码如何与 VLA 的空间定位能力结合？（能画架构图）
7. ✅ 如果实验结果不好，可能的三个原因是什么？

---

*计划制定日期：2026年7月*
*欢迎根据实际进度动态调整*
