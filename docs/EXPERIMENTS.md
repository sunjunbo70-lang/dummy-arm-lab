# 实验管理规范

## 已有基线

experiments/2026-09-16_baseline/raw/ 保存整理时 outputs/ 全部117个普通文件的原样副本，包括成功、失败、截图、模型与数据。manifest.json包含原路径、大小、SHA-256；原outputs保留以兼容旧命令。重复副本用于冻结证据，不代表新增实验。

- reference_learning：合成教师数据、线性policy和rollout。
- verified_poses：有用户确认或完整反馈报告的姿态相关记录；同类失败报告仍保留，需读result字段。
- commissioning_and_failures：串口、原生USB试验及中止记录，不能整体视为成功。
- studio_protocol：Studio抓取、IL、Transform和连接失败记录。
- visual_validation：模型图片与GIF（不是实机录像）。
- machine_specific_usb：旧机器注册表备份与修复结果；仅本机历史证据。

## 新实验

从experiments/_template复制到YYYY-MM-DD_short-name，填写run.json和README。每次运行使用唯一目录，保留命令、环境、配置、随机种子、数据划分、模型哈希、训练/推理耗时、显存、结果和失败原因。不要覆盖baseline或同名权重。

真实数据至少说明单位、坐标系、控制频率、时间戳来源、实际使用硬件和操作是否有监督。用户肉眼确认、固件反馈到位、独立测量精度分别记录。

## 校验

`python tools/maintenance/verify_evidence.py` 只校验归档文件，不打开硬件。
后续如需论文复现，必须单独定义指标、测试集和重复次数；本项目的冒烟测试不等于科研性能评估。
