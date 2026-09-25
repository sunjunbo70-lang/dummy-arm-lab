# 实验总目录

实验采用“大版本 → 修订/阶段 → 方案、记录、训练输出、运行入口”的布局。所有资料已实际迁入，根目录不再设置launchers。

- [00 初期调试与验证](00_initial_debug/README.md)
- [v0.1](v0.1/README.md)
- [v0.2](v0.2/README.md)
- [v0.3](v0.3/README.md)
- [v0.4](v0.4/README.md)
- [v0.5](v0.5/README.md)
- [v0.6](v0.6/README.md)
- [v0.7](v0.7/README.md)
- [v0.8](v0.8/README.md)
- [v0.9](v0.9/README.md)

## 使用规则

每个修订目录：design为设计方案，records为原始记录及诊断，runs为数据/检查点/指标/回放，scripts下按training、visualization等分类。具体失败与成功以各轮报告为准。共享源码dummy_loop、工具tools、机器人模型models不复制到每个版本。

新修订先建立版本目录与README，所有输出通过--out写入本修订runs下的新运行名，不得覆盖旧结果。不要从旧的历史绝对路径直接启动。

大体积runs默认不进入Git；跨电脑继续实验还需复制相应runs数据和本地许可模型并重建环境，仅git clone不含这些结果。

迁移清单与文本原件备份见[_organization/2026-09-25](_organization/2026-09-25/)，历史jsonl及二进制内容未改。

## 新实验设计

- [v0.10：材料感知自主修整与GPU批量强化学习](v0.10/README.md)（r0设计稿，未启动训练）。
