# 新实验修订模板

新建 experiments/vX.Y/rN/；不要再新建日期平铺目录或根launchers。

- design/：审查后的方案、变更理由、验收门槛。
- records/：日期命名的过程记录、测试、失败原因与修复证据。
- runs/<唯一运行名>/：config、checkpoints、metrics、logs、evaluation、records等，运行名不可复用覆盖。
- scripts/training/：启动训练；scripts/visualization/：原生回放；入口从自身目录定位仓库根目录。
- README.md：当前状态、完整结果入口、失败/缺项、复现步骤及共享源码版本。

完成或中止均保留原始日志，不用完成标记替代质量验收。大型runs单独备份，不默认进入Git。
