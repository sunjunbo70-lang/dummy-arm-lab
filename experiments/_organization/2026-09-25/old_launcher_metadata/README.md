# 启动器目录

所有入口按用途分类；根目录只保留 `启动中心.cmd`。

- `simulation/`：纯仿真和模型检查，不访问串口。
- `training/`：离线训练入口。
- `replay/`：原生 MuJoCo 结果回放。
- `hardware/`：可能访问实机的工具，必须遵守 `AGENTS.md`。
- `archive/`：历史只读回放。

启动器必须从自身位置解析仓库根目录。新增入口时同步更新 `catalog.json`。
