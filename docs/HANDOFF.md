# 交接与迁移

## 交付物

- releases/dummy-experiment-runtime-2026-09-18-workspace.zip：当前源码、全部模型、USB客户端、依赖、文档、分类实验结果、兼容入口。排除venv、Git元数据、外部账户配置、历史Studio程序和旧交付包。
- releases/dummy-experiment-history-2026-09-18.zip：整理前源码、旧交付包、历史诊断副本和原始上游归档。作为取证，不作为执行目录。
- 每个zip有SHA-256文件和内容清单。解压运行包后按SETUP重新创建环境，不复制原电脑venv。

在 `D:\VLA` 执行 `python tools/maintenance/package_release.py --tag 新标签` 重新打包；只更新运行包可加 `--kind runtime`；已有同名文件时拒绝覆盖，改用 --tag 新标签。运行包不自动下载外部资料、不写系统注册表、不发送实机命令。

## 完整性和范围

本项目目录中的源码、网格、配置和实验记录均有归档。原始D:/VLA工程的源码、固件、硬件资料、Studio资产和原始压缩包在独立上游zip中；大体积虚拟机镜像及安装器仅列清单，原件仍在原路径。服务器助手仅归档源码和依赖工具，账户配置、known_hosts、preferences和快捷方式排除，目标电脑重新配置。

models/UPSTREAM_LICENSE仅适用于对应来源，不能套用于其他上游。归档包适用于内部交接，公开发布前核对各来源许可和日志中的设备/路径标识。没有把整个混合工程擅自改为统一开源许可。

## 接手者检查

1. 校验zip哈希、解压、建立环境。
2. 运行verify_evidence和单元测试。
3. 离线渲染和GUI预览。
4. 阅读STATUS、HARDWARE，再检查本机设备身份/权限。
5. 在新实验目录继续，不修改既有基线。

旧README_实验闭环.md及docs/history原文保留，可能包含已过时状态；新README优先。2026-09-18整理只验证软件和迁移包，不宣称Linux实机或服务器训练已完成。

## 迁移包自动核验

`python tools/maintenance/verify_release.py releases/dummy-experiment-runtime-2026-09-18-workspace.zip --extract-to "releases/new verification"`

该命令校验压缩包和每个文件的SHA-256，再解压到不存在的新目录，执行离线测试。使用当前Python环境，不安装驱动、不连接机械臂。历史包可用同一工具验证，但不带 --extract-to。

Git尚未建立提交基线；当前完整版本由带文件清单和哈希的release固定。archive、releases、原始diagnostics和当前outputs不进入常规Git；代码与experiments基线可纳入版本控制。后续首次提交前检查第三方许可及体积，模型网格适合用Git LFS管理。
