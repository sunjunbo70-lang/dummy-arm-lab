# vendor/native_client

本目录是 Fibre 传输库的 vendored 副本，用于与 Dummy 控制板的原生 USB 接口通信。
它是实机角度反馈链路的必需组件，因此随仓库分发而非作为 pip 依赖——
上游同名包的版本与本设备固件的兼容性未经确认。

## 本项目的改动

`fibre/protocol.py` 与相关调用点修复了重发超时行为：原实现在通道无响应时
可能无限等待，表现为折叠动作中途卡死。改动记录见
`docs/history/2026-09-16_折叠中断与反馈超时修复.md`。

调用方 `dummy_loop/live_control.py` 另外设置了
`_resend_timeout = 0.15` 与 `_send_attempts = 2`。

## 许可

Fibre 由 ODrive Robotics 及贡献者开发。本副本随
`switchpi/dummy`（GPL-3.0）一并取得。使用者在独立再分发本目录前，
应核对 Fibre 上游仓库的当前许可文本并保留相应声明。

本项目整体以 GPL-3.0 发布，见仓库根目录 `LICENSE`。
