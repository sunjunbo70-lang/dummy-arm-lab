# M4 统一 Robot API 与 GUI 收编

状态：未开始
会话类型：软件会话为主，验收需硬件会话
目标证据级：L3
前置：M3 完成

## 要解决什么

同一台机械臂的参数现在散在两处，互不知情：

| | `live_control.py` | `configs/*.json` |
| --- | --- | --- |
| 限位 | 硬编码 `LOWER` / `UPPER` | `lower_rad` / `upper_rad` |
| 零位偏置 | 硬编码 `[0,-75,180,0,0,0]` | `firmware_zero_deg` |
| 设备身份 | 硬编码序列号 | `firmware_identity` |

后果：两套数字会漂移；GUI 采的数据与 CLI 跑的策略可能坐标系不一致；
换机械臂要改源码。

## 做什么

- 删除 `live_control.py` 中写死的 `LOWER` / `UPPER` / 序列号 / 偏置，改为从 profile 读
- CLI 与 GUI 共用同一个 `DummyArm`
- GUI 原有调试包络保留，但作为 profile 中的 `debug_envelope` 段
- profile 缺失或未标定时，GUI 可连接与显示，但拒绝开启跟随

## 验收

- `configs/README.md` 中「二者尚未合并成统一设备 profile」可以删除
- 用标定后的 profile 重做一次直立 → 折叠全程，结果不差于 2026-09-16 那次
- GUI 在未标定 profile 下确实拒绝跟随

## 证据落点

`experiments/<日期>_unified_api/`
