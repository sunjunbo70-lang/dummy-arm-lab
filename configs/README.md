# 配置说明

本目录存放设备档案（profile）——描述某一台具体机械臂的标定值、安全包络与外设。
它是物理世界进入软件的唯一入口，也是驱动实机的准入凭证。

## 文件

| 文件 | 作用 |
| --- | --- |
| `schema/hardware_profile.v2.json` | 规则：一份合格的 profile 该长什么样 |
| `hardware.dummy_v2.template.json` | 模板：全 null，M3 标定时复制后填写 |
| `hardware.unverified.json` | v1 遗留档案，保留作历史记录，加载必定失败 |

标定产物应命名为 `hardware.<设备>.calibrated.json`，写新文件，不覆盖模板，
也不要把任何文件改名成已验证配置。文件名本身携带状态：`ls` 一眼就该看出哪份能用。

## 加载即准入

`dummy_loop.core.load_profile` 对档案做的不只是读取：

- `schema_version` 必须为 2。v1 档案以迁移提示失败，不会被降级放行。
- 四个 `*_verified` 任一不为 true 即拒绝加载，`jog` 与 `shadow` 无法运行。
- `firmware_identity` 与 `calibration_record` 必须非空——数字要能指回它的来源。
- `timing` 三项必填；`per_axis_freshness_available` 为 false 时，
  必须写下实测的 `observation_latency_upper_bound_s`。
- `CALIBRATED_FIELDS` 中每个字段都必须在 `provenance` 里有记录，
  含 `method` / `measured_at` / `evidence_path`。

最后一条意味着：**不可能在不说明数字来源的情况下把档案标为已标定。**
这是刻意的——标定值是用实机时间一轴一轴量出来的，它们的出处和数值本身同样重要。

装了 `jsonschema`（见 `requirements/dev.txt`）时还会做完整的声明式校验，
能一次报出全部格式问题而不是只报第一个。未安装时退回手写检查，功能不降级。

## 尚未合并的部分

当前实机上位机的角度映射、范围和设备身份仍定义在 `dummy_loop/live_control.py`，
与本目录的 profile 是两套数字。M4 将把 GUI 收编到同一份 profile，
届时 `debug_envelope` 段接管 GUI 的调试范围，本节可以删除。

## 新设备

新机械臂或新标定保存为独立档案，记录来源、日期和验证范围，不覆盖既有基线。
