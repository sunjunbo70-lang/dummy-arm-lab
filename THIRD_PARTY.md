# 第三方来源与许可

本项目以 GPL-3.0 发布（见 `LICENSE`），因为它在 `switchpi/dummy` 的 GPL-3.0
模型与固件参考基础上迭代。下表列出每一项外部来源的出处、许可状态与本仓库的处理方式。

## 随仓库分发

| 内容 | 来源 | 许可 | 说明 |
| --- | --- | --- | --- |
| `models/dummy_reference.xml`、`models/reference_import.urdf`、`models/meshes/` | [`switchpi/dummy`](https://github.com/switchpi/dummy) 分支 `auk`，提交 `3a9d174` | GPL-3.0 | 原始许可文本保留在 `models/UPSTREAM_LICENSE`；逐文件 SHA-256 记录在 `models/provenance.json` |
| `vendor/native_client/fibre/` | Fibre 传输库（ODrive 生态） | 见 `vendor/native_client/NOTICE.md` | 实机原生 USB 通信必需；本项目修复了其超时行为，改动已在该目录说明 |

## 不随仓库分发，需自行获取

| 内容 | 来源 | 为什么不分发 |
| --- | --- | --- |
| `models/studio_meshes/`（约 84 MB STL） | 从 DummyStudio 的 Unity 资源提取 | 上游 [`peng-zhihui/Dummy-Robot`](https://github.com/peng-zhihui/Dummy-Robot) **未声明许可**，其 README 说明上位机暂无开源计划。无许可声明即默认保留所有权利，公开再分发缺乏依据 |
| `models/studio_source/`（原始 OBJ、IL 文本、参数图） | 同上 | 同上 |
| Dummy 使用指南 V3.0.7 的文本与页面渲染 | 厂商文档 | 第三方文档著作权 |

本仓库保留的是**我们自己的处理代码与溯源记录**：提取脚本
`tools/modeling/build_studio_model.py`、装配描述 `models/dummy_studio_visual.xml`、
以及 `models/studio_provenance.json` 中每个部件的 `source_sha256` 与 `stl_sha256`。
持有 DummyStudio 副本的人可以据此在本地重建出字节一致的模型，并用哈希自证。

依赖这些资产的测试在数据缺失时自动跳过，参考仿真链路不受影响。

## 如何重建 Studio 可视化模型

```bash
# 把你自己的 DummyStudio 提取物放到 models/studio_source/
python tools/modeling/build_studio_model.py
# 用 models/studio_provenance.json 中的哈希校验结果
```

## 再分发提醒

以 GPL-3.0 再分发本仓库时，请连同 `LICENSE`、本文件和 `models/provenance.json`
一并保留。若你在自己的副本中加入了上表"不随仓库分发"的内容，
公开前需自行确认其许可状态。
