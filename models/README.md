# Reference model provenance

Additional model (2026-09-16): `dummy_studio_visual.xml` contains all 17 locally extracted DummyStudio visual parts, including motors. Run `tools/view_simulation.py --studio` for the initially paused assembly viewer. All six visual joints are articulated; J5 ownership and transform hierarchy were corrected and checked against extracted Studio transforms. Dynamics remain provisional and uncalibrated. It is not the default training model. See [电机与仿真参数说明.md](电机与仿真参数说明.md) and `studio_provenance.json` for sources, limitations and parameter findings. The auk license below applies to the auk reference assets, not an assertion of licensing for the separate Studio extraction.

Source: local archive of `https://gitee.com/switchpi/dummy/tree/auk`.
Manifest commit: `3a9d17464308e1532b3d648a80da6db899cc1f01` (as recorded in the supplied archive manifest).

Source description: `ros2/dummy_ws/src/dummy-ros2_description/urdf/dummy-ros2.xacro` and its seven STL meshes.
The upstream GNU GPL v3 license is reproduced in `UPSTREAM_LICENSE`; retain this notice and the supplied source/provenance when redistributing these derived assets.

`reference_import.urdf` is the transformed URDF. `dummy_reference.xml` is the MuJoCo conversion. `provenance.json` records source hashes. The conversion algorithm is in `tools/import_reference.py`.

Changes made 2026-09-16: remove ROS/xacro includes; replace mesh paths with portable relative paths; add 1e-6 to inertia diagonals and apply importer inertia balancing; retain separate bodies; add six demo position servos, damping and armature; enable ideal gravity compensation; disable contacts; add floor/light. These modifications exist only in this project copy.

This is an auk reference, not a calibrated Dummy V2 digital twin. Joint zero/mapping, limits, geometry, mass, inertia, actuation and gripper must be matched to the actual hardware. No collision, contact, grasping, or sim-to-real validation is claimed.

## 2026-09-18 基线拆分

`studio_source/` 在基线中只保留测试与文档依赖的小文件：

| 文件 | 依赖方 |
| --- | --- |
| studio_verified_transforms.json | tests/test_studio_articulation.py |
| robot_hierarchy.json | tools/modeling/build_studio_model.py |
| studio_bridge_rotation_il.txt | 桥接代码核对记录 |
| mini8_parameters.JPG、mini11_parameters.JPG | 电机与仿真参数说明.md |
| 128_default_32.obj | 体积最小的样例部件 |

105 MB 的 Unity 原始 OBJ 导出留在 `_archive/studio_source/`。它们是中间产物：
输入（DummyStudio 安装包）在 `_archive/dummy_v2/upstream/` 内，
输出（`studio_meshes/*.stl`）在本基线内，两端都已保存。

对应关系可逐件核对：`studio_provenance.json` 中每个部件记录了
`source_sha256`（原 OBJ）与 `stl_sha256`（生成的 STL）。
需要重建模型时，把归档中的 OBJ 复制回 `studio_source/` 再运行
`tools/modeling/build_studio_model.py`。

## 换机器后 Studio 外观网格缺失（2026-09-20）

`models/studio_meshes/`（21 个 STL）被 `.gitignore` 排除，克隆到新机器时不存在。此时：

- `六轴同时往复仿真.cmd`（`view_simulation.py --studio`）会提示后改用 `dummy_reference.xml` 显示；
- `MuJoCo实机同步.cmd`（`tools/gui/live_mujoco.py`）改用参考模型显示，窗口标题与顶部黄条会注明。
  两份模型的 J1、J4、J6 轴向相反，显示时已按 `dummy_loop.sim_backend.STUDIO_TO_REFERENCE_SIGN`
  换算，保证转向一致（`tests/test_studio_fallback.py` 用两份模型文件逐轴比对锁定这一点）。外形与尺寸仍不同。

恢复完整外观：从原开发机复制 `models/studio_meshes/` 整个文件夹到本仓库同一位置，然后运行
`python tools/maintenance/verify_studio_meshes.py`，21 个文件的 SHA-256 与 `studio_provenance.json` 全部一致即可。

## Dummy V2 模型（2026-09-20，当前仿真与上位机画面使用）

`dummy_v2.xml` + `meshes_v2/`（7 个 STL，约 4.3 MB）+ `dummy_v2_params.json` + `v2_provenance.json`，
由 `tools/modeling/build_dummy_v2.py` 从 V2 装配体 STEP 生成，**不要手改**。

- 运动学严格等于 V2 固件 DH（`tests/test_dummy_v2_model.py` 逐位核对）；模型零位 = 固件 HOME；
  世界系 = 固件基座系（+X 朝前）。上位机画面 `q = deg2rad(固件角 − HOME)`，不再需要符号表。
- 末端是 J6 电机裸轴（Φ5，轴端距腕心 73.4 mm），没有法兰；站点 `shaft_tip`、`fw_end`。
- 限位、执行器上限、反射惯量来自 V2 资料；质量是 CAD 体积 × 估计密度 + 资料质量。全部是候选值。

详见 `docs/hardware/DUMMY_V2.md`。`dummy_reference.xml`（上面的 auk 参考模型）保持不变，
仅供 BC 冒烟闭环复现历史数字。
