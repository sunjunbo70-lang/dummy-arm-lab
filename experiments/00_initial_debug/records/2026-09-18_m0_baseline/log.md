# M0 基线重构执行日志

工作区：`D:\VLA` → 新基线 `D:\VLA\dummy_arm`
证据等级：L1（软件，不连接机械臂，未发送任何运动指令）

## 阶段 0 快照与保险 — 通过

- 生成基线候选清单 223 个文件及 SHA-256：`baseline_files.txt`、`baseline_sha256.txt`
- 根目录结构与体积快照：`root_listing.txt`、`root_sizes.txt`
- 校验 `releases/` 下三个 zip 的 SHA-256，全部 OK：
  - dummy-experiment-history-2026-09-18.zip
  - dummy-experiment-runtime-2026-09-18-workspace.zip
  - dummy-experiment-runtime-2026-09-18.zip
- 未对 `dummy_v2/虚拟机环境/`（33 GB）和 `.venv-loop/` 做哈希，只记录体积

## 阶段 1 建基线（纯复制）— 通过

- 新建 `dummy_arm/`，rsync 复制：dummy_loop、tools、configs、tests、vendor、
  requirements、docs、experiments；models 排除 studio_source/
- 排除 `__pycache__/` 与 `*.pyc`
- 单文件：AGENTS.md、README.md、inspect_sources.py、两个 .cmd
- README_实验闭环.md → docs/history/
- VLA学习计划_一周.md、Dummy_V2_VLA研究メモ.docx → docs/research/
- 写入 `.gitignore`（排除 .venv*/、__pycache__/、outputs/、releases/、experiments/）
- 写入 `.gitattributes`（LFS 跟踪 stl/obj/dae/ply/step/npz/pt/pth/onnx/safetensors/mp4/gif；
  .cmd 与 .ps1 保持 CRLF）
- 基线体积 110 MB
- 原目录未做任何移动或删除

## 结构自检 — 通过，两项待决

- 223/223 文件 SHA-256 与源一致
- MJCF 网格引用：dummy_reference.xml 7/7、dummy_studio_visual.xml 21/21 全部可解析
- 关键文件在位检查全部 OK

待决 1：`docs/career_research_plan/`（9.1 MB）为求职与科研规划材料，非机器人工程内容，
        是否留在代码仓库需用户裁决。
待决 2：`inspect_sources.py` 第 4 行 `ROOT = Path('D:/VLA')` 为硬编码绝对路径，
        M0 完成后布局改变将失效。其余绝对路径出现在文档与历史记录中，属记录性质。

## 未完成

- 阶段 2 Windows 侧验证：本会话的 shell 为 Linux，无法执行 Windows 的 .exe，
  且该环境未安装 mujoco。四条验证命令交由用户在 Windows 执行。
- 阶段 3 git 初始化：待 Windows 侧安装 git-lfs 且阶段 2 验证通过后进行。
- 阶段 4 文档收敛、阶段 5 归档：未开始。

## 工具

- git 2.34.1
- git-lfs 3.8.0（本会话安装至 ~/.local/bin，非用户 Windows 环境）

## 两项待决事项的处理（2026-09-18，用户授权按最合理方案执行）

### 待决 1：career_research_plan

`docs/career_research_plan/`（9.1 MB）移出代码基线，落到 `D:\VLA\career_research_plan\`。

理由：内容为个人求职与科研规划，含学籍、预计毕业时间、投稿状态、目标岗位与
56 张招聘截图的 OCR 结果。它引用本工程的 STATUS/HARDWARE/ROADMAP，但属于
读取方而非工程组成部分。代码仓库未来可能推送到服务器或与他人共享，个人材料
不应随之流动。

未放进 `_archive/`，因为它是活文档（15 周跟踪表、期刊核验模板仍在使用）；
作为根目录下的独立目录保留，不纳入 git。

### 待决 2：inspect_sources.py

从基线根目录移入 `tools/diagnostics/`（按 ARCHITECTURE.md 的分类，该目录收
取证脚本），并重写：

- `ROOT = Path('D:/VLA')` 硬编码改为必填参数 `--root`
- 输出目录由硬编码的 `docs/history/` 改为 `--out`，默认
  `outputs/inspect_sources/`；不再写入 docs/history/，避免覆盖历史证据
- 补上文件头说明：一次性取证工具、产物已归档的位置、重跑需要的前提
- 保留原有 inventory / read / pdf / render 四种模式与路径越界防护
- `py_compile` 通过，`--help` 正常

引用检查：`tools/maintenance/package_release.py` 按整目录收集 `tools/`，
移动后仍会被打包；其根目录文件名单里的 `inspect_sources.py` 失配但不报错。
`tools/maintenance/organize_20260918.py` 是已执行过的一次性迁移脚本，
其中对 inspect_sources.py 的改写行已失效，属历史记录。

## 新发现（待阶段 4/5 处理）

1. 以下取证脚本仍硬编码指向 `D:\VLA\dummy_v2\upstream\...`，阶段 5 归档后
   路径会变，需在归档完成后统一更新为 `_archive/` 下的实际位置：
   - tools/diagnostics/build_studio_capture.ps1
   - tools/diagnostics/inspect_studio_il.ps1
   - tools/diagnostics/inspect_studio_scene.py
   - tools/maintenance/archive_upstream.py
2. `tools/maintenance/package_release.py` 的 history 包收集 `archive` 与
   `.diagnostics` 两个目录，二者在新基线中不存在（归档到 `_archive/`），
   `--kind history` 会打出空包。需在阶段 4 调整其目录清单。
3. 基线体积：101 MB（career_research_plan 移出后）。

## 阶段 2 基线内验证 — 通过（L1）

Windows 侧无法由本会话执行（shell 为 Linux）。改在 Linux 侧安装 mujoco 3.13.0
后执行等效验证：

- 35 个单元测试通过
- 首次运行时 `test_studio_articulation` 报错，暴露出排除 `models/studio_source/`
  会丢失测试依赖 `studio_verified_transforms.json`。已将该目录下的测试与文档
  依赖小文件（6 个，共 324 KB）收入基线，105 MB 的 OBJ 仍归档
- 参考仿真闭环复现：0.3082207001484489 → 0.002226404300892158 rad，
  与归档基线一致（差异仅在末位浮点）

未覆盖：Tk 图形界面、原生 USB 枚举、Windows 上的 venv。这三项需在目标机器复核。

## 阶段 3 git 初始化 — 通过（L1）

- `git init` + `git lfs install --local`，首次提交 d69cdd3，167 个文件
- 46 个文件进 LFS；抽查 `116_dummy_base_ctrl.stl`：仓库内为 3 行指针，
  工作区为 20130184 字节实体文件
- `experiments/` 确认未被跟踪
- **过程中发现连接目录禁止删除导致 git 无法清理 `index.lock` 与临时对象。**
  已申请并获得该目录的删除权限，清理 183 个残留临时对象与 2 个锁文件后
  `git fsck` 与 `git gc` 正常。这是长期问题：git 的 checkout、分支切换与 gc
  都需要删除文件

## 阶段 4 文档收敛 — 通过（L1）

提交 41f3f69。AGENTS.md 重写为工作流约束文件；README 重写；建立 docs/plan/
任务卡；从 dummy_v2 抽取三份硬件文档到 docs/hardware/；新增 docs/DATA_FILES.md；
STATUS.md 证据表加等级列；package_release.py 的历史包来源改为 `_archive/`。

## M1 Profile schema v2 — 通过（L1）

提交 b03312b。schema、模板、load_profile v2 与 14 个新测试。
全量 49 个测试通过，参考闭环误差不变。

关键设计：`CALIBRATED_FIELDS` 每一项都必须有 provenance 记录，
即不可能在不说明数字来源的情况下把档案标为已标定。

## 阶段 5 归档 — 通过（L1）

`D:\VLA` 根目录收敛为三项：`dummy_arm/`（384 MB，含 venv）、
`_archive/`（37 GB）、`career_research_plan/`（9.1 MB）。

全程使用 move，未删除任何原始内容。虚拟机镜像移入 `_archive/_vm/`。
四个取证脚本的硬编码路径更新为指向 `_archive/`。

归档后复检：49 个测试通过、MJCF 网格引用 28/28 可解析、
参考闭环误差不变、package_release 能定位到 `_archive/`。

## 遗留待用户确认

1. 旧 `.venv-loop` 已从原根目录移入基线。venv 通常可直接调用 `Scripts\python.exe`
   而不受目录改名影响，但未在 Windows 上实测。若异常，删除后按 docs/SETUP.md 重建。
2. Windows 侧需安装 git-lfs，否则任何 checkout 或 clone 会把网格取成指针文本。
3. Tk 界面与原生 USB 枚举需在 Windows 上复核。
