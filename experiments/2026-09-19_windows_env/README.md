# 新开发机落地：Windows 环境重建与 L1 基线复核

目的：把仓库克隆到新开发机（desktop-m51oshe，`D:\project\VLA\dummy-arm-lab`），
在该机器上用 `requirements/` 钉死的版本重建运行环境，并复核两条既有 L1 结论
是否在本机成立。不引入新结论，只回答「换了机器之后旧结论还成不成立」。

会话类型：软件会话。全程未连接机械臂，未打开串口，未执行任何带 `--enable`
的命令，未发送任何运动指令。`hardware_motion: false`。

输入数据与版本：
- 仓库 `main` @ `285028445081c370dc53779203177e3bcc0522bc`（Initial public release）
- `requirements/dev.txt`（→ gui.txt → core.txt，外加 jsonschema）
- 随机种子 7，40 episode，240 步，与 2026-09-16 冻结基线相同

执行命令和环境：
`tools/environment/windows_setup.ps1`（本次会话新增，经 `windows_setup.cmd` 启动）
依次执行解释器探测 → 建 `.venv-loop` → 装钉死依赖 → `doctor.py` →
`unittest discover -s tests` → `collect-sim` → `train` → `run-sim`。
逐条输出见本目录 `raw/`。

| 项 | 值 |
| --- | --- |
| 操作系统 | Windows-11-10.0.26200-SP0 |
| Python | 3.12.7（`C:\Users\sunbobo\anaconda3\python.exe`） |
| 虚拟环境 | `D:\project\VLA\dummy-arm-lab\.venv-loop` |
| numpy / mujoco / pyserial | 2.5.3 / 3.13.0 / 3.5（与 `core.txt` 钉死版本一致） |
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, driver 591.86 |
| torch / pyrealsense2 | 未安装 |

## 结果与证据

| 检查项 | 结果 | 证据 | 等级 |
| --- | --- | --- | :---: |
| 钉死依赖可在本机安装 | 通过，无版本回退 | `raw/pip_freeze.txt`、`raw/pip_install.log` | L1 |
| 单元测试 | 49 个通过，3 个跳过 | `raw/tests.log` | L1 |
| 参考仿真闭环 | 0.3082207001484489 → 0.0022264043008919554 rad | `raw/rollout.summary.json` | L1 |

跳过的 3 个用例依赖 DummyStudio 提取数据，本仓库不分发，属设计内跳过，
不是失败。

参考仿真的终值与 2026-09-16 冻结基线
（`experiments/2026-09-16_baseline/raw/reference_learning/rollout.summary.json`）
逐位相同，非近似相等。初值两份记录的十进制打印形式不同
（`0.30822070014844888` 与 `0.3082207001484489`），是同一个 IEEE-754 双精度值
被 .NET 与 Python 用不同最短往返规则打印所致，不是数值差异。

## 失败与局限

- **本次只复核了「装得上、测得过、仿真复现得出来」三件事。** `docs/STATUS.md`
  中「Windows 侧 Tk 界面与原生 USB 枚举未在本次重构中复核」这一条**仍然未关闭**：
  本次既没有启动 `tools/live_mujoco.py`，也没有枚举原生 USB 接口。不要把本记录
  当成 GUI 或 USB 通路在本机可用的证据。
- 本记录全部是 L1。它不涉及控制器反馈，更不涉及实机。
- 参考仿真是针对合成教师的线性行为克隆冒烟测试，复现成功只说明数据通路端到端
  可用，不说明任何抓取或 VLA 能力。
- Python 来自 Anaconda 发行版而非 python.org 安装包。本次未观察到差异，但若
  后续出现与 MKL/OpenMP 相关的异常，这是第一个该怀疑的地方。
- `_archive/` 与归档资料只存在于原开发机，未随仓库到本机。引用归档的历史取证
  脚本在本机无法直接运行。

## 下一步

- 单独复核 Tk 上位机（`tools/live_mujoco.py --offline`，零硬件）与原生 USB 枚举，
  才能关闭 STATUS 中那一条。
- M2 只读链路与时效契约的软件部分可以在本机开工。
