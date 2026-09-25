# 新电脑安装与验证

本机主工作区为 `D:\VLA`；本机运行命令前先 `cd D:\VLA`。迁移到其他电脑时以解压后的工程根目录为准。

## Windows

安装64位 Python 3.12（包含 Tk）。在项目根目录：

```powershell
py -3.12 -m venv .venv-loop
.\.venv-loop\Scripts\python.exe -m pip install -r requirements/gui.txt
.\.venv-loop\Scripts\python.exe tools/doctor.py
.\.venv-loop\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-loop\Scripts\python.exe tools/live_mujoco.py --smoke-test
.\.venv-loop\Scripts\python.exe tools/live_mujoco.py --offline
```

离线双视图确认后再参考 HARDWARE.md 连接。Windows原生USB接口可能需要WINUSB接口注册，不能在新电脑自动套用旧注册表实例路径/备份。

## Ubuntu / 其他 Linux

Ubuntu 是拟验证的目标环境，不是已经完成实机验收的平台。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/gui.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/live_mujoco.py --offline
```

Linux主机需可用Tk和图形驱动。没有桌面的服务器先仅安装core.txt运行单元测试和不渲染的参考实验；虚拟相机仍需渲染环境。不要复制Windows venv。
USB串口名称、udev/用户组权限、libusb访问、原生接口发现需要单独验证。现有原生客户端和设备身份检查仍绑定本实验机械臂，换另一台机械臂不能直接认为等价。

## 参考策略流程

先运行 `python -m dummy_loop --help` 及各子命令 `--help` 查参数。现有 CLI 包括 collect-sim、train、run-sim、shadow、ports 等。归档policy.npz只适用于参考仿真。新增数据必须输出到新的实验目录，不覆盖baseline。

## 复现参考学习闭环（仅仿真）

激活上述环境后，在根目录执行；把 `my-first-run` 换成本次独立实验名：

```text
python -m dummy_loop collect-sim --episodes 40 --seed 7 --output experiments/my-first-run/teacher.npz
python -m dummy_loop train --dataset experiments/my-first-run/teacher.npz --output experiments/my-first-run/policy.npz
python -m dummy_loop run-sim --policy experiments/my-first-run/policy.npz --steps 240 --log experiments/my-first-run/rollout.jsonl
```

教师数据为合成数据；这条命令链不调用实机，不是视觉抓取训练。打开图形显示可在最后一条添加 `--viewer`。实机控制入口与此参考策略尚未合并为已验收的自动策略执行器。

## 验证范围

本次交接执行的软件测试和迁移路径测试记录在 experiments/00_initial_debug/records/2026-09-18_handoff/。未发送实机指令；未连接Ubuntu或服务器；安装命令是目标环境操作说明，不冒充已实测。
