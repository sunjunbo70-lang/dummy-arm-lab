# 源码结构和数据契约

## 核心包

| 文件 | 职责 |
|---|---|
| dummy_loop/core.py | Observation、角度/形状检查、Guard和硬件profile |
| dummy_loop/serial_backend.py | 串口协议、原始收发日志、调试命令 |
| dummy_loop/live_control.py | 单后台线程、原生角度反馈、目标限速、停止与预设状态机 |
| dummy_loop/sim_backend.py | MuJoCo参考模型和统一仿真接口 |
| dummy_loop/policy.py | 合成教师与线性行为克隆；非VLA |
| dummy_loop/__main__.py | 采集、训练、仿真运行和只读影子模式CLI |
| tools/gui/live_mujoco.py | Tk双视图界面与显示刷新 |
| vendor/native_client/fibre/ | 原生USB传输、接口发现；有本项目超时修复 |

## 坐标和单位

核心Robot API及参考策略使用弧度；实机上位机目标/反馈为固件角度（度）。Studio视觉模型显示采用 `qpos = deg2rad(firmware_deg - [0,0,90,0,0,0])`。原生joint.angle需加偏置 `[0,-75,180,0,0,0]`；初始化与一次串口读数交叉核对。
不要把MuJoCo qpos直接当成实机角度。参考模型和Studio模型不是同一套已标定动力学。

## 控制流

UI目标 → LiveController → 角度/目标速度/跟随误差检查 → ASCII串口目标 → 控制板。
原生USB缓存角度 → LiveController → UI反馈姿态和日志。
GUI的mj_forward仅计算显示姿态，未进行训练用的接触动力学验证。

关键参数：主机目标速度1～20°/s；最大目标领先2°；跟随误差超过3°停止；反馈过期会请求取消动作。固件缓存反馈的接收时间不等于传感器采样时间。STOP应答不等于独立实体停止测量。

## 模型

models/dummy_reference.xml 来自auk参考模型，支持早期软件闭环；models/dummy_studio_visual.xml用于上位机和六轴可视化。Studio关节层级通过实际Transform及桥接代码核对，相关来源在 models/studio_source/；质量、惯量、力矩、接触及机械零点仍不能视为实机标定值。

## 工具分类与旧路径

分类实现在tools各子目录；旧tools同名入口转发以保留启动脚本和历史命令。完整映射见tool_migration.json。仅支持从根目录运行，或先将项目根目录加入PYTHONPATH；本工程不宣称可独立pip安装核心包后自动携带外部模型。
