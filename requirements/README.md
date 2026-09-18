# 依赖分层

core.txt 是参考仿真/线性策略环境；gui.txt 加入 GUI 和原生 USB；modeling.txt 加入 Unity 资源提取。
Windows 已用 Python 3.12.14。完整已安装版本见 windows-observed-freeze.txt（取证快照，不是 Linux 通用锁文件）。Tk 为 Python/系统组件，不由 pip 安装。Linux 可能需要管理员提供 python3-tk、OpenGL/EGL 和设备权限。
相机需要额外的 pyrealsense2/librealsense；当前相机未实测，不在可复现基础依赖中锁定未经验证版本。
UnityPy 及其解码依赖只用于模型取证，正常 GUI 不需要。某版本在目标平台无 wheel 时应记录安装失败并建立单独环境，不能默默升级全部依赖后称为原版本复现。
