# 软件回归验证

在项目根目录运行 `python -m unittest discover -s tests -v`。测试涵盖指令格式、软件约束、自动往复、预设、反馈过期/超时，以及 Studio 与 MuJoCo 可视关节变换的一致性。不会向实体机械臂发送指令。

通过测试不等于力学标定、碰撞验证或实机运动验收。渲染验证另用 `python tools/live_mujoco.py --smoke-test`。
