# 末端工具转接头V1设计记录

软件会话，L1 CAD/网格验证，无硬件操作。

交付位于 ../../design/tool_adapter_v1/。按用户要求平底连接、26×26参考四角M3热熔孔、两个可拆椭圆木柄抱箍、侧置D435双孔支架。旧CAD孔距不是改良件的实测证明，随附试孔件。

验证：4个有效单实体，装配零相交；木柄/相机简化包络无穿插；底孔有封底、相机孔贯通；全部4个STL无非流形边和退化三角形、正体积、底面Z=0。已查看两张装配/爆炸渲染。报告见 design/tool_adapter_v1/cad_checks.json 和 mesh_checks.json。

未验证：打印工艺、真实盖板/螺钉配合、夹持力、相机视野、承载/疲劳、全臂碰撞与线缆。未修改任何训练、仿真、控制或标定配置。

CAD环境独立在 D:\VLA\_tools\cadquery-env；生成过程中两次matplotlib着色参数错误已修复，最终生成与网格检查均成功。复现命令：python build_model.py，然后python check_stl.py。
