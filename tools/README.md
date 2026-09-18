# 工具入口

实现位于gui、simulation、modeling、hardware、diagnostics、environment、maintenance子目录。根目录同名脚本为旧命令兼容入口。
日常：tools/live_mujoco.py --offline；tools/view_simulation.py --studio；tools/doctor.py。
--smoke-test只生成预览。硬件脚本不要批量运行，commission_*含真实运动入口，repair_*涉及Windows设备注册。
历史取证脚本含原始D:/VLA路径，应先配置输入。maintenance/organize_20260918.py是一次性迁移记录，检测到备份时拒绝重复执行。
