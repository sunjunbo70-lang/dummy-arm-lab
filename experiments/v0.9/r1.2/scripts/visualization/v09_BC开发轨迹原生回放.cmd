@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\..\..\.."
set "PYTHONPATH=%CD%"
echo 这是BC开发案例，不是最终RL结果。
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle_v09.replay "experiments\v0.9\r1.2\runs\v09_r12\BC005_native_edge_002\60082\trajectory.npz"
pause
