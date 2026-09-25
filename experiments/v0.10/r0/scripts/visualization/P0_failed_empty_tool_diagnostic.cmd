@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\..\..\.."
set "PYTHONPATH=%CD%"
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle_v09.replay "experiments/v0.10/r0/runs/p0_native_001/replay/scene00_L06/trajectory.npz"
pause
