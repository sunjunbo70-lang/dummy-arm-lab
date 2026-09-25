@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\..\..\.."
set "PYTHONPATH=%CD%"
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle_v09.replay "experiments/v0.9/r1.2/runs/v09_r12/campaign_005_a0_log_recovery/records/largest_improvement/80002/trajectory.npz"
pause
