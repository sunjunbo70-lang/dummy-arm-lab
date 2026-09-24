@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
set "PYTHONPATH=%CD%"
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle_v09.report_campaign "outputs\wall_cycle\v09_r12\campaign_001"
pause
