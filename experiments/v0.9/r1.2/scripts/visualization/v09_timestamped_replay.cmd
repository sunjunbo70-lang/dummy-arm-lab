@echo off
cd /d "%~dp0..\..\..\..\.."
set PYTHONPATH=%CD%
if "%~1"=="" (
 echo Usage: drag trajectory_*.npz onto this launcher.
 pause
 exit /b 1
)
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle_v09.replay "%~1"
