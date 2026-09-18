@echo off
cd /d "%~dp0"
"%~dp0.venv-loop\Scripts\python.exe" "%~dp0tools\live_mujoco.py"
if errorlevel 1 pause
