@echo off
cd /d "%~dp0"
if not exist ".venv-loop\Scripts\python.exe" (
 echo Python environment missing. Run tools\environment\windows_setup.cmd first.
 pause
 exit /b 1
)
".venv-loop\Scripts\python.exe" "tools\simulation\inspect_lab_tool.py"
if errorlevel 1 pause
