@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"
if not exist ".venv-loop\Scripts\python.exe" (
  echo Missing Python environment. Run tools\environment\windows_setup.cmd first.
  pause
  exit /b 1
)
".venv-loop\Scripts\python.exe" tools\simulation\open_rl_visualization.py %*
if errorlevel 1 pause


