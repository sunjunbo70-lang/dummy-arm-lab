@echo off
setlocal
for %%I in ("%~dp0..\..\..\..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"
if not exist "%REPO_ROOT%\.venv-loop\Scripts\python.exe" (
  echo Simulation Python environment was not found. Run tools\environment\windows_setup.cmd first.
  pause
  exit /b 1
)
echo Wall-trowel SIMULATION ONLY. No hardware connection, no serial port.
echo J6 reducer + printed holder + pointed trowel (rigid). Random errors + wall probing + load-cell force loop.
echo Close the viewer window to exit.
"%REPO_ROOT%\.venv-loop\Scripts\python.exe" -m dummy_loop.wall demo --viewer --random-scale 1 --probe --servo
if errorlevel 1 pause


