@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv-loop\Scripts\python.exe" (
  echo Simulation Python environment was not found. Run tools\environment\windows_setup.cmd first.
  pause
  exit /b 1
)
echo Wall-trowel SIMULATION ONLY. No hardware connection, no serial port.
echo Random pre-calibration errors + wall probing + compression servo.
echo Close the viewer window to exit.
"%~dp0.venv-loop\Scripts\python.exe" -m dummy_loop.wall demo --viewer --random-scale 1 --probe --servo
if errorlevel 1 pause
