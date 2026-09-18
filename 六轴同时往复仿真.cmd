@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv-loop\Scripts\python.exe" (
  echo Simulation Python environment was not found.
  pause
  exit /b 1
)
echo Starting six-axis SIMULATION ONLY. No hardware connection.
echo All axes: +/-10 degree targets, 8-second cycle.
echo SPACE: pause/resume. R: restart. C: covers. Close viewer to exit.
"%~dp0.venv-loop\Scripts\python.exe" "%~dp0tools\view_simulation.py" --studio --simultaneous
if errorlevel 1 pause
