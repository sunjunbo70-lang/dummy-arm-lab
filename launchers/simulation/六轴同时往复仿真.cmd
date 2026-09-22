@echo off
setlocal
for %%I in ("%~dp0..\..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"
if not exist "%REPO_ROOT%\.venv-loop\Scripts\python.exe" (
  echo Simulation Python environment was not found.
  pause
  exit /b 1
)
echo Starting six-axis SIMULATION ONLY (Dummy V2 model). No hardware connection.
echo All axes: +/-10 degree targets, 8-second cycle.
echo SPACE: pause/resume. R: restart. C: covers. Close viewer to exit.
"%REPO_ROOT%\.venv-loop\Scripts\python.exe" "%REPO_ROOT%\tools\view_simulation.py" --simultaneous
if errorlevel 1 pause


