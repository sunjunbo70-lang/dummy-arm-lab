@echo off
setlocal
for %%I in ("%~dp0..\..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"
if not exist "%REPO_ROOT%\.venv-loop\Scripts\python.exe" (
  echo Simulation Python environment was not found. Run tools\environment\windows_setup.cmd first.
  pause
  exit /b 1
)
set PY="%REPO_ROOT%\.venv-loop\Scripts\python.exe"
echo Trowel-stroke reinforcement learning, SIMULATION ONLY. No hardware connection.
echo Scripted demo -^> behaviour cloning -^> PPO. Two runs, about 6 minutes on CPU.
echo   run 1: warm start from the worker technique (tilt first, then flatten)
echo   run 2: warm start from a flat blade (no technique) -- the control group
echo.
%PY% -m dummy_loop.wall rl-train --updates 100 --script-style technique --out outputs\wall\rl
if errorlevel 1 pause
%PY% -m dummy_loop.wall rl-train --updates 100 --script-style flat --out outputs\wall\rl_flat
if errorlevel 1 pause
%PY% tools\simulation\rl_compare.py --technique outputs\wall\rl --flat outputs\wall\rl_flat --out outputs\wall\rl_compare
if errorlevel 1 pause
echo.
echo Plots written to outputs\wall\rl_compare (training.png, technique.png, session.png)
pause


