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
echo Plastering session, SIMULATION ONLY. No hardware connection.
echo One stroke per band, the whole reachable area, then a thickness map.
echo.
if exist "%~dp0outputs\wall\rl\policy.npz" (
  echo Using the trained policy: outputs\wall\rl\policy.npz
  %PY% -m dummy_loop.wall rl-session --policy outputs\wall\rl\policy.npz --out outputs\wall\session
) else (
  echo No trained policy found; running the hand-written script instead.
  echo Run "抹涂手法强化学习.cmd" first to train one.
  %PY% -m dummy_loop.wall rl-session --out outputs\wall\session
)
if errorlevel 1 pause
echo.
echo Result written to outputs\wall\session\session.json
pause


