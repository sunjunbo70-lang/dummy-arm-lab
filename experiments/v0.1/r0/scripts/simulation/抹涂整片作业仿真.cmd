@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "REPO_ROOT=%%~fI"
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
if exist "%CD%\experiments\v0.1\r0\runs\rl\policy.npz" (
  echo Using the trained policy: experiments\v0.1\r0\runs\rl\policy.npz
  %PY% -m dummy_loop.wall rl-session --policy experiments\v0.1\r0\runs\rl\policy.npz --out experiments\v0.1\r0\runs\session
) else (
  echo No trained policy found; running the hand-written script instead.
  echo Run "抹涂手法强化学习.cmd" first to train one.
  %PY% -m dummy_loop.wall rl-session --out experiments\v0.1\r0\runs\session
)
if errorlevel 1 pause
echo.
echo Result written to experiments\v0.1\r0\runs\session\session.json
pause


