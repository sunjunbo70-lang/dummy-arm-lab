@echo off
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
if not exist ".venv-loop\Scripts\python.exe" (echo Missing .venv-loop & pause & exit /b 1)
set "REPLAY=experiments\2026-09-23_wall_cycle_v06_p2a\raw\native_replay_v3\rollout.npz"
if not exist "%REPLAY%" (echo Replay not found: %REPLAY% & pause & exit /b 1)
echo Native MuJoCo v0.6 replay. L1 simulation only.
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --record "%REPLAY%"
if errorlevel 1 pause
