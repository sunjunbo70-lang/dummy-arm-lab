@echo off
setlocal
rem Wall-cycle v0.5 P1 native MuJoCo replay. Simulation only: no serial port, no robot motion.
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
if not exist ".venv-loop\Scripts\python.exe" (
  echo Missing .venv-loop. Run tools\environment\windows_setup.cmd first.
  pause
  exit /b 1
)
set "POLICY=experiments\2026-09-22_wall_cycle_v05\raw\p1_training\policy.npz"
set "REPLAY=experiments\2026-09-22_wall_cycle_v05\raw\native_replay_seed_20005_bare\rollout.npz"
if exist "%REPLAY%" (
  echo Native MuJoCo L1 replay. Seed 20005, bare initial wall.
  echo This is a failed research rollout: coverage 35.14%%, waste 37.52%%, success=false.
  ".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --record "%REPLAY%"
) else if exist "%POLICY%" (
  echo Rebuilding deterministic seed 20005 replay, then opening MuJoCo...
  ".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --v05 --mixed-initial --policy "%POLICY%" --seed 20005 --out "outputs\wall_cycle\v05_p1_seed20005"
) else (
  echo Missing trained policy: %POLICY%
  pause
  exit /b 1
)
if errorlevel 1 pause


