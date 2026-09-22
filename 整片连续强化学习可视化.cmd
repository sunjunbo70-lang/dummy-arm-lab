@echo off
setlocal
rem v0.3: real Dummy V2 model in MuJoCo's own window (drag to rotate, wheel to zoom).
rem Keys: space pause, left/right step, [ ] speed, R restart. Simulation only, no hardware.
rem Works when double-clicked inside dummy-arm-lab, or from a folder next to it.
set "ROOT="
if exist "%~dp0dummy_loop\" set "ROOT=%~dp0"
if not defined ROOT if exist "%~dp0..\dummy-arm-lab\dummy_loop\" set "ROOT=%~dp0..\dummy-arm-lab\"
if not defined ROOT if exist "D:\project\VLA\dummy-arm-lab\dummy_loop\" set "ROOT=D:\project\VLA\dummy-arm-lab\"
if not defined ROOT (
  echo Cannot find the dummy-arm-lab repository. Put this file in the repository folder.
  pause
  exit /b 1
)
cd /d "%ROOT%"
echo Repository: %CD%
if not exist ".venv-loop\Scripts\python.exe" (
  echo Missing Python environment. Run tools\environment\windows_setup.cmd first.
  pause
  exit /b 1
)
if not exist "dummy_loop\wall_cycle\view.py" (
  echo The v0.3 update is not in this repository yet.
  echo Run D:\project\VLA\wall_cycle_v03_update_2026-09-22.cmd first, then try again.
  pause
  exit /b 1
)
set "POLICY=experiments\2026-09-22_wall_cycle_v03\raw\technique_teacher\policy.npz"
set "REPLAY=experiments\2026-09-22_wall_cycle_v03\raw\replay_seed20000\rollout.npz"
set "CACHE=outputs\wall_cycle\replay_v03\rollout.npz"
if exist "%CACHE%" (
  ".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --record "%CACHE%"
) else if exist "%REPLAY%" (
  ".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --record "%REPLAY%"
) else if exist "%POLICY%" (
  echo Running the trained policy once in the MuJoCo co-simulation, about one minute...
  ".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --policy "%POLICY%" --seed 20000 --out outputs\wall_cycle\replay_v03
) else (
  echo No trained policy found; showing the hand-written teacher instead.
  ".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --teacher technique --seed 20000 --out outputs\wall_cycle\replay_teacher
)
if errorlevel 1 pause
