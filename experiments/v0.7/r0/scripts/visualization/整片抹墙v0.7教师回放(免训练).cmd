@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
if not exist ".venv-loop\Scripts\python.exe" (echo Missing .venv-loop & pause & exit /b 1)
set "OUT=experiments\v0.7\r0\runs\v07_teacher_preview"
if exist "%OUT%\rollout.npz" goto play
echo Recording one teacher episode with full MuJoCo co-simulation, this takes about 5-10 minutes ...
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --teacher technique --v06 --seed 3 --no-window --out "%OUT%"
if errorlevel 1 (pause & exit /b 1)
:play
echo Keys: [ slower, ] faster (0.25x-10x), space pause, R restart.
echo Playing %OUT%\rollout.npz (native MuJoCo, L1 simulation only).
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --record "%OUT%\rollout.npz" --speed 5
if errorlevel 1 pause
