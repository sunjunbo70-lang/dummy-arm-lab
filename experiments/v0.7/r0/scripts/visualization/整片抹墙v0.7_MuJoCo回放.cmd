@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
if not exist ".venv-loop\Scripts\python.exe" (echo Missing .venv-loop & pause & exit /b 1)
set "OUT=experiments\v0.7\r0\runs\v07_p2_manual"
set "POLICY=%OUT%\policy.npz"
set "REPLAY=%OUT%\replay\rollout.npz"
if exist "%REPLAY%" goto play
if not exist "%POLICY%" (echo Policy not found: %POLICY% & echo Run the v0.7 training launcher first & pause & exit /b 1)
echo No saved rollout yet. Recording one full MuJoCo co-simulation episode, this takes about 5-10 minutes ...
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --policy "%POLICY%" --v06 --seed 3 --no-window --out "%OUT%\replay"
if errorlevel 1 (pause & exit /b 1)
:play
echo Keys: [ slower, ] faster (0.25x-10x), space pause, R restart.
echo Playing %REPLAY% (native MuJoCo, L1 simulation only).
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --record "%REPLAY%" --speed 5
if errorlevel 1 pause
