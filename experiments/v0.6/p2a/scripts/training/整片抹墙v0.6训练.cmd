@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
if not exist ".venv-loop\Scripts\python.exe" (echo Missing .venv-loop & pause & exit /b 1)
set "OUT=experiments\v0.6\p2a\runs\v06_p2a_manual"
if exist "%OUT%" (echo Output already exists: %OUT% & pause & exit /b 1)
echo v0.6 P2-A software-only training. No serial port, no robot motion.
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.train --physics v0.6 --recipe v0.7 --updates 50 --steps-per-update 2048 --teacher-episodes 200 --dagger-rounds 4 --dagger-episodes 40 --test-episodes 60 --val-episodes 60 --cosim-test-episodes 20 --out "%OUT%"
if errorlevel 1 pause
