@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%"
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --record "experiments\v0.8\r0\runs\v08_teacher_preview\rollout.npz" --speed 1
if errorlevel 1 pause
