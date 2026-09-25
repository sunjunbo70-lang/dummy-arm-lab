@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
if not exist ".venv-loop\Scripts\python.exe" (echo Missing .venv-loop & pause & exit /b 1)
rem v0.7 one-click: rebuild reach table, train, record replay. Software-only, no robot motion.
set "PYTHONUNBUFFERED=1"
set "OUT=experiments\v0.7\r0\runs\v07_p2_manual"
set "LOG=%OUT%_run_log.txt"
if exist "%OUT%" (echo Output already exists: %OUT% - rename or delete it to re-run & pause & exit /b 1)
if not exist "outputs\wall_cycle" mkdir "outputs\wall_cycle"
echo ==== v0.7 run started %date% %time% ====> "%LOG%"
echo [1/3] Rebuilding reach table ...
echo [1/3] Rebuilding reach table ...>> "%LOG%"
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.reach_table --step 0.02 >> "%LOG%" 2>&1
if errorlevel 1 (echo [FAILED] reach_table, see %LOG% & pause & exit /b 1)
echo [2/3] Training, several hours ...
echo [2/3] Training ...>> "%LOG%"
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.train --physics v0.6 --recipe v0.7 --updates 50 --steps-per-update 2048 --teacher-episodes 200 --dagger-rounds 4 --dagger-episodes 40 --test-episodes 60 --val-episodes 60 --cosim-test-episodes 20 --out "%OUT%" >> "%LOG%" 2>&1
if errorlevel 1 (echo [FAILED] train, see %LOG% & pause & exit /b 1)
echo [3/3] Recording replay ...
echo [3/3] Recording replay ...>> "%LOG%"
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --policy "%OUT%\policy.npz" --v06 --seed 3 --no-window --out "%OUT%\replay" >> "%LOG%" 2>&1
if errorlevel 1 (echo [FAILED] replay, see %LOG% & pause & exit /b 1)
echo ==== all done %date% %time% ====>> "%LOG%"
echo All done. See %OUT%\training.json and %LOG%
pause
