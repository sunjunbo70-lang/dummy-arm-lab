@echo off
setlocal
for %%I in ("%~dp0..\..\..\..\..") do set "ROOT=%%~fI\"
cd /d "%ROOT%"
if not exist ".venv-loop\Scripts\python.exe" (echo Missing .venv-loop & pause & exit /b 1)
rem v0.8 one-click: parallel training (all logical cores but one), then record a replay.
rem Software-only: no serial port, no robot motion. Reach table is unchanged since v0.7.
set "PYTHONUNBUFFERED=1"
set "OUT=experiments\v0.8\r0\runs\v08_manual"
set "LOG=%OUT%_run_log.txt"
if exist "%OUT%" (echo Output already exists: %OUT% - rename or delete it to re-run & pause & exit /b 1)
if not exist "outputs\wall_cycle" mkdir "outputs\wall_cycle"
echo ==== v0.8 run started %date% %time% ====> "%LOG%"
echo [1/3] Self-check (v0.8 unit tests, about 1 minute) ...
echo [1/3] Self-check ...>> "%LOG%"
".venv-loop\Scripts\python.exe" -c "import sys, unittest; r = unittest.main(module='tests.test_wall_cycle_v08', exit=False, argv=['selfcheck']); sys.exit(0 if r.result.wasSuccessful() else 1)" >> "%LOG%" 2>&1
if errorlevel 1 (echo [FAILED] self-check, see %LOG% & pause & exit /b 1)
echo [2/3] Training on %NUMBER_OF_PROCESSORS% logical cores, expected 1-3 hours. Progress: %LOG%
echo [2/3] Training ...>> "%LOG%"
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.train --recipe v0.8 --workers 0 --updates 50 --steps-per-update 2048 --teacher-episodes 200 --dagger-rounds 4 --dagger-episodes 40 --test-episodes 60 --val-episodes 60 --cosim-test-episodes 20 --out "%OUT%" >> "%LOG%" 2>&1
if errorlevel 1 (echo [FAILED] train, see %LOG% & pause & exit /b 1)
echo [3/3] Recording replay ...
echo [3/3] Recording replay ...>> "%LOG%"
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle.view --policy "%OUT%\policy.npz" --v08 --seed 3 --no-window --out "%OUT%\replay" >> "%LOG%" 2>&1
if errorlevel 1 (echo [FAILED] replay, see %LOG% & pause & exit /b 1)
echo ==== all done %date% %time% ====>> "%LOG%"
echo All done. See %OUT%\training.json and %LOG%
pause
