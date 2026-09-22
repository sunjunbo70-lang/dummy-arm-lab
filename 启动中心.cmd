@echo off
setlocal
cd /d "%~dp0"
echo Dummy Arm Lab launcher categories:
echo   1. Simulation
echo   2. Training
echo   3. Replay
echo   4. Hardware tools
echo   5. Archived launchers
choice /c 12345 /n /m "Select [1-5]: "
if errorlevel 5 start "" "%~dp0launchers\archive"
if errorlevel 4 if not errorlevel 5 start "" "%~dp0launchers\hardware"
if errorlevel 3 if not errorlevel 4 start "" "%~dp0launchers\replay"
if errorlevel 2 if not errorlevel 3 start "" "%~dp0launchers\training"
if errorlevel 1 if not errorlevel 2 start "" "%~dp0launchers\simulation"
