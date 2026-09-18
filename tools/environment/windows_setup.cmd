@echo off
REM Double-click entry point for windows_setup.ps1.
REM Software session only: no serial port, no --enable, no motion command.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows_setup.ps1" %*
echo.
pause
