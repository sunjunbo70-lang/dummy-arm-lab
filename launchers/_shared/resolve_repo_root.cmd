@echo off
setlocal
for %%I in ("%~dp0..\..") do set "REPO_ROOT=%%~fI"
if not exist "%REPO_ROOT%\dummy_loop" (
  echo Cannot locate repository root from %~dp0
  exit /b 1
)
cd /d "%REPO_ROOT%"
exit /b 0

