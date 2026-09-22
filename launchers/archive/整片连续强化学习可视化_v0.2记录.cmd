@echo off
setlocal
rem v0.2 record only (stick-figure web replay, see docs/changes/2026-09-22_wall_cycle_v0.3.md C11).
rem Works when double-clicked inside dummy-arm-lab, or from a folder next to it.
set "ROOT="
if exist "%~dp0experiments\2026-09-22_wall_cycle_rl\" set "ROOT=%~dp0"
if not defined ROOT if exist "%~dp0..\dummy-arm-lab\experiments\2026-09-22_wall_cycle_rl\" set "ROOT=%~dp0..\dummy-arm-lab\"
if not defined ROOT if exist "D:\project\VLA\dummy-arm-lab\experiments\2026-09-22_wall_cycle_rl\" set "ROOT=D:\project\VLA\dummy-arm-lab\"
if not defined ROOT (
  echo Cannot find the dummy-arm-lab repository. Put this file in the repository folder.
  pause
  exit /b 1
)
set "PLAYER=%ROOT%experiments\2026-09-22_wall_cycle_rl\raw\interactive_replay_v6\index.html"
if not exist "%PLAYER%" (
  echo Replay not found: %PLAYER%
  pause
  exit /b 1
)
start "" "%PLAYER%"


