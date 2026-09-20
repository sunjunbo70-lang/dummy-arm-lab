@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv-loop\Scripts\python.exe" (
  echo Python environment .venv-loop was not found. Run tools\environment\windows_setup.cmd first.
  pause
  exit /b 1
)
echo MuJoCo live-sync GUI. Starts READ-ONLY; following is only enabled from the UI by the operator.
echo Shows the Dummy V2 model. The window and 3D views scale with the window; Ctrl+wheel changes font size.
"%~dp0.venv-loop\Scripts\python.exe" "%~dp0tools\live_mujoco.py"
if errorlevel 1 pause
