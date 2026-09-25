@echo off
chcp 65001 >nul
cd /d "%~dp0..\..\..\..\.."
set "PYTHONPATH=%CD%"
if "%~1"=="" (
 echo 请将需要检查的模型.pt文件拖到此脚本上。
 pause
 exit /b 1
)
echo 先重新运行MuJoCo闭环并保存记录，完成后打开原生回放窗口。
".venv-loop\Scripts\python.exe" -m dummy_loop.wall_cycle_v09.checkpoint_view "%~1"
pause
