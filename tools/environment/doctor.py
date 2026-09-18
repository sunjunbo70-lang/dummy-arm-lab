"""Read-only environment inventory; no credentials, no motion commands."""
import importlib.metadata
import json
import platform
import shutil
import subprocess

report={'os':platform.platform(),'python':platform.python_version()}
for name in ['numpy','mujoco','pyserial','torch','pyrealsense2']:
    try: report[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: report[name]=None
if shutil.which('nvidia-smi'):
    result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    report['gpu']=result.stdout.strip() if result.returncode==0 else 'query failed'
print(json.dumps(report,indent=2,ensure_ascii=False))
