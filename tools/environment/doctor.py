"""Read-only environment inventory; no credentials, no motion commands.

记录训练相关的硬件事实：GPU 架构决定能否用 bf16 / FlashAttention 2 / TF32。
跨架构训练结果不可能逐位复现，所以实验记录里必须写明架构与实际使用的 dtype。
"""
import importlib.metadata
import json
import platform
import shutil
import subprocess

report={'os':platform.platform(),'python':platform.python_version()}
for name in ['numpy','mujoco','pyserial','jsonschema','matplotlib','torch','pyrealsense2']:
    try: report[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: report[name]=None
if shutil.which('nvidia-smi'):
    result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version,compute_cap','--format=csv,noheader'],
                          capture_output=True,text=True,timeout=15)
    if result.returncode!=0:   # 旧驱动不认识 compute_cap
        result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
                              capture_output=True,text=True,timeout=15)
    report['gpu']=[l.strip() for l in result.stdout.strip().splitlines()] if result.returncode==0 else 'query failed'
if report['torch']:
    try:
        import torch
        cuda={'available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda}
        if cuda['available']:
            devs=[]
            for i in range(torch.cuda.device_count()):
                major,minor=torch.cuda.get_device_capability(i)
                devs.append({'name':torch.cuda.get_device_name(i),'compute_capability':f'{major}.{minor}',
                             'bf16_native':major>=8,          # Ampere 起；Turing (7.5) 没有
                             'tf32':major>=8,
                             'flash_attention_2_supported_arch':major>=8})
            cuda['devices']=devs
        report['torch_cuda']=cuda
    except Exception as exc:   # 只做清点，不因为 torch 出错而失败
        report['torch_cuda']={'error':repr(exc)}
print(json.dumps(report,indent=2,ensure_ascii=False))
