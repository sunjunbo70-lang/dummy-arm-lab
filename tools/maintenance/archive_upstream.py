"""历史取证脚本。

2026-09-18 工作区重构后，上游资料位于与基线并列的 _archive/。
本脚本不是正常开发或实机运行的依赖，产物已归档在 docs/history/。
重跑前确认下方路径指向实际的资料位置。
"""
from pathlib import Path
import os,json,zipfile,hashlib
R=Path.cwd(); out=R/'archive/upstream'; out.mkdir(parents=True,exist_ok=True)
if (out/'original_sources.zip').exists():
 raise SystemExit('Refusing to overwrite the existing upstream evidence archive')
rows=[]; excluded=[]
with zipfile.ZipFile(out/'original_sources.zip','w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
 for label,base in [('VLA',Path('D:/VLA/_archive')),('kit-gpu-launcher',Path('E:/project/kit-gpu-launcher'))]:
  for here,ds,fs in os.walk(base):
   kept=[]
   for d in ds:
    if d in ('.git','__pycache__','node_modules') or d.startswith(('.venv','venv')):
     excluded.append({'path':str(Path(here)/d),'reason':'git metadata or generated environment/cache'})
    else: kept.append(d)
   ds[:]=kept
   for n in fs:
    p=Path(here)/n; rel=p.relative_to(base); reason=None
    if p.suffix.lower() in ('.vdi','.vmdk','.vhd','.vhdx') or n=='SwitchPi_dummy.zip' or n.startswith('VirtualBox-'): reason='legacy virtual-machine disk/archive/installer; retained at original location'
    if label=='kit-gpu-launcher' and (n in ('config.json','preferences.json','known_hosts') or p.suffix.lower()=='.lnk'): reason='machine/account configuration; recreate at destination'
    if reason: excluded.append({'path':str(p),'bytes':p.stat().st_size,'reason':reason}); continue
    arc=label+'/'+rel.as_posix(); h=hashlib.sha256()
    with p.open('rb') as f,z.open(arc,'w') as dst:
     while chunk:=f.read(1024*1024): h.update(chunk); dst.write(chunk)
    rows.append({'source':str(p),'archive_path':arc,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(out/'manifest.json').write_text(json.dumps({'included':rows,'excluded':excluded},ensure_ascii=False,indent=2),encoding='utf-8')
print('Archived',len(rows),'files;',len(excluded),'exclusions;',round((out/'original_sources.zip').stat().st_size/1024**2,1),'MB')
