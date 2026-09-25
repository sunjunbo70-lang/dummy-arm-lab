from pathlib import Path
import json, shutil, hashlib
R=Path(__file__).resolve().parents[2]
def put(path,text):
 p=R/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text,encoding='utf-8')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
# Preserve all pre-organization source files before changing paths.
backup=R/'archive/layout_before_2026-09-18'
if backup.exists(): raise SystemExit('Layout migration already started; do not rerun')
backup.mkdir(parents=True)
for folder in ('tools','dummy_loop','tests','configs'):
 shutil.copytree(R/folder,backup/folder,ignore=shutil.ignore_patterns('__pycache__'))
for p in R.iterdir():
 if p.is_file(): shutil.copy2(p,backup/p.name)
if (R/'整理成果').exists():
 (R/'docs').mkdir(exist_ok=True); shutil.move(str(R/'整理成果'),str(R/'docs/history'))
if (R/'交付包').exists(): shutil.move(str(R/'交付包'),str(R/'archive/legacy_deliveries'))
shutil.copytree(R/'.diagnostics/native_client',R/'vendor/native_client',ignore=shutil.ignore_patterns('__pycache__'))
# Move implementation scripts, retain old filenames as compatibility entrypoints.
groups={
 'gui':['live_mujoco.py'],
 'simulation':['view_simulation.py','render_reference.py'],
 'modeling':['build_studio_model.py','import_reference.py'],
 'hardware':['camera_probe.py','check_live_presets_once.py','commission_fold_once.py','commission_j1_once.py','commission_native_j1.py','native_robot_diagnostic.py','probe_serial_raw.py','probe_studio_transport.py','repair_native_interface.ps1'],
 'diagnostics':['inspect_studio_il.ps1','inspect_studio_native.ps1','inspect_studio_scene.py','build_studio_capture.ps1','read_studio_capture.py'],
 'environment':['doctor.py','ubuntu_setup.sh'],
}
rows=[]
for group,names in groups.items():
 for name in names:
  old=R/'tools'/name; new=R/'tools'/group/name
  s=old.read_text(encoding='utf-8-sig')
  if name.endswith('.py'):
   s=s.replace('.parents[1]','.parents[2]').replace("'.diagnostics/native_client'","'vendor/native_client'")
   put(new.relative_to(R),s)
   put(old.relative_to(R),f'"""Compatibility entrypoint; implementation: tools/{group}/{name}."""\nfrom pathlib import Path\nimport runpy\nimport sys\n_ROOT=Path(__file__).resolve().parents[1]\nsys.path.insert(0,str(_ROOT))\nglobals().update(runpy.run_path(str(_ROOT/"tools/{group}/{name}"),run_name=__name__))\n')
  elif name.endswith('.ps1'):
   s=s.replace('Split-Path $PSScriptRoot -Parent','Split-Path (Split-Path $PSScriptRoot -Parent) -Parent')
   put(new.relative_to(R),s)
   put(old.relative_to(R),f'# Compatibility entrypoint. See tools/{group}/{name}.\n& (Join-Path $PSScriptRoot "{group}/{name}") @args\n')
  else:
   s=s.replace('"$0")/..','"$0")/../..').replace('requirements-loop.txt','requirements/gui.txt').replace('README_实验闭环.md','README.md')
   put(new.relative_to(R),s)
   put(old.relative_to(R),'#!/usr/bin/env bash\nset -euo pipefail\nexec bash "$(dirname -- "$0")/environment/ubuntu_setup.sh" "$@"\n')
  rows.append({'old':str(old.relative_to(R)).replace('\\','/'),'implementation':str(new.relative_to(R)).replace('\\','/'),'category':group})
p=R/'dummy_loop/live_control.py'; p.write_text(p.read_text(encoding="utf-8").replace("'.diagnostics/native_client'","'vendor/native_client'"),encoding='utf-8')
p=R/'inspect_sources.py'; p.write_text(p.read_text(encoding='utf-8').replace("/ '整理成果'","/ 'docs/history'"),encoding='utf-8')
put('docs/tool_migration.json',json.dumps(rows,ensure_ascii=False,indent=2))
# Raw results are immutable evidence copies; outputs remains compatible working scratch.
for p in (R/'outputs').iterdir():
 if not p.is_file(): continue
 n=p.name
 if n.startswith(('teacher','policy','rollout','unseen_rollout')): category='reference_learning'
 elif n.endswith(('.png','.gif')): category='visual_validation'
 elif n.startswith(('studio_',)): category='studio_protocol'
 elif n.startswith(('native_interface_',)): category='machine_specific_usb'
 elif n.startswith(('preset_recheck','correct_fold','capture_correct_fold','fold_once','fold_serial','restart_check')): category='verified_poses'
 else: category='commissioning_and_failures'
 dest=R/'experiments/00_initial_debug/records/2026-09-16_baseline/raw'/category/n
 dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
manifest=[]
for p in sorted((R/'experiments/00_initial_debug/records/2026-09-16_baseline/raw').rglob('*')):
 if p.is_file(): manifest.append({'path':p.relative_to(R).as_posix(),'original_path':'outputs/'+p.name,'bytes':p.stat().st_size,'sha256':digest(p)})
put('experiments/00_initial_debug/records/2026-09-16_baseline/manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
print('Organized tools:',len(rows),'Evidence files:',len(manifest))
