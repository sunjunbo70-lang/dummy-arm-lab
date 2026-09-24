"""Persistent, fail-visible v0.9 campaign. No hardware access and no silent gate bypass.
Training stages, validation selection, final tests and native records are separate.
"""
import argparse,json,time,sys,os,subprocess,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/wall_cycle/v09_r12'
PY=sys.executable
REACH=BASE/'reach_a1_001/reach.npz'
DEMO=BASE/'teacher_400_003_inventory'
DAGGER=BASE/'dagger40_round2_001'
BC=BASE/'BC1_seed11_dagger2_005/checkpoints/bc.pt'

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def atomic(path,data):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(path)
def history(run):
 run=Path(run);rows={};manifest=run/'config/manifest.json'
 if manifest.exists():
  resume=read(manifest)['args'].get('resume')
  if resume:rows.update(history(Path(resume).parents[1]))
 for f in (run/'metrics').glob('validation_*.json'):
  result=read(f);u=int(f.stem.split('_')[-1]);rows[u]={'update':u,'mean_J':result['mean_J'],'coverage':result['mean_coverage'],'checkpoint':str(run/'checkpoints'/f'{u:04d}.pt')}
 return rows

def command(group,seed,stage,out,source=None,bc=BC):
 mode='Scratch' if group=='Scratch' else ('RL2' if group=='RL2' else 'RL1')
 argv=[PY,'-m','dummy_loop.wall_cycle_v09.train','--reach',str(REACH),'--demo',str(BASE/'teacher_A0_400_002' if group=='RL0' else DEMO),'--out',str(out),'--workers','23','--updates',str(stage),'--seed',str(seed),'--mode',mode,'--lr','0.000003']
 if group=='RL0':
  argv+=['--action-space','A0']
  pointer=BASE/'BC0_selected.json'
  if pointer.exists():
   choice=read(pointer);argv+=['--dagger',choice['dagger']]
   if not source:argv+=['--bc-from',choice['checkpoint'],'--bc-iterations','0']
 else:argv+=['--dagger',str(DAGGER)]
 if source:argv+=['--resume',str(source)]
 elif group=='RL0' and not (BASE/'BC0_selected.json').exists() and (BASE/'BC0_seed11_001/metrics/bc_gate.json').exists() and read(BASE/'BC0_seed11_001/metrics/bc_gate.json')['pass']:
  argv+=['--bc-from',str(BASE/'BC0_seed11_001/checkpoints/bc.pt'),'--bc-iterations','0']
 elif group!='Scratch' and group!='RL0':argv+=['--bc-from',str(bc),'--bc-iterations','0']
 if group=='straight':argv+=['--straight']
 if group=='no_edge':argv+=['--no-edge']
 if group=='old_stop':argv+=['--old-stop']
 return argv

def wait_training(jobs,out):
 handles={};logs={}
 while True:
  for j in jobs:
   if j['status']!='running':continue
   mf=Path(j['out'])/'config/manifest.json'
   status=read(mf)['status'] if mf.exists() else 'starting'
   proc=handles.get(j['id'])
   if status=='training_stage_completed':j['status']='complete'
   elif status=='blocked_at_bc_gate':j['status']='blocked_at_bc_gate'
   elif proc is not None and proc.poll() is not None:j['status']='failed';j['returncode']=proc.returncode
   if j['status']!='running' and j['id'] in logs:logs.pop(j['id']).close()
  slots=2-sum(j['status']=='running' for j in jobs)
  for j in jobs:
   if slots<=0:break
   if j['status']!='pending':continue
   if j['group']=='RL0' and j['stage']==100:j['argv']=command('RL0',j['seed'],100,Path(j['out']))
   log=(out/'logs'/f'{j["id"]}.log').open('w',encoding='utf8');logs[j['id']]=log
   proc=subprocess.Popen(j['argv'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000);handles[j['id']]=proc;j.update(status='running',pid=proc.pid,started_utc=time.time());slots-=1
  atomic(out/'jobs.json',jobs)
  if not any(j['status'] in ('pending','running') for j in jobs):return
  time.sleep(5)

def external(group,seed,path):return {'id':f'{group}_{seed}_100','group':group,'seed':seed,'stage':100,'out':str(path),'status':'running','external':True}
def job(group,seed,stage,out,source=None):
 dest=out/'training'/f'{group}_seed{seed}_{stage:04d}'
 return {'id':f'{group}_{seed}_{stage}','group':group,'seed':seed,'stage':stage,'out':str(dest),'status':'pending','argv':command(group,seed,stage,dest,source)}
def run_checked(argv,log):
 with Path(log).open('w',encoding='utf8') as f:
  result=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
 if result.returncode:raise RuntimeError(f'Command failed ({result.returncode}); see {log}')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
 for name in ['logs','training','evaluation','records']: (out/name).mkdir()
 from .provenance import capture
 capture(out/'config')
 manifest={'status':'training','created_utc':time.time(),'scope':'L1 simulation only','reach':str(REACH),'reach_sha256':hashlib.sha256(REACH.read_bytes()).hexdigest(),'bc':str(BC),'bc_gate':read(BC.parents[1]/'metrics/bc_gate.json'),'core_seeds':[11,22,33],'core_groups':['RL1','RL2'],'updates':[100,200,400],'rollout_decisions':2048,'lr':3e-6,'KL_guard':.03,'max_parallel_training':2,'expansion_rule':'At each boundary, compare best validation J in earlier/later halves of the new 100 or 200 updates, per run. Expand all core groups plus RL0/Scratch if median relative improvement of either core family exceeds 2%. Stop expansion if both scheduled cosim checks in the stage show requested contacts with less than 30% execution or downward material-face frames. Do not use final tests.','selection_rule':'Minimum mean J on fixed six lightweight validation scenes across all completed stages, ties earlier update. Full 60 validation scenes reported after selection; no tuning on final tests.','baseline_budget':'RL0 and Scratch follow core budget if expansion occurs; three ablations remain at 100 and compare to matched RL1_100 reference. BC/teachers are evaluation-only.','final_tests':'60 paired nominal scenes per group; 20 paired full MuJoCo scenes for T,T+,BC1 and all six core policies; expand core full tests to 60 only when first20 meet stage coverage70%/edge65%/RMSE1mm.','absolute_success':'coverage>=95%, edge>=95%, RMSE<=0.5mm, P95<=1mm; failure is retained, thresholds unchanged','test_seed_start':80000,'validation_seed_start':70000,'completed_is_not_task_success':True}
 atomic(out/'manifest.json',manifest)
 previous=BASE/'campaign_001'
 # Wait for the original scheduler to finish its remaining jobs; never duplicate them.
 while read(previous/'manifest.json')['status']=='training':time.sleep(10)
 prior=read(previous/'jobs.json')
 if any(j['status']!='complete' for j in prior if j['group']!='straight'):
  raise RuntimeError('Another original group failed; inspect before continuing')
 dag=out/'straight_dagger40'
 run_checked([PY,'-m','dummy_loop.wall_cycle_v09.dagger_straight','--checkpoint',str(BASE/'campaign_002_straight_recovery/training/straight_seed11_0100/checkpoints/bc.pt'),'--reach',str(REACH),'--out',str(dag),'--seed','60820','--episodes','40'],out/'logs/straight_dagger40.log')
 # Preserve both 40-episode DAgger rounds as distinct files; no extra episodes.
 import shutil
 combined=out/'straight_dagger80';combined.mkdir()
 for label,folder in [('round1',BASE/'campaign_002_straight_recovery/straight_dagger40'),('round2',dag)]:
  for f in sorted(folder.glob('*.npz')):shutil.copy2(f,combined/f'{label}_{f.name}')
 if len(list(combined.glob('*.npz')))!=80:raise RuntimeError('Expected exactly 80 DAgger episodes')
 jobs=[dict(j,external=True) for j in prior if j['group']!='straight']
 recovery=job('straight',11,100,out)
 argv=recovery['argv'];argv[argv.index('--bc-iterations')+1]='6000';argv[argv.index('--dagger')+1]=str(combined);argv+=['--dagger-repeat','5']
 jobs.append(recovery)
 manifest.update(parent_campaign=str(previous),recovery_note='Original straight transfer BC gate failed at 72.3%. Two straight-specific 40-episode DAgger rounds (80 total), sampled 5x relative to teacher data, plus 6000 BC updates; other completed jobs adopted without rerun. Adapted initialization is a confound, not a clean single-factor comparison.')
 atomic(out/'manifest.json',manifest)
 wait_training(jobs,out)
 if any(j['status']!='complete' for j in jobs):
  manifest.update(status='needs_attention',reason='At least one planned group failed or did not pass BC; no success claim.');atomic(out/'manifest.json',manifest);return
 hundred=list(jobs)
 for target in [200,400]:
  decisions={};blocked=[]
  current={ (j['group'],j['seed']):j for j in jobs if j['group'] in ['RL1','RL2','RL0','Scratch']}
  for g in ['RL1','RL2']:
   vals=[]
   for s in [11,22,33]:
    j=current[g,s];h=history(j['out']);new=[h[u] for u in sorted(h) if u>j['stage']//2];old=[h[u] for u in sorted(h) if (u<=j['stage']//2 and u>=(1 if j['stage']==100 else j['stage']//4))]
    if new and old:vals.append((min(x['mean_J'] for x in old)-min(x['mean_J'] for x in new))/max(abs(min(x['mean_J'] for x in old)),1e-8))
    phys=[read(f)['episodes'] for f in sorted((Path(j['out'])/'metrics').glob('cosim_development*.json'))[-2:]]
    bad=lambda rr:any(x['face_down_frames']>0 for x in rr) or (sum(x['contact_requests'] for x in rr)>0 and sum(x['executed_contacts'] for x in rr)/sum(x['contact_requests'] for x in rr)<.3)
    if len(phys)==2 and all(bad(rr) for rr in phys):blocked.append(j['id'])
   decisions[g]=float(np.median(vals)) if vals else 0.
  decision={'target':target,'median_relative_improvement':decisions,'physical_blocked':blocked,'expand':not blocked and max(decisions.values())>.02};atomic(out/f'expansion_{target}.json',decision)
  if not decision['expand']:break
  next_jobs=[job(g,s,target,out,Path(j['out'])/'checkpoints'/f'{j["stage"]:04d}.pt') for (g,s),j in current.items()];wait_training(next_jobs,out);jobs+=next_jobs;atomic(out/'all_training_jobs.json',jobs)
  if any(j['status']!='complete' for j in next_jobs):manifest['status']='needs_attention';atomic(out/'manifest.json',manifest);return
 selected={}
 for j in jobs:
  key=f'{j["group"]}_seed{j["seed"]}';h=history(j['out'])
  if h:
   best=min(h.values(),key=lambda x:(x['mean_J'],x['update']))
   if key not in selected or best['mean_J']<selected[key]['mean_J']:selected[key]={**best,'group':j['group'],'seed':j['seed']}
 ref=next(j for j in hundred if j['group']=='RL1' and j['seed']==11);selected['RL1_100_reference']={**min(history(ref['out']).values(),key=lambda x:(x['mean_J'],x['update'])),'group':'reference','seed':11}
 selected['BC1']={'checkpoint':str(BC),'group':'BC1','seed':11}
 a0=next(j for j in hundred if j['group']=='RL0');selected['BC0']={'checkpoint':str(Path(a0['out'])/'checkpoints/bc.pt'),'group':'BC0','seed':11}
 atomic(out/'selected.json',selected);manifest['status']='validation';atomic(out/'manifest.json',manifest)
 for key,item in selected.items():
  dest=out/'evaluation'/f'validation_{key}';run_checked([PY,'-m','dummy_loop.wall_cycle_v09.batch_evaluate','--checkpoint',item['checkpoint'],'--reach',str(REACH),'--out',str(dest),'--seed','70000','--episodes','60','--workers','8'],out/'logs'/f'validation_{key}.log')
 freeze={'selected':selected,'selection_time_utc':time.time(),'code_files':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'dummy_loop/wall_cycle_v09').glob('*.py')},'checkpoints':{k:hashlib.sha256(Path(v['checkpoint']).read_bytes()).hexdigest() for k,v in selected.items()}};atomic(out/'frozen_before_test.json',freeze)
 manifest['status']='independent_tests';atomic(out/'manifest.json',manifest)
 all_groups={**selected,**{g:{'teacher':g,'group':g} for g in ['T','T+','T_budget','FINISH']}}
 for key,item in all_groups.items():
  policy=['--checkpoint',item['checkpoint']] if 'checkpoint' in item else ['--teacher',item['teacher']]
  for backend,n in [('nominal',60)]+([('cosim',20)] if item['group'] in ['RL1','RL2','BC1','T','T+'] else []):
   dest=out/'evaluation'/f'test_{backend}_{key}';run_checked([PY,'-m','dummy_loop.wall_cycle_v09.batch_evaluate',*policy,'--reach',str(REACH),'--out',str(dest),'--seed','80000','--episodes',str(n),'--backend',backend,'--workers','8' if backend=='nominal' else '4'],out/'logs'/f'test_{backend}_{key}.log')
   q=read(dest/'summary.json')['by_task']['all']
   if backend=='cosim' and item['group'] in ['RL1','RL2'] and q['coverage']>=.70 and q['edge_coverage']>=.65 and q['rmse_mm']<=1:
    extra=out/'evaluation'/f'test_cosim_extra40_{key}';run_checked([PY,'-m','dummy_loop.wall_cycle_v09.batch_evaluate',*policy,'--reach',str(REACH),'--out',str(extra),'--seed','80020','--episodes','40','--backend','cosim','--workers','4'],out/'logs'/f'test_cosim_extra40_{key}.log')
 best_key=min([k for k,v in selected.items() if v['group'] in ['RL1','RL2']],key=lambda k:selected[k]['mean_J'])
 for key,policy in [('T',['--teacher','T']),('BC1',['--checkpoint',str(BC)]),(best_key,['--checkpoint',selected[best_key]['checkpoint']])]:
  run_checked([PY,'-m','dummy_loop.wall_cycle_v09.evaluate_run',*policy,'--reach',str(REACH),'--out',str(out/'records'/key),'--seed','60090','--task','edge_ridge','--record'],out/'logs'/f'record_{key}.log')
 # Best/worst full test scenes of the validation-selected core policy.
 test_path=out/'evaluation'/f'test_cosim_{best_key}'/'episodes.jsonl'
 cases=[json.loads(x) for x in test_path.read_text().splitlines()]
 for label,case in [('largest_improvement',min(cases,key=lambda x:x['final']['J']-x['initial_J'])),('worst_final',max(cases,key=lambda x:x['final']['J']))]:
  run_checked([PY,'-m','dummy_loop.wall_cycle_v09.evaluate_run','--checkpoint',selected[best_key]['checkpoint'],'--reach',str(REACH),'--out',str(out/'records'/label),'--seed',str(case['seed']),'--record'],out/'logs'/f'record_{label}.log')
 manifest.update(status='completed',completed_utc=time.time(),quality_success='See independent test per-episode metrics; completion does not imply quality thresholds passed.');atomic(out/'manifest.json',manifest)
 run_checked([PY,'-m','dummy_loop.wall_cycle_v09.report_campaign',str(out)],out/'logs/report.log')
 print('Campaign completed',out,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  if '--out' in sys.argv:
   target=Path(sys.argv[sys.argv.index('--out')+1]);mf=target/'manifest.json'
   if mf.exists():
    record=read(mf);record.update(status='needs_attention',error=repr(error));atomic(mf,record)
  raise
