"""Instrumented hybrid BC/PPO runner. Each checkpoint includes all worker states."""
import argparse,json,os,time,pickle,subprocess,hashlib
from pathlib import Path
import numpy as np
from .workers import Vec

def main():
 import torch
 from .learner import HybridPolicy,MultiHybridPolicy,ppo_update,bc_loss
 from .a0_control import A0Policy
 from .evaluation import evaluate
 p=argparse.ArgumentParser();p.add_argument('--reach',required=True);p.add_argument('--demo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--updates',type=int,default=100);p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=11);p.add_argument('--mode',choices=['RL1','RL2','Scratch'],default='RL1');p.add_argument('--resume',type=Path);p.add_argument('--bc-from',type=Path);p.add_argument('--dagger',type=Path);p.add_argument('--straight',action='store_true');p.add_argument('--no-edge',action='store_true');p.add_argument('--old-stop',action='store_true');p.add_argument('--buffer',type=float,default=.03);p.add_argument('--components',type=int,choices=[1,8],default=8);p.add_argument('--bc-iterations',type=int,default=6000);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--action-space',choices=['A0','A1'],default='A1');p.add_argument('--dagger-repeat',type=int,default=1);a=p.parse_args()
 out=a.out;out.mkdir(parents=True,exist_ok=False)
 from .provenance import capture
 capture(out/'config')
 for name in ('checkpoints','logs','metrics','rollouts','config'): (out/name).mkdir(exist_ok=True)
 torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
 subprocess.Popen([r'C:\Users\28017\anaconda3\python.exe','-m','dummy_loop.wall_cycle_v09.telemetry','--out',str(out/'logs/resources.jsonl'),'--pid',str(os.getpid())],creationflags=0x08000000)
 kwargs={'action_space':a.action_space,'buffer':a.buffer,'straight':a.straight,'edge_term':not a.no_edge,'old_stop':a.old_stop};v=Vec(a.workers,2000000+a.seed*10000,a.reach,kwargs);model=(A0Policy if a.action_space=='A0' else (MultiHybridPolicy if a.components==8 else HybridPolicy))(v.obs.shape[1]).cuda();opt=torch.optim.Adam(model.parameters(),lr=a.lr)
 manifest={'args':{k:str(x) if isinstance(x,Path) else x for k,x in vars(a).items()},'schema':f'v09r12.measured-height.v3/{"contact19" if a.action_space=="A1" else "A0quadratic"}.mixture{a.components}','status':'running','obs_dim':v.obs.shape[1],'reach_sha256':hashlib.sha256(Path(a.reach).read_bytes()).hexdigest(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'scope':'A1 unified nominal training with scheduled cosim validation'}
 (out/'config/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 files=sorted(a.demo.glob('*.npz'))+(sorted(a.dagger.glob('*.npz'))*a.dagger_repeat if a.dagger else []);chunks=[np.load(f) for f in files];obs=np.concatenate([z['obs'][z['valid']] for z in chunks]);ops=np.concatenate([z['op'][z['valid']] for z in chunks]);params=np.concatenate([z['params'][z['valid']] for z in chunks]);masks=np.concatenate([z['mask'][z['valid']] for z in chunks]);[z.close() for z in chunks]
 model.obs_mean.copy_(torch.tensor(obs.mean(0),device='cuda'));model.obs_scale.copy_(torch.tensor(obs.std(0).clip(.05),device='cuda'))
 demo=tuple(torch.tensor(x,device='cuda') for x in (obs,masks,ops,params));del obs,ops,params,masks
 def save(label,update):
  state={'action_space':a.action_space,'components':a.components,'model':model.state_dict(),'optimizer':opt.state_dict(),'update':update,'schema':manifest['schema'],'obs_dim':v.obs.shape[1],'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'obs':v.obs,'mask':v.mask,'worker_states':v.command('snapshot'),'kwargs':kwargs,'seed':a.seed}
  torch.save(state,out/'checkpoints'/f'{label}.pt')
 start_update=1
 if a.resume:
  state=torch.load(a.resume,weights_only=False,map_location='cuda');assert state['schema']==manifest['schema'];model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']]);np.random.set_state(state['numpy_rng']);start_update=state['update']+1
  for group in opt.param_groups:group['lr']=a.lr
  if len(v.pipes)!=len(state['worker_states']):raise ValueError('Exact resume requires the same worker count')
  for pipe,b in zip(v.pipes,state['worker_states']):pipe.send(('restore',b))
  rows=[pipe.recv() for pipe in v.pipes];v.obs=np.array([x[0] for x in rows]);v.mask=np.array([x[1] for x in rows])
 elif a.mode!='Scratch':
  if a.bc_from:
   model.load_state_dict(torch.load(a.bc_from,weights_only=False,map_location='cuda')['model'])
   if a.bc_iterations>0:model.obs_mean.copy_(demo[0].mean(0));model.obs_scale.copy_(demo[0].std(0,unbiased=False).clamp_min(.05))
  bc=torch.optim.Adam(list(model.actor.parameters())+list(model.op.parameters())+list(model.mu.parameters())+(list(model.mode.parameters()) if hasattr(model,'mode') else []),lr=3e-4);dx,dm,do,da=demo
  for iteration in range(a.bc_iterations):
   ix=torch.randint(len(dx),(256,),device='cuda');loss=bc_loss(model,dx[ix],dm[ix],do[ix],da[ix]);bc.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);bc.step()
   if iteration%100==0:
    with (out/'metrics/bc.jsonl').open('a') as f:f.write(json.dumps({'iteration':iteration,'loss':loss.item()})+'\n')
  save('bc',0);torch.save(bc.state_dict(),out/'checkpoints/bc_optimizer.pt')
  check=evaluate(model,a.reach,range(60070,60076),kwargs=kwargs);(out/'metrics/bc_development.json').write_text(json.dumps(check,indent=2),encoding='utf8')
 if a.mode!='Scratch' and not a.resume:
  contacts=sum(x['contact_requests'] for x in check['episodes']);executed=sum(x['executed_contacts'] for x in check['episodes']);improves=any(x['best_J']<x['initial_J']-.01 for x in check['episodes'])
  gate={'executable_fraction':executed/max(contacts,1),'contact_requests':contacts,'improves':improves,'pass':contacts>0 and executed/contacts>=.8 and improves}
  (out/'metrics/bc_gate.json').write_text(json.dumps(gate,indent=2),encoding='utf8')
  if not gate['pass']:
   manifest['status']='blocked_at_bc_gate';(out/'config/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');v.close();return
 best=float('inf');steps_per_env=int(np.ceil(2048/a.workers))
 try:
  for update in range(start_update,a.updates+1):
   if update==101:v.command('curriculum',.75)
   records=[];policy_s=env_s=0.;t0=time.perf_counter()
   for k in range(steps_per_env):
    t=time.perf_counter();o=v.obs.copy();mask=v.mask.copy();to=torch.tensor(o,device='cuda');tm=torch.tensor(mask,device='cuda');op,pa,raw,lp,value=model.sample(to,tm);op=op.cpu().numpy();pa=pa.cpu().numpy();raw=raw.cpu().numpy();lp=lp.cpu().numpy();value=value.cpu().numpy();policy_s+=time.perf_counter()-t
    t=time.perf_counter();active=np.arange(a.workers)<min(a.workers,2048-k*a.workers);r,d,infos=v.step(op[active],pa[active]);env_s+=time.perf_counter()-t;records.append((o,mask,op,raw,lp,value,r,d,active))
    with (out/'metrics/decisions.jsonl').open('a',encoding='utf8') as log:
     for i,info in enumerate(infos):log.write(json.dumps({'update':update,'rollout_index':k,'worker':i,'operation':int(op[i]),'parameters':pa[i].tolist(),**info})+'\n')
    with (out/'metrics/episodes.jsonl').open('a',encoding='utf8') as f:
     for i,done in enumerate(d):
      if done:f.write(json.dumps({'update':update,'worker':i,**infos[i]})+'\n')
   with torch.no_grad():last=model.critic(model.normalize(torch.tensor(v.obs,device='cuda'))).squeeze(-1).cpu().numpy()
   ob,ma,op,raw,lp,val,r,d,active=map(np.asarray,zip(*records));adv=np.zeros_like(r,dtype=np.float32);acc=np.zeros(a.workers);nextv=last
   for k in reversed(range(steps_per_env)):
    candidate=r[k]+.995*nextv*(1-d[k])-val[k]+.995*.95*(1-d[k])*acc;acc=np.where(active[k],candidate,acc);adv[k]=np.where(active[k],acc,0);nextv=np.where(active[k],val[k],nextv)
   ret=adv+val;flat=lambda x:x.reshape((-1,)+x.shape[2:])[active.reshape(-1)];batch=tuple(torch.tensor(flat(x),device='cuda',dtype=torch.float32 if x.dtype.kind=='f' else None) for x in (ob,ma,op,raw,lp,adv,ret))
   np.savez_compressed(out/'rollouts'/f'{update:04d}.npz',obs=ob,mask=ma,op=op,raw=raw,logp=lp,value=val,reward=r,done=d,active=active,advantage=adv,returns=ret)
   beta=.1*max(0,1-(update-1)/50) if a.mode=='RL2' else 0.;t=time.perf_counter();metrics=ppo_update(model,opt,batch,demo=demo,beta=beta);torch.cuda.synchronize();gpu_s=time.perf_counter()-t;t=time.perf_counter();save(f'{update:04d}',update);checkpoint_s=time.perf_counter()-t
   row={'update':update,'steps':update*2048,'policy_s':policy_s,'environment_ipc_s':env_s,'gpu_update_s':gpu_s,'checkpoint_s':checkpoint_s,'total_s':time.perf_counter()-t0,'beta':beta,'minibatches':metrics}
   with (out/'metrics/updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(a.mode,a.seed,'update',update,'seconds',round(row['total_s'],2),flush=True)
   if update%10==0:
    result=evaluate(model,a.reach,range(70000,70006),kwargs=kwargs);(out/'metrics'/f'validation_{update:04d}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if result['mean_J']<best:best=result['mean_J'];(out/'config/selected.json').write_text(json.dumps({'update':update,'mean_J':best}),encoding='utf8')
   if update%50==0:
    result=evaluate(model,a.reach,range(60080,60082),'cosim',kwargs=kwargs);(out/'metrics'/f'cosim_development_{update:04d}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
  manifest['status']='training_stage_completed';(out/'config/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 finally:v.close()
if __name__=='__main__':main()
