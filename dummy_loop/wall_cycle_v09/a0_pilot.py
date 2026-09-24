"""A0 backend pilot only. Not the A1 v0.9 main experiment."""
import argparse,json,os,time,subprocess,pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .clean_env import CleanEnv,config
from ..wall_cycle.reach_table import ReachTableExecutor

class A0Policy(nn.Module):
 def __init__(self,n,a):
  super().__init__();self.pi=nn.Sequential(nn.Linear(n,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,a));self.v=nn.Sequential(nn.Linear(n,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1));self.std=nn.Parameter(torch.full((a,),-.8));self.register_buffer('mean',torch.zeros(n));self.register_buffer('scale',torch.ones(n))
 def evaluate(self,o,z):
  x=((o-self.mean)/self.scale).clamp(-10,10);dist=Normal(self.pi(x),self.std.clamp(-5,1).exp());jac=2*(np.log(2)-z-torch.nn.functional.softplus(-2*z));return (dist.log_prob(z)-jac).sum(-1),dist.entropy().mean(-1),self.v(x).squeeze(-1)
 @torch.no_grad()
 def act(self,o):
  x=((o-self.mean)/self.scale).clamp(-10,10);z=Normal(self.pi(x),self.std.clamp(-5,1).exp()).sample();lp,_,v=self.evaluate(o,z);return z.tanh(),z,lp,v

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--updates',type=int,default=20);p.add_argument('--teacher-episodes',type=int,default=12);args=p.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir();(out/'rollouts').mkdir();torch.set_num_threads(2);torch.manual_seed(9);np.random.seed(9)
 subprocess.Popen([r'C:\Users\28017\anaconda3\python.exe','-m','dummy_loop.wall_cycle_v09.telemetry','--out',str(out/'resources.jsonl'),'--pid',str(os.getpid())],creationflags=0x08000000)
 manifest={'stage':'A0 CUDA backend pilot; NOT A1/full experiment','updates':args.updates,'rollout_steps':2048,'teacher_episodes':args.teacher_episodes,'seed':9,'device':torch.cuda.get_device_name(),'physics':'legacy nominal table, not TimedArm cosim','status':'running'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 c=config();ex=ReachTableExecutor(c);env=CleanEnv(c,60010,executor=ex);o=env.reset();model=A0Policy(len(o),len(env.teacher_action())).cuda();opt=torch.optim.Adam(model.parameters(),lr=3e-5)
 X=[];Y=[];teacher=[];t=time.perf_counter()
 for k in range(args.teacher_episodes):
  e=CleanEnv(c,60010+k,executor=ex);s=e.reset();done=False
  while not done:
   a=e.teacher_action();X.append(s);Y.append(a);s,r,done,info=e.step(a)
  teacher.append(info);print('teacher',k,info['metrics']['coverage'],flush=True)
 X=np.asarray(X,dtype=np.float32);Y=np.asarray(Y,dtype=np.float32);np.savez_compressed(out/'teacher.npz',observations=X,actions=Y)
 x=torch.tensor(X,device='cuda');y=torch.tensor(Y,device='cuda');model.mean.copy_(x.mean(0));model.scale.copy_(x.std(0).clamp_min(.05));bcopt=torch.optim.Adam(model.pi.parameters(),lr=5e-4)
 for i in range(500):
  ix=torch.randint(len(x),(min(256,len(x)),),device='cuda');pred=model.pi(((x[ix]-model.mean)/model.scale).clamp(-10,10)).tanh();loss=(pred-y[ix]).square().mean();bcopt.zero_grad();loss.backward();bcopt.step()
 torch.save({'model':model.state_dict(),'bc_optimizer':bcopt.state_dict(),'teacher_records':teacher,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'checkpoints'/'bc.pt')
 (out/'bc_metrics.json').write_text(json.dumps({'samples':len(X),'mse':loss.item(),'elapsed_s':time.perf_counter()-t},indent=2),encoding='utf8')
 env=CleanEnv(c,60100,executor=ex);o=env.reset();episode=0
 for update in range(1,args.updates+1):
  rows=[];sampling=time.perf_counter();policy_s=env_s=0
  for j in range(2048):
   t=time.perf_counter();a,z,lp,v=model.act(torch.as_tensor(o,device='cuda').unsqueeze(0));a=a[0].cpu().numpy();z=z[0].cpu().numpy();lv=float(lp.item());vv=float(v.item());policy_s+=time.perf_counter()-t
   t=time.perf_counter();n,r,done,info=env.step(a);env_s+=time.perf_counter()-t;rows.append((o.copy(),z,lv,vv,r,done));o=n
   if done:
    with (out/'episodes.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'update':update,'episode':episode,**info},default=str)+'\n')
    episode+=1;env=CleanEnv(c,60100+episode,executor=ex);o=env.reset()
  sampling_s=time.perf_counter()-sampling
  with torch.no_grad():_,_,last=model.evaluate(torch.tensor(o,device='cuda').unsqueeze(0),torch.zeros((1,len(Y[0])),device='cuda'))
  obs,z,oldlp,val,rew,done=map(np.asarray,zip(*rows));adv=np.zeros(2048,dtype=np.float32);carry=0.;nextv=last.item()
  for j in reversed(range(2048)):
   delta=rew[j]+.995*nextv*(1-done[j])-val[j];carry=delta+.995*.95*(1-done[j])*carry;adv[j]=carry;nextv=val[j]
  returns=adv+val;np.savez_compressed(out/'rollouts'/f'{update:04d}.npz',obs=obs,raw=z,logp=oldlp,value=val,reward=rew,done=done,advantage=adv,returns=returns)
  tx=torch.tensor(obs,device='cuda',dtype=torch.float32);tz=torch.tensor(z,device='cuda',dtype=torch.float32);tl=torch.tensor(oldlp,device='cuda',dtype=torch.float32);ta=torch.tensor(adv,device='cuda');tr=torch.tensor(returns,device='cuda',dtype=torch.float32);ta=(ta-ta.mean())/(ta.std()+1e-8);t=time.perf_counter();metrics=[]
  for epoch in range(4):
   for ix in torch.randperm(2048,device='cuda').split(128):
    lp,ent,v=model.evaluate(tx[ix],tz[ix]);ratio=(lp-tl[ix]).exp();pl=-torch.minimum(ratio*ta[ix],ratio.clamp(.9,1.1)*ta[ix]).mean();vl=.5*(v-tr[ix]).square().mean();loss=pl+vl-.001*ent.mean()
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    opt.zero_grad();loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step();metrics.append([pl.item(),vl.item(),ent.mean().item(),grad.item(),(tl[ix]-lp).mean().item()])
  torch.cuda.synchronize();update_s=time.perf_counter()-t
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'update':update,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'observation':o,'episode':episode},out/'checkpoints'/f'{update:04d}.pt')
  with (out/'checkpoints'/f'{update:04d}_environment.pkl').open('wb') as f:pickle.dump(env,f)
  with (out/'updates.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'update':update,'steps':update*2048,'sampling_s':sampling_s,'policy_s':policy_s,'environment_s':env_s,'gpu_update_s':update_s,'loss_columns':['policy','value','entropy_gaussian_proxy','gradient_norm','approx_kl'],'minibatches':metrics})+'\n')
  print('update',update,'sampling',sampling_s,'GPU',update_s,flush=True)
 manifest['status']='completed A0 pilot only';(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
if __name__=='__main__':main()
