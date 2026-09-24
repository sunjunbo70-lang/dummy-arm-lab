"""Conditional hybrid policy: operation plus tanh-transformed active parameters.
No simulator dependency. Raw samples are retained for PPO; no projection in log-prob.
"""
import torch
from torch import nn
from torch.distributions import Categorical,Normal

class HybridPolicy(nn.Module):
 def __init__(self,obs_dim):
  super().__init__()
  self.actor=nn.Sequential(nn.Linear(obs_dim,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh())
  self.op=nn.Linear(256,4);self.mu=nn.Linear(256,4*19);self.log_std=nn.Parameter(torch.full((4,19),-.8))
  self.critic=nn.Sequential(nn.Linear(obs_dim,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))
  mask=torch.zeros(4,19);mask[0,0]=1;mask[1,:]=1
  self.register_buffer('parameter_mask',mask)
  self.register_buffer('obs_mean',torch.zeros(obs_dim));self.register_buffer('obs_scale',torch.ones(obs_dim))
 def normalize(self,obs):return ((obs-self.obs_mean)/self.obs_scale).clamp(-10,10)
 def distributions(self,obs,valid_ops):
  if not valid_ops.any(-1).all():raise ValueError('At least one operation must be valid')
  h=self.actor(self.normalize(obs));logits=self.op(h).masked_fill(~valid_ops,-torch.inf)
  return Categorical(logits=logits),self.mu(h).reshape(-1,4,19)
 def evaluate(self,obs,valid_ops,op,raw):
  cat,mus=self.distributions(obs,valid_ops);ix=torch.arange(len(obs),device=obs.device)
  normal=Normal(mus[ix,op],self.log_std[op].clamp(-5,1).exp());mask=self.parameter_mask[op]
  # Stable log(1-tanh(x)^2), including samples close to +/-1.
  jac=2*(torch.log(torch.tensor(2.,device=raw.device))-raw-torch.nn.functional.softplus(-2*raw))
  logp=cat.log_prob(op)+((normal.log_prob(raw)-jac)*mask).sum(-1)
  all_dist=Normal(mus,self.log_std.clamp(-5,1).exp());z=all_dist.rsample()
  correction=2*(torch.log(torch.tensor(2.,device=z.device))-z-torch.nn.functional.softplus(-2*z))
  per_op=((all_dist.entropy()+correction)*self.parameter_mask).sum(-1)/self.parameter_mask.sum(-1).clamp_min(1)
  entropy=cat.entropy()+(cat.probs*per_op).sum(-1)
  return logp,entropy,self.critic(self.normalize(obs)).squeeze(-1)
 @torch.no_grad()
 def sample(self,obs,valid_ops,deterministic=False):
  cat,mus=self.distributions(obs,valid_ops);op=cat.logits.argmax(-1) if deterministic else cat.sample()
  ix=torch.arange(len(obs),device=obs.device);mu=mus[ix,op]
  raw=mu if deterministic else Normal(mu,self.log_std[op].clamp(-5,1).exp()).sample()
  logp,_,v=self.evaluate(obs,valid_ops,op,raw)
  return op,torch.tanh(raw)*self.parameter_mask[op],raw,logp,v

def ppo_update(model,opt,batch,epochs=4,minibatch=128,clip=.1,demo=None,beta=0.,target_kl=.03):
 obs,valid,op,raw,oldlog,adv,returns=batch
 adv=(adv-adv.mean())/(adv.std()+1e-8);records=[]
 for epoch in range(epochs):
  for ix in torch.randperm(len(obs),device=obs.device).split(minibatch):
   lp,entropy,value=model.evaluate(obs[ix],valid[ix],op[ix],raw[ix]);ratio=(lp-oldlog[ix]).exp()
   stable_kl=((ratio-1)-(lp-oldlog[ix])).mean()
   if stable_kl.item()>target_kl:
    records.append({'early_stop_kl':stable_kl.item(),'epoch':epoch,'minibatches_completed':len(records)});return records
   actor=-torch.minimum(ratio*adv[ix],ratio.clamp(1-clip,1+clip)*adv[ix]).mean()
   critic=.5*(value-returns[ix]).square().mean();aux=torch.zeros((),device=obs.device)
   if demo is not None and beta>0:
    dx,dvalid,dop,da=demo;j=torch.randint(len(dx),(len(ix),),device=obs.device);aux=bc_loss(model,dx[j],dvalid[j],dop[j],da[j])
   loss=actor+critic-.001*entropy.mean()+beta*aux
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
   opt.zero_grad();loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
   records.append({'actor_loss':actor.item(),'value_loss':critic.item(),'entropy':entropy.mean().item(),'gradient_norm':grad.item(),'approx_kl':(oldlog[ix]-lp).mean().item(),'nonnegative_kl':stable_kl.item(),'clip_fraction':((ratio-1).abs()>clip).float().mean().item()})
 return records


class MultiHybridPolicy(HybridPolicy):
 """Eight continuous contact modes; PPO uses the marginal mixture density.
 Modes are initialised by teacher orientation, but every head can learn all
 nineteen continuous parameters. No runtime teacher or action projection.
 """
 def __init__(self,obs_dim,n_ops=4,contact_ops=(1,)):
  super().__init__(obs_dim);self.n_ops=n_ops;self.contact_ops=contact_ops
  self.op=nn.Linear(256,n_ops);self.mu=nn.Linear(256,n_ops*8*19);self.mode=nn.Linear(256,n_ops*8)
  self.log_std=nn.Parameter(torch.full((n_ops,8,19),-2.3))
 def teacher_modes(self,op,target):
  return torch.where(op==1,torch.round(target[:,8]*4).long()%8,0)
 def distributions(self,obs,valid_ops):
  h=self.actor(self.normalize(obs));cat=Categorical(logits=self.op(h).masked_fill(~valid_ops,-torch.inf))
  mus=self.mu(h).reshape(-1,self.n_ops,8,19);logits=self.mode(h).reshape(-1,self.n_ops,8)
  # LOAD/RESCAN/FINISH have one mode, CONTACT has eight.
  active=torch.zeros(self.n_ops,8,dtype=torch.bool,device=obs.device);active[:,0]=True;active[list(self.contact_ops),:]=True
  return cat,mus,Categorical(logits=logits.masked_fill(~active,-torch.inf))
 def conditional_logp(self,mus,mode,op,raw):
  ix=torch.arange(len(raw),device=raw.device);normal=Normal(mus[ix,op],self.log_std[op].clamp(-5,1).exp())
  component=(normal.log_prob(raw[:,None,:])*self.parameter_mask[op,None,:]).sum(-1)
  mix=torch.logsumexp(mode.logits[ix,op]+component,dim=-1)
  jac=2*(torch.log(torch.tensor(2.,device=raw.device))-raw-torch.nn.functional.softplus(-2*raw))
  return mix-(jac*self.parameter_mask[op]).sum(-1)
 def evaluate(self,obs,valid_ops,op,raw):
  cat,mus,mode=self.distributions(obs,valid_ops);lp=cat.log_prob(op)+self.conditional_logp(mus,mode,op,raw)
  # Stratified Monte Carlo: explicitly weight every mode, so gradients also
  # account for changes in mixture weights (not just sampled component paths).
  ix=torch.arange(len(obs),device=obs.device);ent=[]
  for k in range(self.n_ops):
   if not self.parameter_mask[k].any():ent.append(torch.zeros_like(lp));continue
   ops=torch.full_like(op,k);terms=[]
   for component in range(8 if k in self.contact_ops else 1):
    z=Normal(mus[:,k,component],self.log_std[k,component].clamp(-5,1).exp()).rsample()
    terms.append(-self.conditional_logp(mus,mode,ops,z))
   h=(mode.probs[:,k,:len(terms)]*torch.stack(terms,-1)).sum(-1)
   ent.append(h/self.parameter_mask[k].sum().clamp_min(1))
  entropy=cat.entropy()+(cat.probs*torch.stack(ent,-1)).sum(-1)
  return lp,entropy,self.critic(self.normalize(obs)).squeeze(-1)
 @torch.no_grad()
 def sample(self,obs,valid_ops,deterministic=False):
  cat,mus,mode=self.distributions(obs,valid_ops);op=cat.logits.argmax(-1) if deterministic else cat.sample();ix=torch.arange(len(obs),device=obs.device)
  comp=mode.logits[ix,op].argmax(-1) if deterministic else mode.sample()[ix,op]
  mu=mus[ix,op,comp];raw=mu if deterministic else Normal(mu,self.log_std[op,comp].clamp(-5,1).exp()).sample()
  lp=cat.log_prob(op)+self.conditional_logp(mus,mode,op,raw)
  return op,raw.tanh()*self.parameter_mask[op],raw,lp,self.critic(self.normalize(obs)).squeeze(-1)

def bc_loss(model,obs,valid,op,target):
 result=model.distributions(obs,valid);cat,mus=result[:2];ix=torch.arange(len(obs),device=obs.device)
 if isinstance(model,MultiHybridPolicy):
  modes=model.teacher_modes(op,target)
  pred=mus[ix,op,modes].tanh();mode_loss=-result[2].logits[ix,op,modes].mean()
 else:pred=mus[ix,op].tanh();mode_loss=0.
 mask=model.parameter_mask[op];weights=torch.tensor([5,5,5,5,1,1,1,1,2,2,2,2,2,1,1,1,1,1,1],device=obs.device)
 errors=(pred-target).square()
 return -cat.log_prob(op).mean()+mode_loss+((errors*mask*weights).sum(-1)/mask.sum(-1).clamp_min(1)).mean()
