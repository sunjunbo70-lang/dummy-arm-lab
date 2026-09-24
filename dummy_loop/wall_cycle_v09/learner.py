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
 def distributions(self,obs,valid_ops):
  if not valid_ops.any(-1).all():raise ValueError('At least one operation must be valid')
  h=self.actor(obs);logits=self.op(h).masked_fill(~valid_ops,-torch.inf)
  return Categorical(logits=logits),self.mu(h).reshape(-1,4,19)
 def evaluate(self,obs,valid_ops,op,raw):
  cat,mus=self.distributions(obs,valid_ops);ix=torch.arange(len(obs),device=obs.device)
  normal=Normal(mus[ix,op],self.log_std[op].clamp(-5,1).exp());mask=self.parameter_mask[op]
  # Stable log(1-tanh(x)^2), including samples close to +/-1.
  jac=2*(torch.log(torch.tensor(2.,device=raw.device))-raw-torch.nn.functional.softplus(-2*raw))
  logp=cat.log_prob(op)+((normal.log_prob(raw)-jac)*mask).sum(-1)
  entropy=cat.entropy()+(normal.entropy()*mask).sum(-1)
  return logp,entropy,self.critic(obs).squeeze(-1)
 @torch.no_grad()
 def sample(self,obs,valid_ops,deterministic=False):
  cat,mus=self.distributions(obs,valid_ops);op=cat.logits.argmax(-1) if deterministic else cat.sample()
  ix=torch.arange(len(obs),device=obs.device);mu=mus[ix,op]
  raw=mu if deterministic else Normal(mu,self.log_std[op].clamp(-5,1).exp()).sample()
  logp,_,v=self.evaluate(obs,valid_ops,op,raw)
  return op,torch.tanh(raw)*self.parameter_mask[op],raw,logp,v

def ppo_update(model,opt,batch,epochs=4,minibatch=128,clip=.1):
 obs,valid,op,raw,oldlog,adv,returns=batch
 adv=(adv-adv.mean())/(adv.std()+1e-8);records=[]
 for epoch in range(epochs):
  for ix in torch.randperm(len(obs),device=obs.device).split(minibatch):
   lp,entropy,value=model.evaluate(obs[ix],valid[ix],op[ix],raw[ix]);ratio=(lp-oldlog[ix]).exp()
   actor=-torch.minimum(ratio*adv[ix],ratio.clamp(1-clip,1+clip)*adv[ix]).mean()
   critic=.5*(value-returns[ix]).square().mean();loss=actor+critic-.001*entropy.mean()
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
   opt.zero_grad();loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
   records.append({'actor_loss':actor.item(),'value_loss':critic.item(),'entropy':entropy.mean().item(),'gradient_norm':grad.item(),'approx_kl':(oldlog[ix]-lp).mean().item(),'clip_fraction':((ratio-1).abs()>clip).float().mean().item()})
 return records
