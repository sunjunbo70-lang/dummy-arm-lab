"""Trainable shared CNN and operation/continuous heads for the r1 contract.

Raw Gaussian actions and explicit disjoint length-bin samples are retained for PPO.
No observation normalization uses privileged wall/material truth.
"""
import math
import torch
from torch import nn
from torch.distributions import Categorical,Normal
from .actions import PARAM_MASK,Op

class Policy(nn.Module):
    def __init__(self,scalar_dim=32,group='R2'):
        super().__init__()
        if group not in ('R0','R1','R2'): raise ValueError(group)
        self.group=group;self.scalar_dim=scalar_dim
        self.map_encoder=nn.Sequential(nn.Conv2d(6,32,3,2,1),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),nn.Conv2d(64,64,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d((5,5)),nn.Flatten(),nn.Linear(1600,256),nn.SiLU())
        self.scalar_encoder=nn.Sequential(nn.Linear(scalar_dim,64),nn.SiLU())
        self.fusion=nn.Sequential(nn.Linear(320,256),nn.SiLU())
        self.operation=nn.Linear(256,7);self.mean=nn.Linear(256,7*19)
        self.length_bin=nn.Linear(256,7*3);self.log_std=nn.Parameter(torch.full((7,19),-.8))
        self.value=nn.Linear(256,1)
        mask=torch.as_tensor(PARAM_MASK.copy(),dtype=torch.float32)
        if group=='R0':mask[Op.CONTACT_NEW,3]=0
        self.register_buffer('parameter_mask',mask)
        self.register_buffer('length_low',torch.tensor([.03,.08,.16]))
        self.register_buffer('length_width',torch.tensor([.05,.08,.10]))
        nn.init.zeros_(self.mean.weight);nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.length_bin.weight);nn.init.zeros_(self.length_bin.bias)

    def distributions(self,maps,scalars,valid):
        if maps.ndim!=4 or maps.shape[1:]!=(6,100,100):raise ValueError('Expected [B,6,100,100]')
        if scalars.shape!=(len(maps),self.scalar_dim) or valid.shape!=(len(maps),7):raise ValueError('Invalid scalar/mask shape')
        if not valid.any(-1).all():raise ValueError('All operations masked')
        if self.group!='R2' and valid[:,Op.CONTINUE].any():raise ValueError('CONTINUE requires R2')
        h=self.fusion(torch.cat([self.map_encoder(maps),self.scalar_encoder(scalars)],-1))
        return Categorical(logits=self.operation(h).masked_fill(~valid,-torch.inf)),self.mean(h).reshape(-1,7,19),self.length_bin(h).reshape(-1,7,3),self.value(h).squeeze(-1)

    def _contact(self,op):return ((op==Op.CONTACT_NEW)|(op==Op.CONTINUE)) if self.group!='R0' else torch.zeros_like(op,dtype=torch.bool)

    def evaluate(self,maps,scalars,valid,op,raw,length_bin,entropy=False):
        cat,means,length_logits,value=self.distributions(maps,scalars,valid);ix=torch.arange(len(op),device=op.device)
        normal=Normal(means[ix,op],self.log_std[op].clamp(-5,1).exp());mask=self.parameter_mask[op]
        correction=2*(math.log(2)-raw-torch.nn.functional.softplus(-2*raw))
        lp=cat.log_prob(op)+((normal.log_prob(raw)-correction)*mask).sum(-1)
        contact=self._contact(op);bins=Categorical(logits=length_logits[ix,op])
        # Mapping tanh(raw length) into a disjoint subinterval of bounded parameter[3].
        lp=lp+contact*(bins.log_prob(length_bin)-torch.log(self.length_width[length_bin]/.23))
        ent_param=torch.zeros_like(lp)
        if entropy:
            # Entropy on selected conditional head. Operation entropy is exact;
            # parameter entropy is a sample estimate, not an enumeration of all heads.
            z=normal.rsample();jac=2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
            ent_param=((normal.entropy()+jac)*mask).sum(-1)
            ent_param+=contact*(bins.entropy()+(bins.probs*torch.log(self.length_width/.23)).sum(-1))
            ent_param=ent_param/mask.sum(-1).clamp_min(1)
        return lp,value,cat.entropy(),ent_param

    @torch.no_grad()
    def sample(self,maps,scalars,valid,deterministic=False):
        cat,means,logits,value=self.distributions(maps,scalars,valid);op=cat.logits.argmax(-1) if deterministic else cat.sample();ix=torch.arange(len(op),device=op.device)
        normal=Normal(means[ix,op],self.log_std[op].clamp(-5,1).exp());raw=normal.mean if deterministic else normal.sample()
        bins=Categorical(logits=logits[ix,op]);component=bins.logits.argmax(-1) if deterministic else bins.sample()
        action=raw.tanh();length=self.length_low[component]+(action[:,3]+1)*.5*self.length_width[component]
        action[:,3]=torch.where(self._contact(op),(length-.03)/.115-1,action[:,3])
        action*=self.parameter_mask[op]
        lp,_,_,_=self.evaluate(maps,scalars,valid,op,raw,component)
        return dict(op=op,action=action,raw=raw,length_bin=component,log_prob=lp,value=value)
