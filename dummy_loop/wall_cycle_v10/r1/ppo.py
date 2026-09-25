"""PPO optimization primitive for r1. No environment or rollout collection here.
Old masks and raw actions are mandatory; do not re-sample actions during update.
"""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class PPOConfig:
    clip: float=.2
    value_weight: float=.5
    operation_entropy: float=.01
    parameter_entropy: float=.001
    max_grad_norm: float=.5


def objective(log_prob,old_log_prob,value,returns,advantage,op_entropy,param_entropy,cfg=PPOConfig()):
    tensors=(log_prob,old_log_prob,value,returns,advantage,op_entropy,param_entropy)
    if any(t.shape!=log_prob.shape for t in tensors) or log_prob.ndim!=1:
        raise ValueError('Expected same one-dimensional minibatch shape')
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError('Nonfinite PPO inputs')
    log_ratio=log_prob-old_log_prob.detach()
    ratio=log_ratio.exp()
    if not torch.isfinite(ratio).all():raise ValueError('Nonfinite PPO ratio')
    adv=advantage.detach()
    policy_loss=-torch.minimum(ratio*adv,ratio.clamp(1-cfg.clip,1+cfg.clip)*adv).mean()
    value_loss=.5*(value-returns.detach()).square().mean()
    loss=policy_loss+cfg.value_weight*value_loss-cfg.operation_entropy*op_entropy.mean()-cfg.parameter_entropy*param_entropy.mean()
    diagnostics=dict(policy_loss=policy_loss.detach(),value_loss=value_loss.detach(),approx_kl=((ratio-1)-log_ratio).mean().detach(),clip_fraction=((ratio-1).abs()>cfg.clip).float().mean().detach(),operation_entropy=op_entropy.mean().detach(),parameter_entropy=param_entropy.mean().detach())
    return loss,diagnostics


def update_minibatch(policy,optimizer,batch,cfg=PPOConfig()):
    """Batch advantages must be normalized once across the rollout by caller.
    Caller's checkpoint must include optimizer/RNG/environment state for resume.
    KL stopping, epochs and minibatch scheduling are intentionally caller-owned.
    """
    lp,value,oe,pe=policy.evaluate(batch['maps'],batch['scalars'],batch['valid'],batch['op'],batch['raw'],batch['length_bin'],entropy=True)
    loss,metrics=objective(lp,batch['old_log_prob'],value,batch['returns'],batch['advantage'],oe,pe,cfg)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),cfg.max_grad_norm,error_if_nonfinite=True)
    optimizer.step()
    metrics['loss']=loss.detach();metrics['grad_norm']=norm.detach()
    return metrics
