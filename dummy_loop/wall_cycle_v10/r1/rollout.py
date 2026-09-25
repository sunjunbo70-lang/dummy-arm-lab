"""Device-resident rollout preparation and PPO epochs. Not a simulator/collector.

The collector must supply next_values BEFORE reset and mark FINISH, failure and
finite episode budgets terminal. A collection boundary alone is not terminal.
"""
from dataclasses import dataclass
import math
import torch
from .ppo import PPOConfig, objective

FIELDS = ('maps', 'scalars', 'valid', 'op', 'raw', 'length_bin', 'old_log_prob')

@torch.no_grad()
def prepare_rollout(samples, rewards, values, next_values, terminated, durations,
                    time_constant=600.):
    shape = rewards.shape
    arrays = (rewards, values, next_values, durations)
    if rewards.ndim != 2 or not rewards.numel():
        raise ValueError('Expected nonempty [T,B]')
    if any(x.shape != shape or x.device != rewards.device for x in (*arrays, terminated)):
        raise ValueError('Rollout shape/device mismatch')
    if terminated.dtype != torch.bool:
        raise ValueError('terminated must be Boolean; collection cutoff is not terminal')
    if any(not torch.isfinite(x).all() for x in arrays):
        raise ValueError('Nonfinite rollout')
    if (durations < 0).any() or not math.isfinite(time_constant) or time_constant <= 0:
        raise ValueError('Invalid time')
    for name in FIELDS:
        x = samples[name]
        if x.shape[:2] != shape or x.device != rewards.device:
            raise ValueError('Invalid stored field: ' + name)
    if samples['valid'].dtype != torch.bool or samples['valid'].shape != (*shape, 7):
        raise ValueError('Invalid stored operation mask')
    op = samples['op']
    if op.dtype != torch.long or op.shape != shape or ((op < 0) | (op >= 7)).any():
        raise ValueError('Invalid stored operation')
    if not samples['valid'].gather(-1, op.unsqueeze(-1)).all():
        raise ValueError('Stored action was masked during collection')
    advantage = torch.empty_like(values)
    carry = torch.zeros_like(values[0])
    for t in range(shape[0] - 1, -1, -1):
        alive = ~terminated[t]
        delta = rewards[t] + torch.where(alive, next_values[t], 0.) - values[t]
        carry = delta + torch.where(alive, torch.exp(-durations[t] / time_constant) * carry, 0.)
        advantage[t] = carry
    returns = advantage + values
    # Once across the whole rollout, never separately for each minibatch.
    normalized = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    batch = {k: samples[k].detach().reshape(-1, *samples[k].shape[2:]) for k in FIELDS}
    batch.update(returns=returns.flatten(), advantage=normalized.flatten())
    return batch

@dataclass(frozen=True)
class Schedule:
    epochs: int = 4
    minibatch: int = 512
    target_kl: float = .03


def optimize_rollout(policy, optimizer, batch, *, schedule=Schedule(), cfg=PPOConfig(), generator=None):
    """Stop BEFORE applying an update if current minibatch KL exceeds threshold.

    Generator lives on the batch device and must be checkpointed by the caller.
    Tail minibatches are retained. Metrics are sample-weighted over applied updates.
    No complete training or end-to-end acceleration claim follows from this API.
    """
    n = len(batch['returns'])
    if n < 1 or any(len(v) != n for v in batch.values()):
        raise ValueError('Empty/inconsistent batch')
    if schedule.epochs < 1 or schedule.minibatch < 1 or not math.isfinite(schedule.target_kl) or schedule.target_kl <= 0:
        raise ValueError('Invalid PPO schedule')
    device = batch['returns'].device
    sums = {}; updates = 0; count = 0; stopped = False; stopping_kl = None
    for epoch in range(schedule.epochs):
        order = torch.randperm(n, device=device, generator=generator)
        for ids in order.split(schedule.minibatch):
            b = {k: v[ids] for k, v in batch.items()}
            lp, value, oe, pe = policy.evaluate(b['maps'], b['scalars'], b['valid'], b['op'], b['raw'], b['length_bin'], entropy=True)
            loss, metrics = objective(lp, b['old_log_prob'], value, b['returns'], b['advantage'], oe, pe, cfg)
            kl = float(metrics['approx_kl'])
            if kl > schedule.target_kl:
                stopped = True; stopping_kl = kl
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            metrics.update(loss=loss.detach(), grad_norm=norm.detach())
            for k, v in metrics.items():
                sums[k] = sums.get(k, torch.zeros((), device=device)) + v * len(ids)
            updates += 1; count += len(ids)
        if stopped:
            break
    return dict(updates=updates, sample_updates=count, stopped_for_kl=stopped,
                stopping_kl=stopping_kl, metrics={k: float(v/count) for k,v in sums.items()})
