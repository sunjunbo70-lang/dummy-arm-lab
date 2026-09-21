"""抹涂手法的强化学习闭环：StrokeEnv + 纯 numpy PPO。

跑法：
    python -m dummy_loop.wall rl-train --updates 60 --out outputs/wall/rl
    python -m dummy_loop.wall rl-eval  --policy outputs/wall/rl/policy.npz --episodes 20

训练目标就是用户说的那一刀：从待命位出发，刀面下缘先贴墙，一边上行一边放平，
把刀上的料均匀抹到工作区里，尽量别掉料、别撞墙。奖励定义见 StrokeConfig 与 docs/RL.md。

**这是 L1 仿真结论**：材料是降阶高度场代理模型，参数是假设值。学出来的策略只说明
「这套奖励 + 这套环境能不能学到动作结构」，不能直接搬到实机。
"""
import json
import time
from pathlib import Path

import numpy as np

from .ppo import PPO, PPOConfig, gae
from .stroke_env import StrokeEnv, StrokeConfig, rollout
from .errors import Perturbation


def make_env(seed=0, random_scale=0.0, stroke=None):
    pert = Perturbation.sample(np.random.default_rng(seed), random_scale) if random_scale > 0 else Perturbation()
    return StrokeEnv(perturbation=pert, stroke=stroke, seed=seed)


class Scaled:
    """把 PPO 的 [-1,1] 动作换算成环境的物理增量。"""

    def __init__(self, env, agent, deterministic=False):
        self.env, self.agent, self.det = env, agent, deterministic
        self.scale = env.spec.limits[[1, 2, 4]]

    def __call__(self, obs):
        a, _, _ = self.agent.act(obs, deterministic=self.det)
        return a * self.scale


def behaviour_clone(agent, env, episodes=20, iters=400, lr=1e-3, log=None):
    """用脚本基线做监督预热：把策略均值回归到脚本动作。

    纯从零探索时，机械臂几乎碰不到墙，奖励是稀疏的，PPO 学不动。先克隆脚本手法再用 PPO 去优化，
    正是实机上推荐的路线（示教打底 + 强化学习微调），这里在仿真里先把它跑通。
    """
    scale = env.spec.limits[[1, 2, 4]]
    O, A = [], []
    for _ in range(episodes):
        rollout(env, None, record=lambda o, a, r, i: (O.append(o), A.append(np.clip(a / scale, -1, 1))))
    X, Y = np.array(O), np.array(A)
    agent.norm.update(X)
    Xn = agent.norm(X)
    from .ppo import Adam
    opt = Adam(agent.pi.params, lr)
    for i in range(iters):
        mu = agent.pi.forward(Xn)
        d = 2.0 * (mu - Y) / len(Xn)
        gW, gb = agent.pi.backward(d)
        opt.step(agent.pi.params, gW + gb, 1.0)
        if log and (i + 1) % 100 == 0:
            log(f'behaviour clone iter {i + 1}: mse={float(np.mean((mu - Y) ** 2)):.5f}')
    return float(np.mean((agent.pi.forward(Xn) - Y) ** 2))


def evaluate(env, agent=None, episodes=10):
    """跑若干 episode，返回平均回报与平均指标。agent=None 用脚本基线。"""
    policy = None if agent is None else Scaled(env, agent, deterministic=True)
    rets, ms = [], []
    for _ in range(episodes):
        ret, _, m = rollout(env, policy)
        rets.append(ret); ms.append(m)
    keys = ('coverage', 'rms_error_mm', 'flatness_std_mm', 'wasted_frac', 'left_on_tool_frac', 'max_force_N')
    return {'return_mean': round(float(np.mean(rets)), 3), 'return_min': round(float(np.min(rets)), 3),
            **{k: round(float(np.mean([m[k] for m in ms])), 4) for k in keys},
            'aborted_episodes': int(sum(m['aborted'] for m in ms)), 'episodes': episodes}


def train(out_dir, updates=60, cfg: PPOConfig = None, stroke: StrokeConfig = None,
          random_scale=0.0, seed=0, eval_every=10, eval_episodes=10, log=print, warm_start=True):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or PPOConfig(seed=seed)
    env = make_env(seed, random_scale, stroke)
    eval_env = make_env(seed + 1000, random_scale, stroke)
    agent = PPO(env.obs_dim, env.act_dim, cfg)
    scale = env.spec.limits[[1, 2, 4]]
    t0 = time.time()
    bc_mse = behaviour_clone(agent, env, log=log) if warm_start else None
    obs = env.reset(); ep_ret, ep_rets, curve = 0.0, [], []
    baseline = evaluate(eval_env, None, eval_episodes)
    after_bc = None
    if warm_start:
        after_bc = evaluate(eval_env, agent, eval_episodes)
        log(json.dumps({'after_behaviour_clone': after_bc}, ensure_ascii=False))
    def collect(n):
        nonlocal obs, ep_ret
        O, A, L, R, D, V = [], [], [], [], [], []
        for _ in range(n):
            a, raw, logp = agent.act(obs)
            v = agent.value(obs)
            nxt, r, done, _ = env.step(a * scale)
            O.append(obs); A.append(raw); L.append(logp); R.append(r); D.append(done); V.append(v)
            ep_ret += r; obs = nxt
            if done:
                ep_rets.append(ep_ret); ep_ret = 0.0; obs = env.reset()
        adv, ret = gae(np.array(R), np.array(V), np.array(D), agent.value(obs), cfg.gamma, cfg.lam)
        return np.array(O), np.array(A), np.array(L), adv, ret

    if warm_start:      # 价值函数预热：先只练 critic，免得第一次策略更新把克隆来的手法打坏
        O, A, L, adv, ret = collect(cfg.steps_per_update)
        agent.norm.frozen = True          # 观测归一化在预热后冻结
        vmse = agent.fit_value(O, ret)
        log(json.dumps({'value_warmup_mse': round(vmse, 4)}, ensure_ascii=False))
    for it in range(1, updates + 1):
        O, A, L, R, D, V = [], [], [], [], [], []
        for _ in range(cfg.steps_per_update):
            a, raw, logp = agent.act(obs)
            v = agent.value(obs)
            nxt, r, done, _ = env.step(a * scale)
            O.append(obs); A.append(raw); L.append(logp); R.append(r); D.append(done); V.append(v)
            ep_ret += r; obs = nxt
            if done:
                ep_rets.append(ep_ret); ep_ret = 0.0; obs = env.reset()
        adv, ret = gae(np.array(R), np.array(V), np.array(D), agent.value(obs), cfg.gamma, cfg.lam)
        stats = agent.update(np.array(O), np.array(A), np.array(L), adv, ret)
        row = {'update': it, 'episodes': len(ep_rets),
               'train_return_mean': round(float(np.mean(ep_rets[-50:])), 3), **stats,
               'elapsed_s': round(time.time() - t0, 1)}
        if it % eval_every == 0 or it == updates:
            row['eval'] = evaluate(eval_env, agent, eval_episodes)
        curve.append(row)
        log(json.dumps(row, ensure_ascii=False))
    agent.save(out / 'policy.npz')
    final = evaluate(eval_env, agent, eval_episodes)
    result = {'evidence_level': 'L1', 'hardware_motion': False, 'updates': updates,
              'warm_start_behaviour_clone_mse': bc_mse,
              'steps': updates * cfg.steps_per_update, 'ppo': cfg.to_dict(),
              'env': env.describe(), 'random_scale': random_scale, 'seed': seed,
              'scripted_baseline': baseline, 'after_behaviour_clone': after_bc, 'learned_final': final,
              'random_policy': evaluate_random(eval_env, seed, eval_episodes),
              'curve': curve, 'elapsed_s': round(time.time() - t0, 1)}
    (out / 'training.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return result


def evaluate_random(env, seed=0, episodes=10):
    rng = np.random.default_rng(seed)
    scale = env.spec.limits[[1, 2, 4]]
    rets = [rollout(env, lambda o: rng.uniform(-1, 1, 3) * scale)[0] for _ in range(episodes)]
    return {'return_mean': round(float(np.mean(rets)), 3), 'episodes': episodes}


def load_policy(env, path, cfg: PPOConfig = None):
    agent = PPO(env.obs_dim, env.act_dim, cfg or PPOConfig())
    agent.load(path)
    return agent
