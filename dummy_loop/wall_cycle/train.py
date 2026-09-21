"""Behaviour-cloning warm start plus PPO for the whole-cycle manager."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..wall.ppo import PPO, PPOConfig, Adam, gae
from .config import CycleConfig
from .env import WallCycleEnv


def collect_teacher(cfg, episodes=120, seed=0):
    obs, acts, returns = [], [], []
    for ep in range(episodes):
        env = WallCycleEnv(cfg, seed+ep)
        o = env.reset(); done = False; ret = 0
        while not done:
            a = env.teacher_action(); obs.append(o); acts.append(a)
            o, r, done, _ = env.step(a); ret += r
        returns.append(ret)
    return np.asarray(obs), np.asarray(acts), returns


def collect_dagger(cfg, agent, episodes=25, seed=0, learner_probability=.7):
    """Label states visited by the learner, preventing open-loop BC drift."""
    rng = np.random.default_rng(seed); obs, labels = [], []
    for ep in range(episodes):
        env = WallCycleEnv(cfg, seed+ep); o = env.reset(); done = False
        while not done:
            teacher = env.teacher_action(); obs.append(o); labels.append(teacher)
            learned = agent.act(o, deterministic=True)[0]
            executed = learned if rng.random() < learner_probability else teacher
            o, _, done, _ = env.step(executed)
    return np.asarray(obs), np.asarray(labels)


def behaviour_clone(agent, observations, actions, iters=800, lr=5e-4, log=print):
    agent.norm.update(observations); x = agent.norm(observations)
    opt = Adam(agent.pi.params, lr)
    # Position/mode mistakes cause distribution shift much sooner than small
    # force/speed errors, so fit them more strongly than scalar refinements.
    weights = np.array([5, 6, 3, 6, 3, 2, 2, 1, 1, 1, 4], float)
    for i in range(iters):
        mu = agent.pi.forward(x)
        d = 2*(mu-actions)*weights/len(x)
        gW, gb = agent.pi.backward(d); opt.step(agent.pi.params, gW+gb, 1.0)
        if log and (i+1) % 100 == 0:
            log(f'BC {i+1}: mse={np.mean((mu-actions)**2):.6f}')
    return float(np.mean((agent.pi.forward(x)-actions)**2))


def evaluate(cfg, agent=None, episodes=30, seed=10000):
    rows = []
    for i in range(episodes):
        env = WallCycleEnv(cfg, seed+i)
        o = env.reset(); done = False; ret = 0
        while not done:
            if agent is None:
                a = env.teacher_action()
            else:
                a = agent.act(o, deterministic=True)[0]
            o, r, done, info = env.step(a); ret += r
        rows.append({'return': ret, **info['metrics'], 'cycles': info['cycles'],
                     'steps': info['control_steps'], 'success': info['success'],
                     'volume_error': abs(info['volume_balance_m3'])})
    keys = ('return','coverage','rmse_mm','p95_error_mm','waste_frac','cycles','steps')
    return {f'{k}_mean': float(np.mean([r[k] for r in rows])) for k in keys} | {
        'success_rate': float(np.mean([r['success'] for r in rows])),
        'max_volume_error_m3': float(max(r['volume_error'] for r in rows)),
        'episodes': episodes}


def train(out_dir, updates=60, seed=0, teacher_episodes=120, log=print):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False)
    cfg = CycleConfig()
    pcfg = PPOConfig(hidden=(128,128), lr=2e-5, gamma=.99, lam=.95, clip=.10,
                     epochs=4, minibatch=128, steps_per_update=512,
                     entropy_coef=0.0, init_log_std=-3.0, seed=seed)
    probe = WallCycleEnv(cfg, seed)
    agent = PPO(probe.obs_dim, probe.act_dim, pcfg)
    t0 = time.time(); O_demo, A_demo, teacher_rets = collect_teacher(cfg, teacher_episodes, seed)
    bc_mse = behaviour_clone(agent, O_demo, A_demo, log=log)
    dagger_rows = []
    for round_i in range(3):
        Od, Ad = collect_dagger(cfg, agent, 25, seed+1000+round_i*100,
                                learner_probability=.45+.2*round_i)
        O_demo = np.concatenate([O_demo, Od]); A_demo = np.concatenate([A_demo, Ad])
        bc_mse = behaviour_clone(agent, O_demo, A_demo, iters=400, lr=3e-4, log=None)
        dagger_rows.append({'round': round_i+1, 'new_samples': len(Od),
                            'total_samples': len(O_demo), 'mse': bc_mse})
    teacher_eval = evaluate(cfg, None, 30, 10000)
    after_bc = evaluate(cfg, agent, 30, 11000)
    agent.save(out/'policy_bc.npz')
    env = WallCycleEnv(cfg, seed+5000); obs = env.reset(); ep_ret = 0.; recent=[]; curve=[]

    # Warm the critic on one on-policy batch, then freeze observation statistics.
    def collect(n):
        nonlocal obs, ep_ret
        O,A,L,R,D,V=[],[],[],[],[],[]
        for _ in range(n):
            a, raw, lp = agent.act(obs); value = agent.value(obs)
            nxt, reward, done, _ = env.step(a)
            O.append(obs);A.append(raw);L.append(lp);R.append(reward);D.append(done);V.append(value)
            ep_ret += reward; obs=nxt
            if done:
                recent.append(ep_ret);ep_ret=0.;obs=env.reset()
        adv, ret = gae(np.asarray(R),np.asarray(V),np.asarray(D),agent.value(obs),pcfg.gamma,pcfg.lam)
        return np.asarray(O),np.asarray(A),np.asarray(L),adv,ret

    O,A,L,adv,ret=collect(pcfg.steps_per_update)
    value_warmup=agent.fit_value(O,ret,epochs=20);agent.norm.frozen=True
    for update in range(1,updates+1):
        O,A,L,adv,ret=collect(pcfg.steps_per_update)
        stats=agent.update(O,A,L,adv,ret)
        row={'update':update,'episodes':len(recent),'train_return_mean':float(np.mean(recent[-50:])) if recent else 0,
             **stats,'elapsed_s':round(time.time()-t0,1)}
        if update%10==0 or update==updates:
            row['eval']=evaluate(cfg,agent,20,12000+update*100)
        curve.append(row);log(json.dumps(row,ensure_ascii=False))
    agent.save(out/'policy_ppo.npz')
    ppo_final=evaluate(cfg,agent,100,20000)
    def selection_score(x):
        return 10*x['success_rate'] + x['coverage_mean'] - .2*x['rmse_mm_mean']
    if selection_score(ppo_final) >= selection_score(after_bc):
        selected='ppo'; final=ppo_final
    else:
        selected='dagger_bc'; agent.load(out/'policy_bc.npz'); final=evaluate(cfg,agent,100,21000)
    agent.save(out/'policy.npz')
    result={'experiment':'wall_cycle_v2','evidence_level':'L1','hardware_motion':False,
            'seed':seed,'updates':updates,'ppo_steps':updates*pcfg.steps_per_update,
            'teacher_samples':len(O_demo),'teacher_return_mean':float(np.mean(teacher_rets)),
            'behaviour_clone_mse':bc_mse,'dagger':dagger_rows,'value_warmup_mse':value_warmup,
            'config':cfg.to_dict(),'ppo':pcfg.to_dict(),'teacher_eval':teacher_eval,
            'after_bc_eval':after_bc,'ppo_final':ppo_final,'selected_checkpoint':selected,
            'learned_final':final,'curve':curve,
            'elapsed_s':round(time.time()-t0,1),
            'limitations':['reduced-order 2.5-D mortar, not CFD','D435 parameters provisional until physical calibration',
                           'non-contact phases are deterministic motion skills; manager acts once per scan']}
    (out/'training.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--updates',type=int,default=60);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--teacher-episodes',type=int,default=120);a=p.parse_args()
    r=train(a.out,a.updates,a.seed,a.teacher_episodes)
    print(json.dumps(r['learned_final'],ensure_ascii=False,indent=2))


if __name__=='__main__': main()
