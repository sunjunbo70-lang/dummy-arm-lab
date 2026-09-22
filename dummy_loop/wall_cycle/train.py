"""Behaviour-cloning warm start plus PPO for the whole-cycle manager."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..wall.ppo import PPO, PPOConfig, Adam, gae
from .config import CycleConfig
from .env import WallCycleEnv, MODES

# One executor per process and kind: building the MuJoCo scene / loading the table is the
# slow part. The executors keep no episode state apart from the last joint solution (an IK seed).
_EXECUTOR = {}


def make_env(cfg, seed, teacher_style='technique', executor='table', **kw):
    """executor: 'table' (training: precomputed reach table, nominal-path mortar physics),
    'cosim' (evaluation / replay: full planner + MuJoCo co-simulation), or None."""
    ex = None
    if cfg.physics != 'v0.2' and cfg.use_arm and executor:
        key = (executor, cfg.physics, cfg.tool_profile, cfg.scene_wall_distance_m, cfg.area_centre_u_m,
               cfg.area_centre_z_m, cfg.width_m, cfg.height_m)
        if key not in _EXECUTOR:
            if executor == 'cosim':
                from .arm import ArmExecutor
                _EXECUTOR[key] = ArmExecutor(cfg)
            else:
                from .reach_table import ReachTableExecutor
                _EXECUTOR[key] = ReachTableExecutor(cfg)
        ex = _EXECUTOR[key]
    env = WallCycleEnv(cfg, seed, executor=ex, **kw)
    env.teacher_style = teacher_style
    return env


def collect_teacher(cfg, episodes=120, seed=0, teacher_style='technique'):
    obs, acts, returns = [], [], []
    for ep in range(episodes):
        env = make_env(cfg, seed+ep, teacher_style)
        o = env.reset(); done = False; ret = 0
        while not done:
            a = env.teacher_action(); obs.append(o); acts.append(a)
            o, r, done, _ = env.step(a); ret += r
        returns.append(ret)
    return np.asarray(obs), np.asarray(acts), returns


def collect_dagger(cfg, agent, episodes=25, seed=0, learner_probability=.7, teacher_style='technique'):
    """Label states visited by the learner, preventing open-loop BC drift."""
    rng = np.random.default_rng(seed); obs, labels = [], []
    for ep in range(episodes):
        env = make_env(cfg, seed+ep, teacher_style); o = env.reset(); done = False
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
    weights = np.array([5, 6, 3, 6, 3, 2, 2, 1, 1, 1, 4, 2, 2, 4][:actions.shape[1]], float)
    if len(weights) != actions.shape[1]:
        raise ValueError(f'no behaviour-cloning weights for {actions.shape[1]} action dimensions')
    for i in range(iters):
        mu = agent.pi.forward(x)
        d = 2*(mu-actions)*weights/len(x)
        gW, gb = agent.pi.backward(d); opt.step(agent.pi.params, gW+gb, 1.0)
        if log and (i+1) % 100 == 0:
            log(f'BC {i+1}: mse={np.mean((mu-actions)**2):.6f}')
    return float(np.mean((agent.pi.forward(x)-actions)**2))


def evaluate(cfg, agent=None, episodes=30, seed=10000, teacher_style='technique', executor='table'):
    rows = []; pitches = []; forces = []; strokes = []; carries = []; headings = []
    for i in range(episodes):
        env = make_env(cfg, seed+i, teacher_style, executor)
        o = env.reset(); done = False; ret = 0
        while not done:
            if agent is None:
                a = env.teacher_action()
            else:
                a = agent.act(o, deterministic=True)[0]
            if env.v3:
                d = env.decode(a)
                if d.mode in ('DEPOSIT', 'REUSE'):
                    pitches.append((np.rad2deg(d.pitch_start), np.rad2deg(d.pitch_end)))
                if d.mode in ('DEPOSIT', 'REUSE', 'LEVEL'):
                    forces.append(d.force_N)
                    if env.v5:
                        carries.append(d.carry_face_up)
                        dv = np.asarray(d.end)-np.asarray(d.start)
                        headings.append(float(np.rad2deg(np.arctan2(dv[1], dv[0])) % 180))
            o, r, done, info = env.step(a); ret += r
            if getattr(env, 'last_stroke', None) and executor == 'cosim':
                strokes.append(env.last_stroke); env.last_stroke = None
        rows.append({'return': ret, **info['metrics'], 'cycles': info['cycles'],
                     'steps': info['control_steps'], 'success': info['success'],
                     'unreachable': info.get('unreachable_strokes', 0),
                     'projected': info.get('projected_strokes', 0),
                     'carry_loss_frac': info.get('carry_loss_m3', 0.0) /
                     max(env.material.supplied_m3 + env.material.initial_m3, 18e-6),
                     'volume_error': abs(info['volume_balance_m3'])})
    keys = ('return','coverage','rmse_mm','p95_error_mm','waste_frac','carry_loss_frac',
            'cycles','steps','unreachable','projected')
    out = {f'{k}_mean': float(np.mean([r[k] for r in rows])) for k in keys} | {
        'success_rate': float(np.mean([r['success'] for r in rows])),
        'max_volume_error_m3': float(max(r['volume_error'] for r in rows)),
        'episodes': episodes, 'seeds': [seed, seed+episodes-1]}
    if pitches:
        p = np.asarray(pitches)
        out['deposit_pitch_start_deg_mean'] = float(p[:, 0].mean())
        out['deposit_pitch_end_deg_mean'] = float(p[:, 1].mean())
        out['deposit_pitch_start_deg_quartiles'] = [float(x) for x in np.percentile(p[:, 0], [25, 50, 75])]
    if forces:
        out['contact_force_N_mean'] = float(np.mean(forces))
    if carries:
        out['carry_face_up_mean'] = float(np.mean(carries))
    if headings:
        h = np.asarray(headings)
        out['path_heading_deg_quartiles'] = [float(x) for x in np.percentile(h, [25, 50, 75])]
        out['path_family_fraction'] = {
            'horizontal': float(np.mean((h < 22.5) | (h >= 157.5))),
            'diagonal': float(np.mean(((h >= 22.5) & (h < 67.5)) | ((h >= 112.5) & (h < 157.5)))),
            'vertical': float(np.mean((h >= 67.5) & (h < 112.5)))}
    if strokes:        # co-simulation diagnostics
        for k in ('force_rmse_N', 'tracking_max_mm', 'peak_force_N'):
            vals = [s_[k] for s_ in strokes if s_.get(k) is not None]
            if vals:
                out[f'cosim_{k}_mean'] = float(np.mean(vals)); out[f'cosim_{k}_max'] = float(np.max(vals))
        for k in ('carry_face_up_min', 'rotation_face_up_min'):
            vals = [s_[k] for s_ in strokes if s_.get(k) is not None]
            if vals:
                out[f'cosim_{k}'] = float(np.min(vals))
        for k in ('face_down_frames', 'approach_reversal_count'):
            vals = [s_[k] for s_ in strokes if s_.get(k) is not None]
            if vals:
                out[f'cosim_{k}_total'] = int(np.sum(vals))
    out['executor'] = executor
    return out


TEST_SEED, VAL_SEED = 20000, 30000


def train(out_dir, updates=60, seed=0, teacher_episodes=120, log=print, cfg=None,
          teacher_style='technique', steps_per_update=1024, lr=3e-5, test_episodes=100,
          val_episodes=40, eval_every=10, explore_log_std=-1.5, cosim_test_episodes=30,
          hidden=(128, 128), experiment_name=None, dagger_rounds=3, dagger_episodes=25):
    """Teacher -> BC -> 3x DAgger -> PPO, then compare ALL checkpoints on one fixed test set.

    v0.2 compared the BC checkpoint (30 episodes, seeds 11000+) with PPO (100 episodes,
    seeds 20000+): different sets, so the comparison was not like-for-like. Here the
    teacher, the DAgger/BC checkpoint and the PPO checkpoint are all scored on the same
    test seeds, and the deployed checkpoint is chosen on a SEPARATE validation set so the
    test numbers are not used for selection.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False)
    from .area import load_work_area
    cfg = load_work_area(cfg or CycleConfig())
    pcfg = PPOConfig(hidden=hidden, lr=lr, gamma=.99, lam=.95, clip=.10,
                     epochs=4, minibatch=128, steps_per_update=steps_per_update,
                     entropy_coef=0.0, init_log_std=-3.0, seed=seed)
    probe = make_env(cfg, seed, teacher_style)
    agent = PPO(probe.obs_dim, probe.act_dim, pcfg)
    t0 = time.time()
    O_demo, A_demo, teacher_rets = collect_teacher(cfg, teacher_episodes, seed, teacher_style)
    bc_mse = behaviour_clone(agent, O_demo, A_demo, log=log)
    dagger_rows = []
    for round_i in range(dagger_rounds):
        Od, Ad = collect_dagger(cfg, agent, dagger_episodes, seed+1000+round_i*100,
                                learner_probability=.45+.2*round_i, teacher_style=teacher_style)
        O_demo = np.concatenate([O_demo, Od]); A_demo = np.concatenate([A_demo, Ad])
        bc_mse = behaviour_clone(agent, O_demo, A_demo, iters=400, lr=3e-4, log=None)
        dagger_rows.append({'round': round_i+1, 'new_samples': len(Od),
                            'total_samples': len(O_demo), 'mse': bc_mse})
    agent.save(out/'policy_bc.npz')
    log(json.dumps({'stage': 'bc_done', 'elapsed_s': round(time.time()-t0, 1)}))
    # Exploration. All dims keep the small noise inherited from v0.2 (log_std -3: the warm
    # start's structure -- where to stroke, which mode -- must not be shaken apart), EXCEPT the
    # dims this experiment is about: contact force and the pitch profile. With log_std -3 a
    # flat-teacher policy explores +-0.9 deg of pitch and could never find out whether tilting
    # pays; -1.5 gives ~ +-4 deg per decision.
    explored = ('force', 'pitch_start', 'pitch_end')
    if cfg.physics in ('v0.5','v0.6'):
        explored += ('start_u', 'start_v', 'end_u', 'end_v',
                     'blade_cos', 'blade_sin')
        if cfg.physics == 'v0.5': explored += ('carry_face_up',)
    explore = [agent_i for agent_i, n in enumerate(probe.action_names) if n in explored]
    agent.log_std[explore] = explore_log_std
    env = make_env(cfg, seed+5000, teacher_style); obs = env.reset(); ep_ret = 0.; recent=[]; curve=[]

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

    # Warm the critic on on-policy data first, then freeze observation statistics.
    O,A,L,adv,ret=collect(pcfg.steps_per_update)
    value_warmup=agent.fit_value(O,ret,epochs=30);agent.norm.frozen=True
    best_val, best_update = -np.inf, 0
    for update in range(1,updates+1):
        O,A,L,adv,ret=collect(pcfg.steps_per_update)
        stats=agent.update(O,A,L,adv,ret)
        row={'update':update,'episodes':len(recent),'train_return_mean':float(np.mean(recent[-50:])) if recent else 0,
             **stats,'elapsed_s':round(time.time()-t0,1)}
        if update%eval_every==0 or update==updates:
            row['val']=evaluate(cfg,agent,val_episodes,VAL_SEED,teacher_style)
            if selection_score(row['val']) > best_val:
                best_val, best_update = selection_score(row['val']), update
                agent.save(out/'policy_ppo_best_val.npz')
        curve.append(row);log(json.dumps(row,ensure_ascii=False))
    agent.save(out/'policy_ppo.npz')
    # ---- like-for-like comparison on the fixed test set --------------------------------
    teacher_test = evaluate(cfg, None, test_episodes, TEST_SEED, teacher_style)
    bc = PPO(probe.obs_dim, probe.act_dim, pcfg).load(out/'policy_bc.npz')
    bc_test = evaluate(cfg, bc, test_episodes, TEST_SEED, teacher_style)
    bc_val = evaluate(cfg, bc, val_episodes, VAL_SEED, teacher_style)
    ppo_last_test = evaluate(cfg, agent, test_episodes, TEST_SEED, teacher_style)
    best = PPO(probe.obs_dim, probe.act_dim, pcfg).load(out/'policy_ppo_best_val.npz') if best_update else agent
    ppo_best_test = evaluate(cfg, best, test_episodes, TEST_SEED, teacher_style)
    if best_update and best_val > selection_score(bc_val):
        selected, src = f'ppo_update_{best_update}', out/'policy_ppo_best_val.npz'
    else:
        selected, src = 'dagger_bc', out/'policy_bc.npz'
    (out/'policy.npz').write_bytes(Path(src).read_bytes())
    # The training environment approximates the arm (reach table, nominal path). Re-test the
    # teacher and the deployed checkpoint with the full planner + MuJoCo co-simulation.
    cosim = {}
    if cosim_test_episodes and cfg.use_arm:
        sel = PPO(probe.obs_dim, probe.act_dim, pcfg).load(src)
        cosim = {'teacher': evaluate(cfg, None, cosim_test_episodes, TEST_SEED, teacher_style, 'cosim'),
                 'selected': evaluate(cfg, sel, cosim_test_episodes, TEST_SEED, teacher_style, 'cosim')}
        log(json.dumps({'stage': 'cosim_test_done', 'elapsed_s': round(time.time()-t0, 1)}))
    result={'experiment':experiment_name or f'wall_cycle_{cfg.physics}',
            'evidence_level':'L1','hardware_motion':False,
            'seed':seed,'teacher_style':teacher_style,'updates':updates,
            'ppo_steps':updates*pcfg.steps_per_update,
            'teacher_samples':len(O_demo),'teacher_return_mean':float(np.mean(teacher_rets)),
            'behaviour_clone_mse':bc_mse,'dagger':dagger_rows,'value_warmup_mse':value_warmup,
            'config':cfg.to_dict(),'ppo':pcfg.to_dict(),
            'test_set':{'seeds':[TEST_SEED, TEST_SEED+test_episodes-1],'episodes':test_episodes},
            'validation_set':{'seeds':[VAL_SEED, VAL_SEED+val_episodes-1],'episodes':val_episodes},
            'test':{'teacher':teacher_test,'dagger_bc':bc_test,'ppo_last':ppo_last_test,
                    f'ppo_best_val_update_{best_update}':ppo_best_test},
            'validation_bc':bc_val,'selected_checkpoint':selected,
            'test_cosim':cosim,'exploration':{'log_std_default':pcfg.init_log_std,'explored_dims':
                          [probe.action_names[i] for i in explore],'explored_log_std':explore_log_std},
            'curve':curve,'elapsed_s':round(time.time()-t0,1),
            'limitations':['reduced-order 2.5-D mortar, not CFD; trowel coefficients assumed',
                           'D435 parameters provisional until physical calibration',
                           'MuJoCo servos/contact are simulated; parameters are not yet identified from hardware',
                           'feed is gated by the executed face-up pose but remains a reduced feed proxy, not granular contact',
                           'free-space transport and approach are constrained motion skills; manager acts once per scan']}
    (out/'training.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def selection_score(x):
    return 10*x['success_rate'] + x['coverage_mean'] - .2*x['rmse_mm_mean'] - 2*x['waste_frac_mean']


def main():
    p=argparse.ArgumentParser(description='Whole-wall plastering: teacher -> BC -> DAgger -> PPO (v0.3 physics, real-arm executor).')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--updates',type=int,default=60);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--teacher-episodes',type=int,default=120)
    p.add_argument('--teacher-style',choices=('technique','flat','legacy_transport'),default='technique',
                   help="technique = meet the wall tilted, flatten; flat = control group (no tilt)")
    p.add_argument('--steps-per-update',type=int,default=1024)
    p.add_argument('--test-episodes',type=int,default=100)
    p.add_argument('--val-episodes',type=int,default=40)
    p.add_argument('--cosim-test-episodes',type=int,default=30)
    p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--physics',choices=('v0.5','v0.6'),default='v0.6')
    p.add_argument('--dagger-rounds',type=int,default=4)
    p.add_argument('--dagger-episodes',type=int,default=40)
    p.add_argument('--no-arm',action='store_true',help='skip the MuJoCo reachability executor (faster, less faithful)')
    a=p.parse_args()
    v6=a.physics=='v0.6'
    cfg=CycleConfig(use_arm=not a.no_arm, physics=a.physics, tool_profile='lab_20260922',
                    lift_wall_fraction=.75,
                    base_steps=12000 if v6 else 6000,max_steps=24000 if v6 else 12000,
                    extension_steps=2000 if v6 else 1000,max_cycles=160 if v6 else 80,
                    max_reload_cycles=60 if v6 else 30,stall_limit=1 if v6 else 5,
                    stall_window=12 if v6 else 5,min_cycles_before_stall=25 if v6 else 0)
    r=train(a.out,a.updates,a.seed,a.teacher_episodes,cfg=cfg,teacher_style=a.teacher_style,
            steps_per_update=a.steps_per_update, test_episodes=a.test_episodes,
            val_episodes=a.val_episodes, cosim_test_episodes=a.cosim_test_episodes,
            eval_every=a.eval_every, hidden=(256,256),
            experiment_name='wall_cycle_v0.6_p2' if v6 else 'wall_cycle_v0.5_p1',
            dagger_rounds=a.dagger_rounds,dagger_episodes=a.dagger_episodes,
            log=lambda x: print(x, flush=True))
    print(json.dumps({'selected':r['selected_checkpoint'],'test':r['test']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
