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


def run_eval_episode(env, agent, executor):
    """One evaluation episode -> record consumed by aggregate(). Shared by the serial path and
    the worker processes (parallel.eval_episode), so both give identical per-episode numbers."""
    pitches, forces, carries, headings, strokes = [], [], [], [], []
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
    supplied = max(env.material.supplied_m3 + env.material.initial_m3, 18e-6)
    ledger = info.get('loss_ledger_m3') or {}
    row = {'return': ret, **info['metrics'], 'cycles': info['cycles'],
           'steps': info['control_steps'], 'success': info['success'],
           'unreachable': info.get('unreachable_strokes', 0),
           'projected': info.get('projected_strokes', 0),
           'carry_loss_frac': info.get('carry_loss_m3', 0.0) / supplied,
           'other_drop_frac': ledger.get('other_drop_m3', 0.0) / supplied,
           'outside_frac': ledger.get('outside_m3', 0.0) / supplied,
           'end_reason': info.get('end_reason'),
           'volume_error': abs(info['volume_balance_m3'])}
    return {'row': row, 'pitches': pitches, 'forces': forces, 'carries': carries,
            'headings': headings, 'strokes': strokes}


def aggregate(records, episodes, seed, executor):
    rows = [r['row'] for r in records]
    pitches = [x for r in records for x in r['pitches']]; forces = [x for r in records for x in r['forces']]
    carries = [x for r in records for x in r['carries']]; headings = [x for r in records for x in r['headings']]
    strokes = [x for r in records for x in r['strokes']]
    keys = ('return','coverage','rmse_mm','p95_error_mm','waste_frac','carry_loss_frac',
            'cycles','steps','unreachable','projected')
    out = {f'{k}_mean': float(np.mean([r[k] for r in rows])) for k in keys} | {
        'success_rate': float(np.mean([r['success'] for r in rows])),
        'max_volume_error_m3': float(max(r['volume_error'] for r in rows)),
        'episodes': episodes, 'seeds': [seed, seed+episodes-1]}
    # v0.8 additions (present when the environment reports them)
    for k in ('edge_coverage', 'edge_bare_frac', 'other_drop_frac', 'outside_frac'):
        if all(k in r for r in rows):
            out[f'{k}_mean'] = float(np.mean([r[k] for r in rows]))
    if rows and 'score_side_m' in rows[0]:
        out['score_side_m'] = rows[0]['score_side_m']
    reasons = [r.get('end_reason') for r in rows if r.get('end_reason')]
    if reasons:
        out['end_reasons'] = {k: reasons.count(k) for k in sorted(set(reasons))}
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
        for k in ('face_down_frames', 'approach_reversal_count', 'feed_transit_loaded_tilted_frames'):
            vals = [s_[k] for s_ in strokes if s_.get(k) is not None]
            if vals:
                out[f'cosim_{k}_total'] = int(np.sum(vals))
    out['executor'] = executor
    return out


def evaluate(cfg, agent=None, episodes=30, seed=10000, teacher_style='technique', executor='table',
             runner=None, policy_path=None, hidden=None):
    """Serial when runner is None (v0.7 path); otherwise episodes go to runner's workers and the
    policy travels as policy_path. Per-episode results are identical either way."""
    if runner is None:
        records = [run_eval_episode(make_env(cfg, seed+i, teacher_style, executor), agent, executor)
                   for i in range(episodes)]
    else:
        from .parallel import eval_episode
        records = runner.map(eval_episode, [(cfg, policy_path, hidden, seed+i, teacher_style, executor)
                                            for i in range(episodes)])
    return aggregate(records, episodes, seed, executor)


TEST_SEED, VAL_SEED = 20000, 30000
# v0.7: promotion to a larger next stage (e.g. P2-B) additionally requires PPO's coverage not
# to have regressed against DAgger-BC by more than this fraction. P2-A's "2 of 3 metrics"
# gate alone let a policy pass by trading coverage away for a lower waste_frac; this closes
# that specific hole without loosening the existing 2-of-3 rule.
MIN_COVERAGE_RATIO = 0.90


def promotion_ok(ppo_test, dagger_test):
    """Bool verdict for whether PPO earns the next, larger training stage. Does not change
    which checkpoint gets deployed (selection_score/best_val still decide that) -- this only
    gates P2-B-style scale-ups."""
    two_of_three = sum([ppo_test['coverage_mean'] >= dagger_test['coverage_mean'],
                        ppo_test['rmse_mm_mean'] <= dagger_test['rmse_mm_mean'],
                        ppo_test['waste_frac_mean'] <= dagger_test['waste_frac_mean']]) >= 2
    no_regression = ppo_test['coverage_mean'] >= MIN_COVERAGE_RATIO * dagger_test['coverage_mean']
    return {'promoted': bool(two_of_three and no_regression),
            'two_of_three_metrics': bool(two_of_three),
            'coverage_regression_ok': bool(no_regression),
            'min_coverage_ratio': MIN_COVERAGE_RATIO}


def train(out_dir, updates=60, seed=0, teacher_episodes=120, log=print, cfg=None,
          teacher_style='technique', steps_per_update=1024, lr=3e-5, test_episodes=100,
          val_episodes=40, eval_every=10, explore_log_std=-1.5, cosim_test_episodes=30,
          hidden=(128, 128), experiment_name=None, dagger_rounds=3, dagger_episodes=25,
          recipe=None, workers=1):
    """Teacher -> BC -> 3x DAgger -> PPO, then compare ALL checkpoints on one fixed test set.

    v0.2 compared the BC checkpoint (30 episodes, seeds 11000+) with PPO (100 episodes,
    seeds 20000+): different sets, so the comparison was not like-for-like. Here the
    teacher, the DAgger/BC checkpoint and the PPO checkpoint are all scored on the same
    test seeds, and the deployed checkpoint is chosen on a SEPARATE validation set so the
    test numbers are not used for selection.
    """
    if recipe == 'v0.8':
        return train_v08(out_dir, updates=updates, seed=seed, teacher_episodes=teacher_episodes, log=log,
                         cfg=cfg, teacher_style=teacher_style, steps_per_update=steps_per_update, lr=lr,
                         test_episodes=test_episodes, val_episodes=val_episodes, eval_every=eval_every,
                         explore_log_std=explore_log_std, cosim_test_episodes=cosim_test_episodes,
                         hidden=hidden, experiment_name=experiment_name, dagger_rounds=dagger_rounds,
                         dagger_episodes=dagger_episodes, workers=workers)
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
    # v0.7: promotion verdict for the PPO checkpoint actually being considered for deployment
    # (best-val if it beat BC on the validation set, otherwise last-update), against DAgger-BC
    # on the fixed test set. This does not affect `selected` above.
    promotion = promotion_ok(ppo_best_test if (best_update and best_val > selection_score(bc_val))
                             else ppo_last_test, bc_test)
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
            'validation_bc':bc_val,'selected_checkpoint':selected,'promotion':promotion,
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


# v0.8: a PPO checkpoint is only eligible for selection if, on the validation set, neither its
# coverage nor its edge-band coverage fell below this fraction of DAgger-BC's (review
# 2026-09-24: v0.7 deployed a PPO checkpoint whose coverage was 15 % below BC's).
ELIGIBLE_RATIO = 0.90


def eligible(val, bc_val):
    return (val['coverage_mean'] >= ELIGIBLE_RATIO * bc_val['coverage_mean'] and
            val.get('edge_coverage_mean', 0.0) >= ELIGIBLE_RATIO * bc_val.get('edge_coverage_mean', 0.0))


def train_v08(out_dir, updates, seed, teacher_episodes, log, cfg, teacher_style, steps_per_update, lr,
              test_episodes, val_episodes, eval_every, explore_log_std, cosim_test_episodes, hidden,
              experiment_name, dagger_rounds, dagger_episodes, workers):
    """v0.8 flow: same stages as train(), but every episode-level job runs on worker processes,
    PPO samples from persistent rollout workers, checkpoints are selected on a fresh validation
    set under coverage/edge-coverage eligibility, and tests use a fresh seed set (the v0.7 test
    set is kept as a historical regression set)."""
    from .parallel import Runner, RolloutPool, teacher_episode, dagger_episode
    from .recipes import SEEDS
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False)
    work = out/'_work'; work.mkdir()
    seeds = SEEDS['v0.8']; val_seed, test_seed, legacy_seed = seeds['val'], seeds['test'], seeds['legacy']
    runner = Runner(workers); n_workers = runner.workers
    pcfg = PPOConfig(hidden=hidden, lr=lr, gamma=.99, lam=.95, clip=.10,
                     epochs=4, minibatch=128, steps_per_update=steps_per_update,
                     entropy_coef=0.0, init_log_std=-3.0, seed=seed)
    probe = make_env(cfg, seed, teacher_style)
    agent = PPO(probe.obs_dim, probe.act_dim, pcfg)
    t0 = time.time(); stage = {}
    published = []

    def publish(ag):
        """Write the current policy to a NEW file (workers cache by path) and drop old ones."""
        path = work/f'policy_{len(published)+1:04d}.npz'; ag.save(path); published.append(path)
        for old in published[:-2]:
            if old.exists():
                old.unlink()
        return path

    def ev(policy_path, episodes, seed0, executor='table'):
        return evaluate(cfg, None, episodes, seed0, teacher_style, executor,
                        runner=runner, policy_path=policy_path, hidden=hidden)

    def mark(name):
        stage[name] = round(time.time()-t0, 1)
        log(json.dumps({'stage': name, 'elapsed_s': stage[name], 'workers': n_workers}))

    log(json.dumps({'recipe': 'v0.8', 'workers': n_workers, 'obs_dim': probe.obs_dim,
                    'score_side_m': probe.material.metrics()['score_side_m']}))
    # ---- teacher demonstrations, BC, DAgger ------------------------------------------------
    demos = runner.map(teacher_episode, [(cfg, seed+ep, teacher_style) for ep in range(teacher_episodes)])
    O_demo = np.concatenate([d[0] for d in demos]); A_demo = np.concatenate([d[1] for d in demos])
    teacher_rets = [d[2] for d in demos]
    mark('teacher_done')
    bc_mse = behaviour_clone(agent, O_demo, A_demo, log=log)
    dagger_rows = []
    for round_i in range(dagger_rounds):
        path = publish(agent)
        base = seed+1000+round_i*100
        res = runner.map(dagger_episode, [(cfg, path, hidden, base+ep, (seed, round_i, ep),
                                           .45+.2*round_i, teacher_style) for ep in range(dagger_episodes)])
        Od = np.concatenate([r[0] for r in res]); Ad = np.concatenate([r[1] for r in res])
        O_demo = np.concatenate([O_demo, Od]); A_demo = np.concatenate([A_demo, Ad])
        bc_mse = behaviour_clone(agent, O_demo, A_demo, iters=400, lr=3e-4, log=None)
        dagger_rows.append({'round': round_i+1, 'new_samples': len(Od),
                            'total_samples': len(O_demo), 'mse': bc_mse})
    agent.save(out/'policy_bc.npz')
    mark('bc_done')
    bc_val = ev(out/'policy_bc.npz', val_episodes, val_seed)
    log(json.dumps({'stage': 'bc_validation', 'val': {k: bc_val[k] for k in
                    ('coverage_mean', 'edge_coverage_mean', 'rmse_mm_mean', 'waste_frac_mean', 'cycles_mean')}}))
    # ---- PPO --------------------------------------------------------------------------------
    explored = ('force', 'pitch_start', 'pitch_end', 'start_u', 'start_v', 'end_u', 'end_v',
                'blade_cos', 'blade_sin')
    explore = [i for i, n in enumerate(probe.action_names) if n in explored]
    agent.log_std[explore] = explore_log_std
    rollouts = RolloutPool(cfg, workers, seed, teacher_style)
    recent, curve = [], []
    calls = [0]

    def collect(n):
        path = publish(agent); calls[0] += 1
        parts = rollouts.collect(path, list(hidden), n, calls[0])
        Os, As, Ls, advs, rets = [], [], [], [], []
        for O, A, L, R, D, last_obs, finished in parts:
            V = agent.vf.forward(agent.norm(O))[:, 0]
            adv, ret = gae(R, V, D, agent.value(last_obs), pcfg.gamma, pcfg.lam)
            Os.append(O); As.append(A); Ls.append(L); advs.append(adv); rets.append(ret)
            recent.extend(finished)
        return (np.concatenate(Os), np.concatenate(As), np.concatenate(Ls),
                np.concatenate(advs), np.concatenate(rets))

    try:
        O, A, L, adv, ret = collect(pcfg.steps_per_update)
        value_warmup = agent.fit_value(O, ret, epochs=30); agent.norm.frozen = True
        best_val, best_update = -np.inf, 0
        for update in range(1, updates+1):
            O, A, L, adv, ret = collect(pcfg.steps_per_update)
            stats = agent.update(O, A, L, adv, ret)
            row = {'update': update, 'episodes': len(recent),
                   'train_return_mean': float(np.mean(recent[-50:])) if recent else 0,
                   **stats, 'elapsed_s': round(time.time()-t0, 1)}
            if update % eval_every == 0 or update == updates:
                path = publish(agent)
                row['val'] = ev(path, val_episodes, val_seed)
                row['val_eligible'] = eligible(row['val'], bc_val)
                if row['val_eligible'] and selection_score(row['val']) > best_val:
                    best_val, best_update = selection_score(row['val']), update
                    agent.save(out/'policy_ppo_best_val.npz')
            curve.append(row); log(json.dumps(row, ensure_ascii=False))
    finally:
        rollouts.close()
    agent.save(out/'policy_ppo.npz')
    mark('ppo_done')
    # ---- fixed test set (fresh seeds) --------------------------------------------------------
    teacher_test = ev(None, test_episodes, test_seed)
    bc_test = ev(out/'policy_bc.npz', test_episodes, test_seed)
    ppo_last_test = ev(out/'policy_ppo.npz', test_episodes, test_seed)
    ppo_best_test = ev(out/'policy_ppo_best_val.npz', test_episodes, test_seed) if best_update else None
    if best_update and best_val > selection_score(bc_val):
        selected, src = f'ppo_update_{best_update}', out/'policy_ppo_best_val.npz'
        considered = ppo_best_test
    else:
        selected, src = 'dagger_bc', out/'policy_bc.npz'
        considered = ppo_best_test or ppo_last_test
    (out/'policy.npz').write_bytes(Path(src).read_bytes())
    mark('test_done')
    cosim = {}
    if cosim_test_episodes and cfg.use_arm:
        cosim = {'teacher': ev(None, cosim_test_episodes, test_seed, 'cosim'),
                 'selected': ev(out/'policy.npz', cosim_test_episodes, test_seed, 'cosim')}
        mark('cosim_test_done')
    legacy = {'teacher': ev(None, test_episodes, legacy_seed),
              'selected': ev(out/'policy.npz', test_episodes, legacy_seed)}
    mark('legacy_test_done')
    runner.close()
    for f in work.glob('*'):
        f.unlink()
    work.rmdir()
    promotion = promotion_ok(considered, bc_test)
    promotion['edge_coverage_regression_ok'] = bool(
        considered.get('edge_coverage_mean', 0) >= MIN_COVERAGE_RATIO * bc_test.get('edge_coverage_mean', 0))
    promotion['promoted'] = bool(promotion['promoted'] and promotion['edge_coverage_regression_ok'])
    test = {'teacher': teacher_test, 'dagger_bc': bc_test, 'ppo_last': ppo_last_test}
    if ppo_best_test is not None:
        test[f'ppo_best_val_update_{best_update}'] = ppo_best_test
    result = {'experiment': experiment_name or 'wall_cycle_v0.8', 'recipe': 'v0.8',
              'evidence_level': 'L1', 'hardware_motion': False,
              'seed': seed, 'teacher_style': teacher_style, 'updates': updates,
              'ppo_steps': updates*pcfg.steps_per_update, 'workers': n_workers,
              'teacher_samples': len(O_demo), 'teacher_return_mean': float(np.mean(teacher_rets)),
              'behaviour_clone_mse': bc_mse, 'dagger': dagger_rows, 'value_warmup_mse': value_warmup,
              'config': cfg.to_dict(), 'ppo': pcfg.to_dict(),
              'score_mask': {'side_m': probe.material.metrics()['score_side_m'],
                             'cells': list(probe.material.score_cells),
                             'note': 'per-side margin rounded up; identical to the mask v0.7 used'},
              'test_set': {'seeds': [test_seed, test_seed+test_episodes-1], 'episodes': test_episodes},
              'validation_set': {'seeds': [val_seed, val_seed+val_episodes-1], 'episodes': val_episodes},
              'legacy_test_set': {'seeds': [legacy_seed, legacy_seed+test_episodes-1], 'episodes': test_episodes,
                                  'note': 'v0.7 test seeds; 20000-20007 were used to choose the v0.8 teacher, '
                                          'so this set is a historical regression check, not an independent test'},
              'test': test, 'validation_bc': bc_val, 'selected_checkpoint': selected,
              'selection_rule': f'PPO checkpoint eligible only if val coverage and edge coverage >= '
                                f'{ELIGIBLE_RATIO} x DAgger-BC val; then selection_score must beat BC',
              'promotion': promotion, 'test_cosim': cosim, 'test_legacy': legacy,
              'exploration': {'log_std_default': pcfg.init_log_std,
                              'explored_dims': [probe.action_names[i] for i in explore],
                              'explored_log_std': explore_log_std},
              'curve': curve, 'stage_elapsed_s': stage, 'elapsed_s': round(time.time()-t0, 1),
              'limitations': ['reduced-order 2.5-D mortar, not CFD; trowel coefficients assumed',
                              'D435 parameters provisional until physical calibration',
                              'MuJoCo servos/contact are simulated; parameters are not yet identified from hardware',
                              'training uses the reach table: carry loss is ~0 there but ~20 % in co-simulation',
                              'material pushed off the simulated grid is charged at 1/4 of the waste rate (user '
                              'decision 2026-09-24); outside_frac is reported so that exploiting it is visible',
                              'single training seed: an engineering run, not a stability result']}
    (out/'training.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


def selection_score(x):
    return 10*x['success_rate'] + x['coverage_mean'] - .2*x['rmse_mm_mean'] - 2*x['waste_frac_mean']


def main():
    p=argparse.ArgumentParser(description='Whole-wall plastering: teacher -> BC -> DAgger -> PPO (v0.6 physics, real-arm executor).')
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
    p.add_argument('--recipe',choices=('v0.7','v0.8'),default='v0.8',
                   help='v0.6-physics experiment recipe (recipes.py); v0.7 reproduces the 2026-09-23 run')
    p.add_argument('--workers',type=int,default=0,
                   help='v0.8 only: worker processes (0 = all logical cores but one; 1 = no subprocesses)')
    p.add_argument('--dagger-rounds',type=int,default=4)
    p.add_argument('--dagger-episodes',type=int,default=40)
    p.add_argument('--no-arm',action='store_true',help='skip the MuJoCo reachability executor (faster, less faithful)')
    a=p.parse_args()
    v6=a.physics=='v0.6'
    if v6:
        from .recipes import make_config
        cfg=make_config(a.recipe, use_arm=not a.no_arm)
        name={'v0.7':'wall_cycle_v0.7_p2','v0.8':'wall_cycle_v0.8'}[a.recipe]
        recipe=a.recipe
    else:
        cfg=CycleConfig(use_arm=not a.no_arm, physics=a.physics, tool_profile='lab_20260922',
                        lift_wall_fraction=.75, base_steps=6000, max_steps=12000, extension_steps=1000,
                        max_cycles=80, max_reload_cycles=30, stall_limit=5, stall_window=5,
                        min_cycles_before_stall=0)
        name='wall_cycle_v0.5_p1'; recipe=None
    r=train(a.out,a.updates,a.seed,a.teacher_episodes,cfg=cfg,teacher_style=a.teacher_style,
            steps_per_update=a.steps_per_update, test_episodes=a.test_episodes,
            val_episodes=a.val_episodes, cosim_test_episodes=a.cosim_test_episodes,
            eval_every=a.eval_every, hidden=(256,256), experiment_name=name,
            dagger_rounds=a.dagger_rounds,dagger_episodes=a.dagger_episodes,
            recipe=recipe, workers=a.workers,
            log=lambda x: print(x, flush=True))
    print(json.dumps({'selected':r['selected_checkpoint'],'promotion':r['promotion'],
                      'test':r['test']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
