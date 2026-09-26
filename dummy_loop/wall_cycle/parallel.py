"""Multi-process execution for wall-cycle training and evaluation (v0.8).

Why: one decision costs ~64 ms of mortar physics (table executor) or ~2.5 s (MuJoCo
co-simulation), while the policy network costs < 1 ms. Training was a single Python process,
so it used one logical core and no GPU. Episodes are independent, so they are spread over
worker processes:

- per-episode jobs (teacher demonstrations, DAgger, every evaluation) go through a process
  pool; each episode has a fixed seed and a fresh environment, so the result of an episode
  does not depend on which worker ran it or on the worker count;
- PPO sampling uses persistent rollout workers, each owning one environment that continues
  across updates (like a vectorised env). Every worker gets its own sampling seed, so the
  samples depend on the worker count (deterministic for a fixed count).

Workers use the 'spawn' start method on every OS (Windows has no fork) and are limited to one
BLAS/OpenMP thread each, so N workers use N cores instead of fighting over them. The policy
is handed over as an .npz file path; a worker reloads it only when the path changes.
"""
import multiprocessing as mp
import os

import numpy as np

_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def resolve_workers(workers):
    """0 or None -> all logical cores but one."""
    if not workers:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(workers))


def _single_thread_env():
    for k in _THREAD_VARS:
        os.environ[k] = '1'


# ----------------------------------------------------------------------------- policy cache
_POLICY = {}


def load_policy(path, obs_dim, act_dim, hidden):
    if path is None:
        return None
    key = (str(path), os.path.getmtime(path), tuple(hidden))
    if key not in _POLICY:
        from ..wall.ppo import PPO, PPOConfig
        _POLICY.clear()
        _POLICY[key] = PPO(obs_dim, act_dim, PPOConfig(hidden=tuple(hidden))).load(path)
    return _POLICY[key]


# ----------------------------------------------------------------------------- episode jobs
def teacher_episode(job):
    """(cfg, seed, teacher_style) -> (obs[T,d], actions[T,a], return)."""
    from .train import make_env
    cfg, seed, style = job
    env = make_env(cfg, seed, style)
    o = env.reset(); done = False; ret = 0.0; O, A = [], []
    while not done:
        a = env.teacher_action(); O.append(o); A.append(a)
        o, r, done, _ = env.step(a); ret += r
    return np.asarray(O), np.asarray(A), ret


def dagger_episode(job):
    """(cfg, policy_path, hidden, seed, rng_seed, learner_probability, style) -> (obs, labels).
    The learner/teacher coin flips use a per-episode stream, so a DAgger episode is the same
    whichever worker runs it."""
    from .train import make_env
    cfg, path, hidden, seed, rng_seed, p_learner, style = job
    env = make_env(cfg, seed, style)
    agent = load_policy(path, env.obs_dim, env.act_dim, hidden)
    rng = np.random.default_rng(rng_seed)
    o = env.reset(); done = False; O, Y = [], []
    while not done:
        teacher = env.teacher_action(); O.append(o); Y.append(teacher)
        learned = agent.act(o, deterministic=True)[0]
        o, _, done, _ = env.step(learned if rng.random() < p_learner else teacher)
    return np.asarray(O), np.asarray(Y)


def eval_episode(job):
    """(cfg, policy_path|None, hidden, seed, style, executor) -> per-episode record."""
    from .train import make_env, run_eval_episode
    cfg, path, hidden, seed, style, executor = job
    env = make_env(cfg, seed, style, executor)
    agent = load_policy(path, env.obs_dim, env.act_dim, hidden)
    return run_eval_episode(env, agent, executor)


class Runner:
    """Pool for per-episode jobs; workers == 1 runs them inline (same functions)."""

    def __init__(self, workers=1):
        self.workers = resolve_workers(workers)
        self._pool = None

    def map(self, fn, jobs):
        jobs = list(jobs)
        if self.workers == 1 or len(jobs) <= 1:
            return [fn(j) for j in jobs]
        if self._pool is None:
            saved = {k: os.environ.get(k) for k in _THREAD_VARS}
            _single_thread_env()              # inherited by the spawned children
            try:
                self._pool = mp.get_context('spawn').Pool(self.workers)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        return self._pool.map(fn, jobs, chunksize=1)

    def close(self):
        if self._pool is not None:
            self._pool.close(); self._pool.join(); self._pool = None


# ----------------------------------------------------------------------------- PPO sampling
class _RolloutState:
    """One environment that keeps running across PPO updates."""

    def __init__(self, cfg, env_seed, style):
        from .train import make_env
        self.env = make_env(cfg, env_seed, style)
        self.obs = self.env.reset(); self.ep_ret = 0.0

    def collect(self, path, hidden, n, rng_seed):
        agent = load_policy(path, self.env.obs_dim, self.env.act_dim, hidden)
        agent.rng = np.random.default_rng(rng_seed)
        O, A, L, R, D, finished = [], [], [], [], [], []
        for _ in range(n):
            a, raw, lp = agent.act(self.obs)
            nxt, r, done, _ = self.env.step(a)
            O.append(self.obs); A.append(raw); L.append(lp); R.append(r); D.append(done)
            self.ep_ret += r; self.obs = nxt
            if done:
                finished.append(self.ep_ret); self.ep_ret = 0.0; self.obs = self.env.reset()
        return (np.asarray(O), np.asarray(A), np.asarray(L), np.asarray(R, float),
                np.asarray(D, bool), self.obs.copy(), finished)


def _rollout_worker(conn, cfg, env_seed, style):
    state = _RolloutState(cfg, env_seed, style)
    while True:
        msg = conn.recv()
        if msg[0] == 'close':
            conn.close(); return
        try:
            conn.send(('ok', state.collect(*msg[1:])))
        except Exception as e:                      # report instead of hanging the trainer
            import traceback
            conn.send(('error', f'{e!r}\n{traceback.format_exc()}'))


class RolloutPool:
    """n persistent samplers. collect() splits n_steps over them and returns one segment per
    worker (a contiguous trajectory piece + its last observation) so GAE can bootstrap each
    segment separately."""

    def __init__(self, cfg, workers, seed, style):
        self.workers = resolve_workers(workers)
        self.seed = seed
        env_seeds = [seed + 5000 + 101 * w for w in range(self.workers)]
        if self.workers == 1:
            self._inline = _RolloutState(cfg, env_seeds[0], style); self._procs = []
            return
        self._inline = None
        saved = {k: os.environ.get(k) for k in _THREAD_VARS}
        _single_thread_env()
        ctx = mp.get_context('spawn'); self._conns, self._procs = [], []
        try:
            for w in range(self.workers):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_rollout_worker, args=(child, cfg, env_seeds[w], style), daemon=True)
                p.start(); child.close()
                self._conns.append(parent); self._procs.append(p)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def collect(self, path, hidden, n_steps, call_index):
        per = [n_steps // self.workers + (1 if w < n_steps % self.workers else 0) for w in range(self.workers)]
        seeds = [(self.seed, 7, call_index, w) for w in range(self.workers)]
        if self._inline is not None:
            return [self._inline.collect(path, hidden, per[0], seeds[0])]
        for conn, n, s in zip(self._conns, per, seeds):
            conn.send(('collect', path, hidden, n, s))
        out = []
        for conn in self._conns:
            status, payload = conn.recv()
            if status != 'ok':
                raise RuntimeError(f'rollout worker failed:\n{payload}')
            out.append(payload)
        return out

    def close(self):
        for conn in getattr(self, '_conns', []):
            try:
                conn.send(('close',))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=10)
