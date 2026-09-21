"""最小 PPO（纯 numpy，不依赖 torch）。

为什么自己写：这一步只要验证「奖励设计 + 环境」能不能让机械臂学到抹涂手法，
策略很小（7 维观测、3 维动作、两层 64），CPU 上跑得比装训练框架还快，
也不给 `requirements/core.txt` 增加依赖。以后换 ACT / Diffusion Policy 再上 torch。

实现的是标准 PPO-clip：高斯策略（对角、状态无关的 log_std）、GAE(λ)、
价值函数裁剪、观测归一化、Adam。参数都在 PPOConfig 里。
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class PPOConfig:
    hidden: tuple = (64, 64)
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 256
    steps_per_update: int = 2048
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    init_log_std: float = -1.0
    seed: int = 0

    def to_dict(self):
        d = asdict(self); d['hidden'] = list(self.hidden); return d


def _init(rng, shape, scale):
    return rng.normal(0, scale, shape) / np.sqrt(shape[0])


class MLP:
    """tanh 隐层的多层感知机，手写前向与反向。"""

    def __init__(self, rng, sizes, out_scale=0.01):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = out_scale if i == len(sizes) - 2 else 1.0
            self.W.append(_init(rng, (sizes[i], sizes[i + 1]), scale))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, x):
        self.cache = [x]
        h = x
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i] + self.b[i])
            self.cache.append(h)
        return h @ self.W[-1] + self.b[-1]

    def backward(self, dout):
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        h = self.cache[-1]
        gW[-1] = h.T @ dout; gb[-1] = dout.sum(0)
        d = dout @ self.W[-1].T
        for i in range(len(self.W) - 2, -1, -1):
            d = d * (1 - self.cache[i + 1] ** 2)
            gW[i] = self.cache[i].T @ d; gb[i] = d.sum(0)
            d = d @ self.W[i].T
        return gW, gb

    @property
    def params(self):
        return self.W + self.b

    def load(self, arrays):
        n = len(self.W)
        self.W = [np.array(a) for a in arrays[:n]]; self.b = [np.array(a) for a in arrays[n:]]


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, max_norm=None):
        self.t += 1
        if max_norm:
            total = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads))
            if total > max_norm:
                grads = [g * (max_norm / (total + 1e-12)) for g in grads]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim); self.var = np.ones(dim); self.count = 1e-4
        self.frozen = False

    def update(self, x):
        if self.frozen:      # 冻结后不再变：否则价值网络的输入分布一直漂移，优势估计会失真
            return
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean; tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


class PPO:
    def __init__(self, obs_dim, act_dim, cfg: PPOConfig = None):
        self.cfg = cfg or PPOConfig()
        rng = np.random.default_rng(self.cfg.seed)
        self.rng = rng
        self.pi = MLP(rng, [obs_dim, *self.cfg.hidden, act_dim])
        self.vf = MLP(rng, [obs_dim, *self.cfg.hidden, 1], out_scale=1.0)
        self.log_std = np.full(act_dim, self.cfg.init_log_std)
        self.norm = RunningNorm(obs_dim)
        self.opt_pi = Adam(self.pi.params + [self.log_std], self.cfg.lr)
        self.opt_vf = Adam(self.vf.params, self.cfg.lr)
        self.act_dim = act_dim

    # ------------------------------------------------------------------ 策略
    def act(self, obs, deterministic=False):
        x = self.norm(np.atleast_2d(obs))
        mu = self.pi.forward(x)[0]
        if deterministic:
            return np.clip(mu, -1, 1), mu, 0.0
        std = np.exp(self.log_std)
        raw = mu + std * self.rng.normal(size=self.act_dim)
        return np.clip(raw, -1, 1), raw, self._logp(mu, raw)

    def _logp(self, mu, raw):
        std = np.exp(self.log_std)
        return float(np.sum(-0.5 * ((raw - mu) / std) ** 2 - self.log_std - 0.5 * np.log(2 * np.pi)))

    def value(self, obs):
        return float(self.vf.forward(self.norm(np.atleast_2d(obs)))[0, 0])

    # ------------------------------------------------------------------ 更新
    def update(self, obs, raw_actions, logp_old, adv, ret):
        c = self.cfg
        self.norm.update(obs)
        x = self.norm(obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(obs); idx = np.arange(n)
        stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'kl': 0.0}
        for _ in range(c.epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, c.minibatch):
                b = idx[start:start + c.minibatch]
                xb, ab, lb, advb, retb = x[b], raw_actions[b], logp_old[b], adv[b], ret[b]
                mu = self.pi.forward(xb)
                std = np.exp(self.log_std)
                logp = np.sum(-0.5 * ((ab - mu) / std) ** 2 - self.log_std - 0.5 * np.log(2 * np.pi), axis=1)
                ratio = np.exp(np.clip(logp - lb, -20, 20))
                clipped = np.clip(ratio, 1 - c.clip, 1 + c.clip)
                use_unclipped = (ratio * advb) <= (clipped * advb)
                # d/dmu of -mean(min(r*A, clip(r)*A))
                coef = np.where(use_unclipped, ratio * advb, 0.0)
                dlogp = -coef[:, None] / len(b)
                dmu = dlogp * (ab - mu) / std ** 2
                gW, gb = self.pi.backward(dmu)
                dlog_std = np.sum(dlogp * (((ab - mu) / std) ** 2 - 1.0), axis=0) - c.entropy_coef
                self.opt_pi.step(self.pi.params + [self.log_std], gW + gb + [dlog_std], c.max_grad_norm)
                v = self.vf.forward(xb)[:, 0]
                dv = (2.0 * (v - retb) / len(b) * c.value_coef)[:, None]
                gWv, gbv = self.vf.backward(dv)
                self.opt_vf.step(self.vf.params, gWv + gbv, c.max_grad_norm)
                stats['policy_loss'] += float(-np.mean(np.minimum(ratio * advb, clipped * advb)))
                stats['value_loss'] += float(np.mean((v - retb) ** 2))
                stats['kl'] += float(np.mean(lb - logp))
        k = max(1, c.epochs * int(np.ceil(n / c.minibatch)))
        return {key: round(val / k, 5) for key, val in stats.items()}

    def fit_value(self, obs, ret, epochs=30):
        """只训练价值函数。预热用：价值函数没练过时优势估计噪声极大，第一次策略更新会把好策略打坏。"""
        c = self.cfg
        self.norm.update(obs); x = self.norm(obs)
        n = len(obs); idx = np.arange(n)
        for _ in range(epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, c.minibatch):
                b = idx[start:start + c.minibatch]
                v = self.vf.forward(x[b])[:, 0]
                dv = (2.0 * (v - ret[b]) / len(b))[:, None]
                gW, gb = self.vf.backward(dv)
                self.opt_vf.step(self.vf.params, gW + gb, c.max_grad_norm)
        return float(np.mean((self.vf.forward(x)[:, 0] - ret) ** 2))

    # ------------------------------------------------------------------ 存取
    def save(self, path):
        arrays = {f'pi_{i}': p for i, p in enumerate(self.pi.params)}
        arrays.update({f'vf_{i}': p for i, p in enumerate(self.vf.params)})
        arrays.update({'log_std': self.log_std, 'norm_mean': self.norm.mean, 'norm_var': self.norm.var,
                       'norm_count': np.array([self.norm.count])})
        np.savez(path, **arrays)

    def load(self, path):
        d = np.load(path)
        self.pi.load([d[f'pi_{i}'] for i in range(len(self.pi.params))])
        self.vf.load([d[f'vf_{i}'] for i in range(len(self.vf.params))])
        self.log_std = d['log_std']; self.norm.mean = d['norm_mean']
        self.norm.var = d['norm_var']; self.norm.count = float(d['norm_count'][0])
        self.norm.frozen = True          # 载入已训练策略后观测归一化不再更新
        return self


def gae(rewards, values, dones, last_value, gamma, lam):
    adv = np.zeros(len(rewards)); run = 0.0; nxt = last_value
    for t in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * nxt * nonterminal - values[t]
        run = delta + gamma * lam * nonterminal * run
        adv[t] = run; nxt = values[t]
    return adv, adv + values
