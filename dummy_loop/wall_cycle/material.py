"""Conservative 2.5-D wall and trowel material fields.

This is the v0.2 material model, kept byte-for-byte in behaviour so that
experiments/v0.2/r0/records/2026-09-22_wall_cycle_rl stays reproducible (CycleConfig(physics='v0.2')).
v0.3 replaces the stroke physics with dummy_loop/wall_cycle/mortar.py (see
experiments/v0.3/r0/design/2026-09-22_wall_cycle_v0.3.md for why each rule here was replaced).

This is deliberately a reduced-order proxy, not CFD. Unlike the first experiment,
material on the trowel has a spatial distribution and can only appear on the wall
after it is removed from a corresponding trowel cell.
"""
from dataclasses import dataclass
import math
import numpy as np

from .config import CycleConfig


@dataclass
class TransferStats:
    deposited_m3: float = 0.0
    scraped_m3: float = 0.0
    dropped_m3: float = 0.0
    outside_m3: float = 0.0
    peak_force_N: float = 0.0


class MaterialSystem:
    def __init__(self, cfg: CycleConfig, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        nv, nu = cfg.wall_shape
        bz, bx = cfg.blade_shape
        self.u = np.linspace(-cfg.width_m / 2 + cfg.cell_m / 2,
                             cfg.width_m / 2 - cfg.cell_m / 2, nu)
        self.v = np.linspace(cfg.cell_m / 2, cfg.height_m - cfg.cell_m / 2, nv)
        self.wall = np.zeros((nv, nu), float)
        self.wall_age = np.zeros_like(self.wall)
        self.blade = np.zeros((bz, bx), float)
        self.blade_age = np.zeros_like(self.blade)
        self.supplied_m3 = 0.0
        self.dropped_m3 = 0.0
        self.outside_m3 = 0.0
        self.initial_m3 = 0.0
        # v0.7: scored sub-window inside the (possibly larger) simulation grid. metrics() only
        # looks here; overtravel into the margin outside it is not counted as waste (see
        # config.py: score_width_m/score_height_m, area.py: load_work_area()). Concentric crop,
        # so if score==sim (v0.2/pre-v0.7 configs) this is a no-op full-grid slice.
        score_w = getattr(cfg, 'score_width_m', cfg.width_m)
        score_h = getattr(cfg, 'score_height_m', cfg.height_m)
        # v0.8: the per-side margin is rounded UP explicitly, so the scored square never exceeds
        # the nominal one. v0.7 used round() on 2.5000000000000018 cells, which also gave 3 --
        # i.e. v0.7 already scored 40x40 cells = 200 mm, not the nominal 205 mm. This keeps
        # exactly that mask and makes it explicit (score_cells / score_side_m in metrics).
        du = max(0, int(math.ceil((cfg.width_m - score_w) / 2 / cfg.cell_m - 1e-6)))
        dv = max(0, int(math.ceil((cfg.height_m - score_h) / 2 / cfg.cell_m - 1e-6)))
        self._score_rows = slice(dv, nv - dv)
        self._score_cols = slice(du, nu - du)
        self.score_cells = (nv - 2 * dv, nu - 2 * du)
        # Edge band of the scored square (reported only; not part of reward or gates).
        band = max(1, int(round(getattr(cfg, 'edge_band_m', 0.03) / cfg.cell_m)))
        ii, jj = np.indices(self.score_cells)
        dist = np.minimum.reduce([ii, jj, self.score_cells[0] - 1 - ii, self.score_cells[1] - 1 - jj])
        self._edge_mask = dist < band

    @property
    def wall_cell_area(self):
        return self.cfg.cell_m ** 2

    @property
    def blade_cell_area(self):
        return self.cfg.blade_cell_m ** 2

    @property
    def blade_volume_m3(self):
        return float(self.blade.sum() * self.blade_cell_area)

    def reset(self, initial='bare'):
        self.wall.fill(0); self.wall_age.fill(0)
        self.blade.fill(0); self.blade_age.fill(0)
        self.supplied_m3 = self.dropped_m3 = self.outside_m3 = 0.0
        if initial == 'partial':
            # Random, smooth-ish prior work: the policy must decide whether to add or level.
            for _ in range(int(self.rng.integers(2, 6))):
                u0 = self.rng.uniform(self.u.min(), self.u.max())
                v0 = self.rng.uniform(self.v.min(), self.v.max())
                su = self.rng.uniform(.018, .055); sv = self.rng.uniform(.010, .030)
                amp = self.rng.uniform(.0005, .0045)
                U, V = np.meshgrid(self.u, self.v)
                self.wall += amp * np.exp(-.5 * ((U-u0)/su)**2 - .5 * ((V-v0)/sv)**2)
            self.wall = np.clip(self.wall, 0, .006)
        self.initial_m3 = float(self.wall.sum()*self.wall_cell_area)
        return self

    def load_random(self, requested_ml):
        """Add 1-3 bounded Gaussian blobs, preserving existing residue."""
        c = self.cfg
        scale = self.rng.normal(1.0, .10)
        scale = float(np.clip(scale, *c.load_scale_range))
        wanted = requested_ml * scale * 1e-6
        z, x = np.indices(self.blade.shape)
        shape = np.zeros_like(self.blade)
        for _ in range(int(self.rng.integers(1, 4))):
            x0 = self.rng.uniform(.2, .8) * (self.blade.shape[1] - 1)
            z0 = self.rng.uniform(.2, .8) * (self.blade.shape[0] - 1)
            sx = self.rng.uniform(3., 7.); sz = self.rng.uniform(.8, 2.0)
            shape += self.rng.uniform(.7, 1.3) * np.exp(-.5*((x-x0)/sx)**2 - .5*((z-z0)/sz)**2)
        if shape.sum() > 0:
            shape *= wanted / (shape.sum() * self.blade_cell_area)
        self.blade += shape
        cap = c.blade_capacity_ml * 1e-6
        total = self.blade_volume_m3
        overflow = max(total - cap, 0.0)
        if overflow:
            self.blade *= cap / total
            self.dropped_m3 += overflow
        actual = wanted - overflow
        self.supplied_m3 += wanted
        return {'requested_ml': requested_ml, 'delivered_ml': wanted * 1e6,
                'retained_ml': actual * 1e6, 'overflow_ml': overflow * 1e6}

    def _blade_coordinates(self, center, phi):
        """World-wall coordinates for every blade material cell."""
        bz, bx = self.blade.shape
        along = (np.arange(bx) - (bx - 1) / 2) * self.cfg.blade_cell_m
        across = (np.arange(bz) - (bz - 1) / 2) * self.cfg.blade_cell_m
        X, Z = np.meshgrid(along, across)
        ca, sa = np.cos(phi), np.sin(phi)
        return center[0] + X*ca - Z*sa, center[1] + X*sa + Z*ca

    def _indices(self, U, V):
        iu = np.floor((U + self.cfg.width_m / 2) / self.cfg.cell_m).astype(int)
        iv = np.floor(V / self.cfg.cell_m).astype(int)
        good = ((iu >= 0) & (iu < self.wall.shape[1]) &
                (iv >= 0) & (iv < self.wall.shape[0]))
        return iv, iu, good

    def stroke(self, mode, start, end, phi, bend, force_N, speed_m_s, callback=None):
        """v0.2 stroke (hand-written transfer fractions). Kept unchanged for reproducing
        experiments/v0.2/r0/records/2026-09-22_wall_cycle_rl. v0.3 uses mortar.MortarSystem.stroke."""
        c = self.cfg
        stats = TransferStats()
        p0, p2 = np.asarray(start, float), np.asarray(end, float)
        d = p2 - p0; norm = np.linalg.norm(d)
        if norm < 1e-9:
            return stats
        normal = np.array([-d[1], d[0]]) / norm
        pc = (p0 + p2) / 2 + normal * bend
        transfer = np.clip(.06 + .015 * force_N - .20 * speed_m_s, .03, .32)
        level_strength = np.clip(.08 + .02 * force_N, .10, .42)
        for k, t in enumerate(np.linspace(0, 1, c.stroke_samples)):
            center = (1-t)**2*p0 + 2*(1-t)*t*pc + t**2*p2
            U, V = self._blade_coordinates(center, phi)
            iv, iu, good = self._indices(U, V)
            # Multiple blade cells can fall in one wall cell.  Split that cell's
            # demand between them so vectorisation preserves both volume and bounds.
            outside = ~good
            if mode != 'LEVEL' and np.any(outside):
                lost = self.blade[outside] * self.blade_cell_area * transfer * .2
                self.blade[outside] -= lost / self.blade_cell_area
                amount = float(lost.sum())
                self.outside_m3 += amount; stats.outside_m3 += amount
            if np.any(good):
                br, bc = np.nonzero(good)
                wr, wc = iv[good], iu[good]
                flat = wr*self.wall.shape[1] + wc
                counts = np.bincount(flat, minlength=self.wall.size)[flat]
                if mode == 'LEVEL':
                    excess = np.maximum(self.wall[wr, wc]-c.target_m, 0) * level_strength/counts
                    vol = excess*self.wall_cell_area
                    np.add.at(self.wall.ravel(), flat, -excess)
                    self.blade[br, bc] += vol/self.blade_cell_area
                    stats.scraped_m3 += float(vol.sum())
                else:
                    deficit = np.maximum(c.acceptable_high_m-self.wall[wr, wc], 0)/counts
                    want = deficit*self.wall_cell_area*transfer
                    available = self.blade[br, bc]*self.blade_cell_area
                    vol = np.minimum(want, available)
                    np.add.at(self.wall.ravel(), flat, vol/self.wall_cell_area)
                    self.blade[br, bc] -= vol/self.blade_cell_area
                    stats.deposited_m3 += float(vol.sum())
                    high = np.maximum(self.wall[wr, wc]-.0035, 0)*level_strength*.25/counts
                    scraped = high*self.wall_cell_area
                    np.add.at(self.wall.ravel(), flat, -high)
                    self.blade[br, bc] += scraped/self.blade_cell_area
                    stats.scraped_m3 += float(scraped.sum())
            # A small gravity/pressure redistribution on the trowel.
            self.blade = .92*self.blade + .02*(np.roll(self.blade, 1, 0) +
                                                np.roll(self.blade, -1, 0) +
                                                np.roll(self.blade, 1, 1) +
                                                np.roll(self.blade, -1, 1))
            capacity_cell = .018
            spill = np.maximum(self.blade - capacity_cell, 0)
            spill_v = float(spill.sum() * self.blade_cell_area)
            self.blade -= spill
            self.dropped_m3 += spill_v; stats.dropped_m3 += spill_v
            stats.peak_force_N = max(stats.peak_force_N,
                                     force_N + 1800 * stats.scraped_m3)
            if callback is not None:
                callback(k, center.copy(), self.wall.copy(), self.blade.copy())
        self.wall_age += norm / max(speed_m_s, 1e-4)
        self.blade_age += norm / max(speed_m_s, 1e-4)
        return stats

    def metrics(self):
        c = self.cfg; h = self.wall[self._score_rows, self._score_cols]
        err = np.abs(h - c.target_m)
        acceptable = (h >= c.acceptable_low_m) & (h <= c.acceptable_high_m)
        rough_u = np.abs(np.diff(h, axis=1)).mean() if h.shape[1] > 1 else 0
        rough_v = np.abs(np.diff(h, axis=0)).mean() if h.shape[0] > 1 else 0
        return {'coverage': float(acceptable.mean()),
                'mean_thickness_mm': float(h.mean()*1000),
                'rmse_mm': float(np.sqrt(np.mean((h-c.target_m)**2))*1000),
                'p95_error_mm': float(np.percentile(err, 95)*1000),
                'under_frac': float(np.mean(h < c.acceptable_low_m)),
                'over_frac': float(np.mean(h > c.acceptable_high_m)),
                'roughness_mm': float((rough_u+rough_v)*500),
                'blade_load_ml': self.blade_volume_m3*1e6,
                'waste_frac': (self.dropped_m3+self.outside_m3)/max(self.supplied_m3+self.initial_m3, 1e-12),
                'edge_coverage': float(acceptable[self._edge_mask].mean()),
                'edge_bare_frac': float(np.mean(h[self._edge_mask] < 0.0005)),
                'score_side_m': float(self.score_cells[1] * c.cell_m)}

    def quality_cost(self):
        m = self.metrics(); c = self.cfg
        return (0.45*min(m['rmse_mm']/(c.target_m*1000), 2) +
                0.20*m['under_frac'] + 0.20*m['over_frac'] +
                0.10*min(m['roughness_mm']/2, 1) +
                0.05*min(m['p95_error_mm']/4, 1))

    def volume_balance(self):
        return (float(self.wall.sum()*self.wall_cell_area) + self.blade_volume_m3 +
                self.dropped_m3 + self.outside_m3 - self.supplied_m3 - self.initial_m3)
