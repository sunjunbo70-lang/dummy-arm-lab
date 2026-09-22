"""Physics-based reduced-order mortar model (v0.3, "plan B").

WHY THIS FILE EXISTS
    v0.2 moved mortar with hand-written transfer fractions (more force -> more deposit,
    a fixed 3.5 mm scrape line, np.roll smoothing). An intermediate draft of v0.3 added a
    "wedge capacity" rule taken from the worker's explanation of the tilt technique. That
    rule would have put the answer into the simulator: the policy would "learn" to tilt
    only because we had written that tilting holds material. The user asked for the
    opposite: simulate the material honestly and let the policy find the angle. So this
    model contains NO rule that mentions pitch. Pitch only enters through geometry
    (the gap under the blade and the direction of gravity on the blade face).

WHAT IS MODELLED (Bingham / yield-stress material, quasi-static per stroke sample)
    Mortar is a yield-stress fluid: below tau_y it holds like a solid, above it flows.
    Parameters are physical quantities, not behaviour rules:
        rho   density                (1900 kg/m^3)
        tau_y yield stress           (250 Pa; Banfill 2003 lists ~400 Pa for mortar,
                                      plasters are softer; swept 100-500 Pa in tests)
        mu_p  plastic viscosity      (2 Pa s; Banfill 2003: 1-3 Pa s for mortar)
        lift_wall_fraction           (0.5; how a squeezed layer splits when the blade
                                      is pulled straight off -- ASSUMED, to be measured)

    1. Squeeze force balance -> gap. A blade pressed with normal force F rides on the
       mortar under it. For a yield-stress layer squeezed between plates at gap g the
       pressure rises from 0 at a free edge with slope 2*tau_y/g (perfectly plastic
       squeeze-flow limit). Integrating over the wetted part of the blade gives the
       supporting force S(g0), decreasing in the trailing-edge gap g0. We solve
       S(g0) = F. More force -> thinner layer (the v0.2 sign was reversed). If the
       mortar cannot carry F even at a vanishing gap, the steel touches the wall (g0=0)
       and the rigid contact carries the rest (that part is MuJoCo's job).
    2. Screeding. Wall cells the leading edge reaches are absorbed into the mortar column
       under the blade. Inside the gap the wall slides backwards relative to the blade and
       shears the mortar with it; the mean transport speed of a sheared layer is half the
       wall speed (Couette), so the column content advects towards the trailing edge by
       couette_fraction x the blade displacement per sample (0.5, the Newtonian value;
       a yield-stress plug can stick to either surface -- ASSUMED). Wall cells the trailing
       edge leaves keep min(column, local gap). Filling lows and cutting highs are therefore
       one mechanism (no separate DEPOSIT / LEVEL physics any more).
    3. Extrusion. Where a column holds more than the gap, the excess is squeezed out.
       Thin-film flow rate scales with gap^3, so the excess splits between the trailing
       and the leading edge in proportion to g_trail^3 : g_lead^3. A flat blade splits
       it evenly (half of it leaves a ridge BEHIND the stroke); a tilted blade sends it
       ahead. What leaves the leading edge is a bead the blade pushes along (it rests
       against the moving blade, so it is not subject to wall slump); it re-enters the
       gap at the leading edge next sample and is put on the wall only when the blade
       lifts off. This is where tilt can pay off -- or not -- as a consequence, not a rule.
    4. Gravity drainage on the blade face. With the face inclined so that gravity has a
       component |g_t| along it, a layer is stable up to h_y = tau_y/(rho |g_t|). Excess
       drains downhill (Bingham film flux at these thicknesses empties a 5 mm cell in
       well under one sample, so drainage is treated as immediate). At a downhill edge
       that is sealed against the wall the drained material is caught; at a free edge it
       falls (dropped). If the face overhangs (gravity pulls material off the face) a
       layer whose weight per area exceeds tau_y peels off.
    5. Wall slump. A layer on the vertical wall thicker than tau_y/(rho g) sloughs off.

    Volume is conserved exactly: wall + blade + dropped + outside = supplied + initial.

NOT MODELLED (state it, do not hide it): thixotropy and setting over time, water
absorption by the substrate, adhesion strength different from cohesion, 3-D flow along
the blade length (the width direction is the short escape path and dominates), surface
tension, and the blade's own edges as a sharp scraper at high speed. Evidence level L1.
"""
from dataclasses import dataclass
import numpy as np

G = 9.81


@dataclass
class MortarParams:
    rho: float = 1900.0
    tau_y: float = 250.0
    mu_p: float = 2.0
    lift_wall_fraction: float = 0.5
    couette_fraction: float = 0.5
    max_gap_m: float = 0.012          # bisection bracket for the force balance
    min_gap_m: float = 1e-5

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class SampleResult:
    gap_m: float = 0.0
    support_N: float = 0.0            # normal force carried by mortar (rest = steel on wall)
    drag_N: float = 0.0
    deposited_m3: float = 0.0         # left on the wall behind the blade this sample
    absorbed_m3: float = 0.0          # picked up from the wall ahead
    extruded_trail_m3: float = 0.0
    extruded_lead_m3: float = 0.0
    dropped_m3: float = 0.0
    outside_m3: float = 0.0
    wetted_frac: float = 0.0


def stable_thickness(p: MortarParams, g_tangential):
    """Largest layer a surface can hold before the yield stress is exceeded.
    g_tangential: gravity component along the surface in units of g (1 = vertical face)."""
    return np.inf if g_tangential < 1e-9 else p.tau_y / (p.rho * G * g_tangential)


def squeeze_support(t, gap, cell, tau_y):
    """Normal force carried by columns t (rows = across width from the trailing edge,
    cols = along the blade) at local gaps gap[row]. Plastic squeeze: dp/ds = 2 tau_y / g,
    zero pressure at the free edges of each wetted run across the width.

    Vectorised over columns: for every cell, the distance to the start / end of its wetted
    run is counted row by row (only nw = 6 rows), then p = 2 tau_y d / g."""
    wet = t >= gap[:, None] - 1e-12
    nw = t.shape[0]
    up = np.zeros(t.shape); dn = np.zeros(t.shape)
    run = np.zeros(t.shape[1])
    for r in range(nw):
        run = (run + 1) * wet[r]; up[r] = run
    run = np.zeros(t.shape[1])
    for r in range(nw - 1, -1, -1):
        run = (run + 1) * wet[r]; dn[r] = run
    d = (np.minimum(up, dn) - 0.5) * cell            # centre of the cell to the nearest run edge
    p = np.where(wet, 2 * tau_y * np.maximum(d, 0) / gap[:, None], 0.0)
    return float(p.sum()) * cell * cell


def solve_gap(t, pitch, force_N, cell, p: MortarParams):
    """Trailing-edge gap g0 such that the mortar carries force_N (bisection)."""
    nw = t.shape[0]
    rise = (np.arange(nw) + 0.5) * cell * np.sin(max(pitch, 0.0))
    def support(g0):
        return squeeze_support(t, g0 + rise, cell, p.tau_y)
    if force_N <= 0:
        return p.max_gap_m, 0.0
    if support(p.min_gap_m) < force_N:          # mortar cannot carry it: steel on the wall
        return 0.0, support(p.min_gap_m)
    lo, hi = p.min_gap_m, p.max_gap_m
    if support(hi) >= force_N:
        return hi, support(hi)
    for _ in range(20):                      # 12 mm / 2^20 ~ 0.01 um: far below anything that matters
        mid = 0.5 * (lo + hi)
        if support(mid) >= force_N:
            lo = mid
        else:
            hi = mid
    return lo, support(lo)


def extrude(t, gap):
    """Squeeze columns down to the local gap. Excess first fills unfilled wedge space
    (towards larger gap), the rest leaves through the two width edges in proportion to
    edge_gap^3. Returns new t and per-column (trail, lead) volumes-per-area."""
    t = t.copy()
    excess = np.maximum(t - gap[:, None], 0.0)
    t -= excess
    E = excess.sum(axis=0)                                    # per column, m (x cell area later)
    room = np.maximum(gap[:, None] - t, 0.0)
    # fill remaining room from the trailing side forward (material is pushed into the wedge)
    for r in range(t.shape[0]):
        take = np.minimum(room[r], E)
        t[r] += take; E = E - take
    g_t, g_l = max(gap[0], 1e-9), max(gap[-1], 1e-9)
    f_trail = g_t ** 3 / (g_t ** 3 + g_l ** 3)
    return t, E * f_trail, E * (1 - f_trail)


def _cascade(a, downhill_up, share, h_stable):
    """Move material above h_stable along axis 0 of `a` (rows = positions). downhill_up:
    True if downhill is increasing row index. Returns new a and what leaves the last row."""
    a = a.copy()
    rows = range(a.shape[0]) if downhill_up else range(a.shape[0] - 1, -1, -1)
    carry = np.zeros(a.shape[1])
    for i in rows:
        a[i] += carry
        ex = np.maximum(a[i] - h_stable, 0.0) * share
        a[i] -= ex
        carry = ex
    return a, carry


def drain(h, g_x, g_w, h_stable, sealed_low_w=False, sealed_high_w=False):
    """Gravity drainage on a blade layer h[w, x] (rows across width, cols along length).

    g_x, g_w: gravity components along +x (blade length) and +w (width, trailing ->
    leading), in units of g. Material above h_stable moves downhill; what crosses an
    unsealed edge is returned as dropped (thickness summed over cells, m). A width edge
    that is pressed on the wall (sealed) keeps what reaches it. The long ends of the blade
    are never sealed.
    """
    tot = abs(g_x) + abs(g_w)
    if not np.isfinite(h_stable) or tot < 1e-9:
        return h.copy(), 0.0
    dropped = 0.0
    h = h.copy()
    if abs(g_w) > 1e-9:
        down_up = g_w > 0
        h, out = _cascade(h, down_up, abs(g_w) / tot, h_stable)
        sealed = sealed_high_w if down_up else sealed_low_w
        if sealed:
            h[-1 if down_up else 0] += out
        else:
            dropped += float(out.sum())
    if abs(g_x) > 1e-9:
        t, out = _cascade(h.T, g_x > 0, abs(g_x) / tot, h_stable)
        h = t.T
        dropped += float(out.sum())
    return h, dropped


# ====================================================================== stroke system
from .material import MaterialSystem   # noqa: E402  (grid, loading, metrics, volume books)


class StrokeStats(dict):
    """dict with attribute access, so env code can use stats.peak_force_N as before."""
    __getattr__ = dict.get

    @property
    def __dict__(self):
        return dict(self)


class MortarSystem(MaterialSystem):
    """Wall + blade layers with the physics in this module.

    Blade layer `blade[r, c]`: rows across the blade width, cols along its length (same
    layout as v0.2). While the blade is on the wall it holds the whole mortar column
    between steel and wall (wall cells under it are absorbed and re-deposited as the
    trailing edge leaves them).

    Per-sample API (used by the MuJoCo co-simulation in arm.py):
        begin_stroke(phi, direction) -> air(pitch) -> contact(centre, pitch, ...)* ->
        end_stroke() -> air(0)
    `stroke()` runs the same sequence on the nominal path (kinematic fallback).
    """

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.p = MortarParams(rho=cfg.mortar_rho, tau_y=cfg.mortar_tau_y_Pa, mu_p=cfg.mortar_mu_p_Pa_s,
                              lift_wall_fraction=cfg.lift_wall_fraction)
        self.h_slump = stable_thickness(self.p, 1.0)      # vertical wall
        self.last_pitch = 0.0
        self._covered = {}
        self._flip = False
        self._bead = np.zeros(cfg.blade_shape[1])     # volume-per-cell-area pushed ahead of the blade

    @property
    def blade_volume_m3(self):
        """Material the blade carries: the layer on its face plus the bead it pushes."""
        return float((self.blade.sum() + self._bead.sum()) * self.blade_cell_area)

    def reset(self, initial='bare'):
        self._bead = np.zeros(self.cfg.blade_shape[1])
        return super().reset(initial)

    # ------------------------------------------------------------------ orientation
    def _axes(self, phi):
        e_x = np.array([np.cos(phi), np.sin(phi)])
        e_w = np.array([-np.sin(phi), np.cos(phi)])       # blade row index increases along e_w
        return e_x, e_w

    def begin_stroke(self, phi, direction):
        """Orient the blade so that row 0 is the TRAILING edge for this stroke."""
        self.phi = float(phi)
        e_x, e_w = self._axes(phi)
        self._flip = float(e_w @ np.asarray(direction, float)) < 0
        self.e_x = e_x
        self.e_w = -e_w if self._flip else e_w              # trailing -> leading, in the wall plane
        self._covered = {}
        self._gaps = None
        self._g0 = None
        self._last_centre = None

    def _rows(self):
        return self.blade[::-1].copy() if self._flip else self.blade.copy()

    def _store(self, B):
        self.blade[:] = B[::-1] if self._flip else B

    def _coords(self, centre, row_offsets):
        """Wall (u, v) of blade cells; row_offsets in cells across the width from the centre line."""
        bz, bx = self.blade.shape
        along = (np.arange(bx) - (bx - 1) / 2) * self.cfg.blade_cell_m
        across = np.asarray(row_offsets, float) * self.cfg.blade_cell_m
        X, Z = np.meshgrid(along, across)
        U = centre[0] + X * self.e_x[0] + Z * self.e_w[0]
        V = centre[1] + X * self.e_x[1] + Z * self.e_w[1]
        return U, V

    def _gravity(self, pitch):
        """Gravity on the blade face in units of g: (along length, along width trailing->leading,
        normal towards the wall). Wall frame: u horizontal, v up, n into the wall."""
        c, s = np.cos(pitch), np.sin(pitch)
        g_x = -self.e_x[1]
        g_w = -c * self.e_w[1]
        g_n = -s * self.e_w[1]
        return g_x, g_w, g_n

    # ------------------------------------------------------------------ phases
    def air(self, pitch, stats=None):
        """Blade off the wall at this pitch: drainage / peel-off. Returns dropped m^3."""
        B = self._rows()
        g_x, g_w, g_n = self._gravity(pitch)
        dropped = 0.0
        if g_n > 1e-9:
            # face overhangs: weight pulls the layer off the face
            peel = self.p.rho * G * B * g_n > self.p.tau_y
            dropped += float(B[peel].sum()); B[peel] = 0.0
        h_st = stable_thickness(self.p, float(np.hypot(g_x, g_w)))
        B, d = drain(B, g_x, g_w, h_st)
        dropped = (dropped + d) * self.blade_cell_area
        self._store(B)
        self.dropped_m3 += dropped
        if stats is not None:
            stats['dropped_m3'] = stats.get('dropped_m3', 0.0) + dropped
            stats['air_dropped_m3'] = stats.get('air_dropped_m3', 0.0) + dropped
        return dropped

    def contact(self, centre, pitch, speed_m_s, force_N=None, gap_m=None, stats=None):
        """One sample of the blade sliding on the wall. Give force_N (quasi-static force
        balance decides the gap) or gap_m (co-simulation: MuJoCo decides where the blade is).
        Returns SampleResult."""
        c = self.cfg; A_w, A_b = self.wall_cell_area, self.blade_cell_area
        nw = self.blade.shape[0]
        self.last_pitch = float(pitch)
        B = self._rows()
        U, V = self._coords(centre, np.arange(nw) - (nw - 1) / 2)
        iv, iu, good = self._indices(U, V)
        flat = np.where(good, iv * self.wall.shape[1] + iu, -1)
        res = SampleResult()
        # 0) the wall slides back under the blade and shears the column towards the trailing edge
        if self._last_centre is not None:
            ds = float((np.asarray(centre, float) - self._last_centre) @ self.e_w)
            a = np.clip(self.p.couette_fraction * max(ds, 0.0) / c.blade_cell_m, 0.0, 1.0)
            if a > 0:
                moved = a * B
                B = B - moved
                B[:-1] += moved[1:]          # row j+1 -> row j (towards the trailing edge)
                B[0] += moved[0]             # trailing row: waits for the wall cells behind it
        self._last_centre = np.asarray(centre, float).copy()
        # 1) wall cells the trailing edge has left: re-deposit from the column that covered them
        current = set(flat[good].tolist())
        if self._covered:
            gaps_prev = self._gaps if self._gaps is not None else np.zeros(nw)
            # The trailing edge is a line at gap g0; it wipes the layer to g0, not to the gap at
            # the centre of the trailing row (which is g0 + half a cell x sin(pitch)).
            exit_gap = gaps_prev.copy()
            exit_gap[0] = self._g0 if self._g0 is not None else gaps_prev[0]
            for key in [k for k in self._covered if k not in current]:
                r, cc = self._covered.pop(key)
                amount = min(B[r, cc], exit_gap[r]) * A_b / A_w
                self.wall.ravel()[key] += amount
                B[r, cc] -= amount * A_w / A_b
                res.deposited_m3 += amount * A_w
        # 2) newly covered wall cells: absorb into the moving column
        new = good & ~np.isin(flat, list(self._covered.keys()))
        if new.any():
            keys = flat[new]
            counts = np.bincount(keys, minlength=self.wall.size)[keys]
            h = self.wall.ravel()[keys]
            B[new] += h / counts * A_w / A_b
            res.absorbed_m3 = float(np.sum(h / counts) * A_w)
            self.wall.ravel()[np.unique(keys)] = 0.0
        # 2b) the bead the blade is pushing re-enters the gap at the leading edge
        if self._bead.any():
            B[nw - 1] += self._bead; self._bead = np.zeros_like(self._bead)
        # 3) where does the blade ride?
        rise = (np.arange(nw) + 0.5) * c.blade_cell_m * np.sin(max(pitch, 0.0))
        if gap_m is None:
            g0, support = solve_gap(B, pitch, force_N or 0.0, c.blade_cell_m, self.p)
        else:
            g0 = max(float(gap_m), 0.0)
            support = squeeze_support(B, g0 + rise, c.blade_cell_m, self.p.tau_y)
        gaps = g0 + rise
        wet = B >= gaps[:, None] - 1e-12
        res.gap_m, res.support_N, res.wetted_frac = g0, support, float(wet.mean())
        res.drag_N = float(np.sum(wet * (self.p.tau_y + self.p.mu_p * speed_m_s / np.maximum(gaps[:, None], 1e-4)))) * A_b
        # 4) squeeze out what does not fit under the blade
        B, trail, lead = extrude(B, gaps)
        # leading-edge extrusion: the bead stays against the moving blade
        self._bead = self._bead + lead
        res.extruded_lead_m3 = float(lead.sum() * A_b)
        for vols, row in ((trail, -1),):
            if not np.any(vols > 0):
                continue
            Ue, Ve = self._coords(centre, [row - (nw - 1) / 2])
            ev, eu, eg = self._indices(Ue[0], Ve[0])
            vol = vols * A_b
            inside = eg & ~np.isin(np.where(eg, ev * self.wall.shape[1] + eu, -1), list(current))
            np.add.at(self.wall, (ev[inside], eu[inside]), vol[inside] / A_w)
            back = eg & ~inside                  # lands under the blade itself: stays in the column
            if back.any():
                B[0 if row < 0 else nw - 1, back] += vols[back]
            lost = float(vol[~eg].sum())
            self.outside_m3 += lost; res.outside_m3 += lost
            res.extruded_trail_m3 = float(vol[inside].sum())
        # 5) remember which blade cell covers which wall cell (for step 1 next sample)
        rr, cc = np.nonzero(good)
        for key, r, col in zip(flat[good].tolist(), rr.tolist(), cc.tolist()):
            self._covered[key] = (r, col)
        self._gaps = gaps; self._g0 = g0
        self._store(B)
        # 6) wall slump: layers too thick for the vertical wall slough off
        res.dropped_m3 = self._slump()
        if stats is not None:
            for k in ('deposited_m3', 'dropped_m3', 'outside_m3', 'extruded_trail_m3', 'extruded_lead_m3'):
                stats[k] = stats.get(k, 0.0) + getattr(res, k)
            stats['peak_force_N'] = max(stats.get('peak_force_N', 0.0), (force_N or support) + res.drag_N * 0)
            stats['max_drag_N'] = max(stats.get('max_drag_N', 0.0), res.drag_N)
            stats.setdefault('gaps_mm', []).append(g0 * 1000)
            stats.setdefault('support_N', []).append(support)
        return res

    def end_stroke(self, stats=None):
        """Blade pulled off the wall: each covered column splits between wall and blade."""
        B = self._rows(); A_w, A_b = self.wall_cell_area, self.blade_cell_area
        f = self.p.lift_wall_fraction
        users = {}
        for key, rc in self._covered.items():
            users.setdefault(rc, []).append(key)
        left = 0.0
        for (r, cc), keys in users.items():
            share = f * B[r, cc] / len(keys)
            for key in keys:
                self.wall.ravel()[key] += share * A_b / A_w
            B[r, cc] -= share * len(keys)
            left += share * len(keys) * A_b
        self._covered = {}
        self._store(B)
        # the bead is left on the wall just ahead of where the blade lifted off
        if self._bead.any() and self._last_centre is not None:
            nw = self.blade.shape[0]
            Ue, Ve = self._coords(self._last_centre, [nw - (nw - 1) / 2])
            ev, eu, eg = self._indices(Ue[0], Ve[0])
            vol = self._bead * A_b
            np.add.at(self.wall, (ev[eg], eu[eg]), vol[eg] / A_w)
            lost = float(vol[~eg].sum())
            self.outside_m3 += lost
            if stats is not None:
                stats['outside_m3'] = stats.get('outside_m3', 0.0) + lost
            left += float(vol[eg].sum())
            self._bead = np.zeros_like(self._bead)
        dropped = self._slump()
        if stats is not None:
            stats['deposited_m3'] = stats.get('deposited_m3', 0.0) + left
            stats['dropped_m3'] = stats.get('dropped_m3', 0.0) + dropped
        return left

    def _slump(self):
        ex = np.maximum(self.wall - self.h_slump, 0.0)
        if not ex.any():
            return 0.0
        self.wall -= ex
        v = float(ex.sum() * self.wall_cell_area)
        self.dropped_m3 += v
        return v

    # ------------------------------------------------------------------ kinematic stroke
    def stroke(self, mode, start, end, phi, bend, force_N, speed_m_s, callback=None,
               pitch_start=0.0, pitch_end=0.0):
        """Nominal-path stroke (no arm): same physics as the co-simulation, but the blade is
        assumed to follow the plan exactly and the gap comes from the force balance."""
        c = self.cfg
        p0, p2 = np.asarray(start, float), np.asarray(end, float)
        d = p2 - p0; L = np.linalg.norm(d)
        stats = StrokeStats(peak_force_N=0.0, deposited_m3=0.0, dropped_m3=0.0, outside_m3=0.0)
        if L < 1e-9:
            return stats
        direction = d / L
        pc = (p0 + p2) / 2 + np.array([-direction[1], direction[0]]) * bend
        self.begin_stroke(phi, direction)
        self.air(pitch_start, stats)                                 # approach, blade stood up
        for k, t in enumerate(np.linspace(0, 1, c.stroke_samples)):
            centre = (1-t)**2*p0 + 2*(1-t)*t*pc + t**2*p2
            pitch = pitch_start + (pitch_end - pitch_start) * t
            self.contact(centre, pitch, speed_m_s, force_N=force_N, stats=stats)
            if callback is not None:
                callback(k, centre.copy(), self.wall.copy(), self.blade.copy())
        self.end_stroke(stats)
        self.air(0.0, stats)                                          # retreat, blade level
        self.wall_age += L / max(speed_m_s, 1e-4)
        return stats
