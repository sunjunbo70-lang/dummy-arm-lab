"""Candidate numerical repair; historical mortar remains unchanged."""
import numpy as np
from ...wall_cycle.mortar import MortarSystem, SampleResult, solve_gap, squeeze_support, extrude

class StableIndexMortar(MortarSystem):
    """Snap only floating-point roundoff at exact grid boundaries (5e-12 m)."""
    physics_version='v10r1.grid_boundary_roundoff.v1'
    def _indices(self,U,V):
        x=(U+self.cfg.width_m/2)/self.cfg.cell_m;y=V/self.cfg.cell_m
        x=np.where(np.abs(x-np.rint(x))<1e-9,np.rint(x),x)
        y=np.where(np.abs(y-np.rint(y))<1e-9,np.rint(y),y)
        iu=np.floor(x).astype(int);iv=np.floor(y).astype(int)
        good=(iu>=0)&(iu<self.wall.shape[1])&(iv>=0)&(iv<self.wall.shape[0])
        return iv,iu,good


class DisplacementMortar(StableIndexMortar):
    """Candidate distance-scaled exchange, not calibrated or approved training physics.

    Existing blade cell width defines the travel scale. This changes the discrete
    dynamics and requires conservation and convergence validation before adoption.
    """
    physics_version='v10r1.displacement_flux_candidate.v1'
    def contact(self, centre, pitch, speed_m_s, force_N=None, gap_m=None, stats=None):
        """One sample of the blade sliding on the wall. Give force_N (quasi-static force
        balance decides the gap) or gap_m (co-simulation: MuJoCo decides where the blade is).
        Returns SampleResult."""
        c = self.cfg; A_w, A_b = self.wall_cell_area, self.blade_cell_area
        nw = self.blade.shape[0]
        self.last_pitch = float(pitch)
        B = self._rows()
        displacement = (0.0 if self._last_centre is None else
                        float(np.linalg.norm(np.asarray(centre)-self._last_centre)))
        exchange = 1.0 if self._last_centre is None else min(displacement/c.blade_cell_m, 1.0)
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
            enter = exchange*self._bead
            B[nw - 1] += enter; self._bead -= enter
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
        equilibrium, trail, lead = extrude(B, gaps)
        B += exchange*(equilibrium-B)
        trail *= exchange; lead *= exchange
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
