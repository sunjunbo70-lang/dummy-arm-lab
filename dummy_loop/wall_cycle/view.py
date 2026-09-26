"""Whole-wall plastering replay on the REAL Dummy V2 MuJoCo model (not a stick figure).

    python -m dummy_loop.wall_cycle.view --policy <policy.npz> --seed 3
    python -m dummy_loop.wall_cycle.view --record experiments/00_initial_debug/runs/wall_cycle_unclassified/replay/rollout.npz
    python -m dummy_loop.wall_cycle.view --teacher technique --seed 3      # no policy needed

Opens MuJoCo's own 3-D window: the Dummy V2 meshes, the J6 reducer + holder + trowel,
the wall, the work square outline, and the mortar layer drawn cell by cell on the wall
(colour = thickness; the geometry is display-only and never enters the physics).
Left mouse rotates, right mouse pans, wheel zooms.

Keys:  space pause/play   right/left one frame   [ / ] slower/faster   R restart

What you see is the actual rollout, recorded through the MuJoCo co-simulation
(dummy_loop/wall_cycle/arm.py): during every stroke the joint angles are the ones MuJoCo
simulated (servos, torque limits, mortar reaction force), and the mortar on the wall and
on the trowel face is the mortar model's state after that sample. Approach / lift poses
are the planned ones; moves between them (to the scan pose, to the loading pose) are
joint interpolation in free space, labelled "move", and are checked for wall clearance.
Evidence level L1. Why this replaced the v0.2 web player: experiments/v0.3/r0/design/2026-09-22_wall_cycle_v0.3.md C11.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from .arm import ArmExecutor
from .config import CycleConfig
from .env import WallCycleEnv

PHASE_TEXT = {
    'SCAN': 'D435 scan', 'DECIDE': 'policy decides', 'SCAN_RETURN': 'back to scan pose',
    'LOAD_APPROACH': 'to loading pose (position illustrative)', 'DISPENSE': 'random load onto trowel',
    'TOOL_INSPECT': 'check load', 'WALL_APPROACH': 'approach: level -> stand blade up',
    'CONTACT_ACQUIRE': 'touch: trailing edge first', 'WORK_STEP': 'stroke', 'WORK': 'stroke done',
    'LIFT': 'lift', 'RETREAT': 'retreat', 'UNREACHABLE': 'REJECTED: arm cannot do this stroke',
    'LOAD': 'move to feed board', 'SCOOP': 'scoop mortar',
    'LIFT_FROM_FEED': 'lift from feed board', 'CARRY': 'carry face-up',
    'PRECONTACT': 'pre-contact waypoint', 'ROTATE_TO_WALL': 'rotate near wall',
    'SEPARATE': 'separate from wall', 'RECOVER_RETURN': 'recover / return',
    'FINISH': 'finish check', 'move': 'move',
    'FEED_TRANSIT': 'move to the loading pose', 'FEED_ALIGN_UP': 'align material face upward', 'FEED_SCOOP': 'load at face-up pose',
    'CARRY_FACE_UP': 'carry with measured face-up pose',
}


def load_pose(ex: ArmExecutor):
    """Loading station is not modelled; turn J1 away from the wall from the scan pose."""
    q = ex.q_scan.copy(); q[0] = np.clip(q[0] - np.deg2rad(60), ex.ik.lo[0], ex.ik.hi[0])
    return q


def load_policy(env, path):
    """Load a PPO checkpoint without assuming the training MLP width.

    Checkpoints predate embedded model metadata, so infer the hidden widths from the
    policy weight matrices. This keeps v0.2/v0.3 policies compatible with v0.5 P1.
    """
    from ..wall.ppo import PPO, PPOConfig
    with np.load(path, allow_pickle=False) as z:
        indices = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('pi_'))
        if not indices or indices != list(range(len(indices))) or len(indices) % 2:
            raise ValueError(f'invalid PPO checkpoint layout: {path}')
        layers = len(indices) // 2
        weights = [z[f'pi_{i}'] for i in range(layers)]
        if weights[0].shape[0] != env.obs_dim or weights[-1].shape[1] != env.act_dim:
            raise ValueError(f'policy dimensions do not match {env.cfg.physics}: '
                             f'{weights[0].shape[0]}->{weights[-1].shape[1]} vs '
                             f'{env.obs_dim}->{env.act_dim}')
        hidden = tuple(int(w.shape[1]) for w in weights[:-1])
    return PPO(env.obs_dim, env.act_dim, PPOConfig(hidden=hidden)).load(path)


MODES_ = ('DEPOSIT', 'REUSE', 'LEVEL', 'RESCAN', 'FINISH')   # env.MODES order, for progress lines


def record(cfg: CycleConfig, seed=3, policy=None, teacher_style='technique',
           max_deg_per_frame=2.0, initial_mix=False):
    """Run one episode with the arm executor and return display frames (numpy arrays)."""
    from .train import make_env
    env = make_env(cfg, seed, teacher_style, 'cosim', initial_mix=initial_mix, record=True)
    ex = env.executor
    agent = None
    if policy is not None:
        agent = load_policy(env, policy)
    obs = env.reset(); done = False; total = 0.0
    print('recording one full MuJoCo co-simulation episode (progress per stroke) ...', flush=True)
    while not done:
        a = env.teacher_action() if agent is None else agent.act(obs, deterministic=True)[0]
        obs, r, done, info = env.step(a); total += r
        m = info['metrics']
        print(f"  stroke {info['cycles']:3d}  {MODES_[env.last_mode]:<7s}  coverage {m['coverage']*100:5.1f} %  "
              f"rmse {m['rmse_mm']:.2f} mm", flush=True)

    q_load = load_pose(ex)
    keys = []           # (phase, q, wall, blade_ml, metrics, pitch, cycle, action)
    plan = None; act = {}
    traces = {e['cycle']: e['trajectory'] for e in env.events if e['phase'] == 'ARM_TRACE'}
    trace_added = set()
    for ev in env.events:
        ph = ev['phase']
        if ph == 'ARM_PLAN':
            plan = ev; continue
        if cfg.physics == 'v0.6' and ph == 'ARM_TRACE':
            continue
        wall = ev['wall']; m = ev['metrics']; act = ev.get('action') or act
        base = dict(wall=wall, blade=ev['blade'], blade_ml=m['blade_load_ml'], metrics=m, cycle=ev['cycle'], action=act)
        if cfg.physics == 'v0.6' and ph == 'DECIDE' and ev['cycle'] in traces and ev['cycle'] not in trace_added:
            for tr in traces[ev['cycle']]:
                if tr['phase'] in ('WORK_STEP','SEPARATE','SCAN_RETURN'):
                    continue
                tm=tr['metrics']
                keys.append(dict(phase=tr['phase'],q=tr['q'],pitch=0.0,wall=tr['wall'],blade=tr['blade'],
                                 blade_ml=tm['blade_load_ml'],metrics=tm,cycle=ev['cycle'],action=act,
                                 face_up_score=tr.get('face_up_score')))
            trace_added.add(ev['cycle'])
        if ph in ('SCAN', 'SCAN_RETURN', 'DECIDE', 'FINISH', 'UNREACHABLE'):
            keys.append(dict(phase=ph, q=keys[-1]['q'] if (keys and ph in ('DECIDE', 'UNREACHABLE', 'FINISH'))
                             else ex.q_scan, pitch=0.0, **base))
        elif cfg.physics == 'v0.6' and ph in ('LOAD', 'SCOOP', 'LIFT_FROM_FEED', 'TOOL_INSPECT', 'CARRY',
                                               'PRECONTACT', 'ROTATE_TO_WALL', 'WALL_APPROACH',
                                               'CONTACT_ACQUIRE', 'SEPARATE', 'RECOVER_RETURN'):
            continue
        elif ph in ('LOAD_APPROACH', 'DISPENSE', 'TOOL_INSPECT', 'LOAD', 'SCOOP', 'LIFT_FROM_FEED'):
            keys.append(dict(phase=ph, q=q_load, pitch=0.0, **base))
        elif ph == 'CARRY' and plan is not None:
            # Feed-board routing is not yet solved; display its transition to the first
            # collision-checked wall waypoint and label the approximation in the report.
            keys.append(dict(phase=ph, q=plan['q_approach'][0], pitch=0.0, **base))
        elif ph == 'PRECONTACT' and plan is not None:
            keys.append(dict(phase=ph, q=plan['q_approach'][1], pitch=0.0, **base))
        elif ph == 'ROTATE_TO_WALL' and plan is not None:
            keys.append(dict(phase=ph, q=plan['q_approach'][1],
                             pitch=np.deg2rad(act.get('pitch_start_deg', 0.0)), **base))
        elif ph == 'WALL_APPROACH' and plan is not None:
            p0 = np.deg2rad(act.get('pitch_start_deg', 0.0))
            keys.append(dict(phase=ph, q=plan['q_approach'][0], pitch=0.0, **base))
            keys.append(dict(phase=ph, q=plan['q_approach'][1], pitch=p0, **base))
        elif ph == 'CONTACT_ACQUIRE' and plan is not None:
            keys.append(dict(phase=ph, q=plan['q_approach'][2],
                             pitch=np.deg2rad(act.get('pitch_start_deg', 0.0)), **base))
        elif ph == 'WORK_STEP' and 'q' in ev:
            keys.append(dict(phase=ph, q=ev['q'], pitch=np.deg2rad(ev.get('pitch_deg', 0.0)), **base))
        elif cfg.physics == 'v0.6' and ph == 'WORK' and ev['cycle'] in traces:
            for tr in traces[ev['cycle']]:
                if tr['phase'] not in ('SEPARATE','SCAN_RETURN'):
                    continue
                tm=tr['metrics']
                keys.append(dict(phase=tr['phase'],q=tr['q'],pitch=0.0,wall=tr['wall'],blade=tr['blade'],
                                 blade_ml=tm['blade_load_ml'],metrics=tm,cycle=ev['cycle'],action=act,
                                 face_up_score=tr.get('face_up_score')))
        elif ph in ('LIFT', 'RETREAT', 'SEPARATE', 'RECOVER_RETURN') and plan is not None:
            lift_i = 0 if ph in ('LIFT', 'SEPARATE') else 1
            keys.append(dict(phase=ph, q=plan['q_lift'][lift_i], pitch=0.0, **base))
    # joint-space interpolation between key poses so nothing jumps on screen
    frames = []
    for k, f in enumerate(keys):
        if frames and cfg.physics != 'v0.6':
            q0 = frames[-1]['q']; jump = np.rad2deg(np.max(np.abs(np.asarray(f['q']) - q0)))
            n = int(np.ceil(jump / max_deg_per_frame))
            for j in range(1, n):
                frames.append({**frames[-1], 'phase': 'move', 'q': q0 + (np.asarray(f['q']) - q0) * j / n})
        frames.append({**f, 'q': np.asarray(f['q'], float)})
    # free-space moves are not planned by the executor: report how many come close to the wall
    close = sum(1 for f in frames if f['phase'] == 'move' and not ex.clear_of_wall(f['q']))
    decisions = [e['action'] for e in env.events if e['phase'] == 'DECIDE']
    carry_scores = [f.get('face_up_score') for f in frames
                    if f['phase'] in ('FEED_ALIGN_UP', 'FEED_SCOOP', 'CARRY_FACE_UP')
                    and f.get('face_up_score') is not None]
    rotate_scores = [f.get('face_up_score') for f in frames
                     if f['phase'] == 'ROTATE_TO_WALL' and f.get('face_up_score') is not None]
    note = ('v0.6 frames are actual executor records for feed alignment, carry, near-wall rotation, '
            'approach, work, separation and scan return; the feed material interaction remains a '
            'pose-gated reduced proxy rather than granular contact') if cfg.physics == 'v0.6' else (
            'stroke frames: joint angles from MuJoCo co-simulation; approach/lift: planned poses; '
            'load/carry/free-space moves: illustrative joint interpolation because the feed station path '
            'is not yet solved by the arm planner')
    report = {'evidence_level': 'L1', 'hardware_motion': False, 'seed': seed,
              'initial_distribution': 'mixed_test' if initial_mix else 'bare',
              'driver': str(policy) if policy is not None else f'teacher:{teacher_style}',
              'return': total, 'success': bool(info['success']), 'metrics': info['metrics'],
              'cycles': info['cycles'], 'reloads': info['reloads'],
              'unreachable_strokes': info['unreachable_strokes'],
              'projected_strokes': info.get('projected_strokes', 0),
              'mode_counts': {m: sum(d['mode'] == m for d in decisions)
                              for m in ('DEPOSIT', 'REUSE', 'LEVEL', 'RESCAN', 'FINISH')},
              'pitch_start_deg_mean': float(np.mean([d['pitch_start_deg'] for d in decisions
                                                     if d['mode'] in ('DEPOSIT', 'REUSE')] or [0])),
              'frames': len(frames), 'move_frames': sum(f['phase'] == 'move' for f in frames),
              'move_frames_near_wall': close,
              'carry_face_up_min': float(min(carry_scores)) if carry_scores else None,
              'rotation_face_up_min': float(min(rotate_scores)) if rotate_scores else None,
              'face_down_frames': int(sum(x < -1e-3 for x in carry_scores + rotate_scores)),
              'config': cfg.to_dict(), 'note': note}
    return frames, report


def save(frames, report, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    phases = sorted({f['phase'] for f in frames})
    np.savez_compressed(
        out / 'rollout.npz',
        q=np.array([f['q'] for f in frames], np.float32),
        wall_mm=np.array([f['wall'] * 1000 for f in frames], np.float16),
        blade_mm=np.array([f['blade'] * 1000 for f in frames], np.float16),
        blade_angle_deg=np.array([(f.get('action') or {}).get('blade_angle_deg', 180.0) for f in frames], np.float32),
        pitch_deg=np.array([np.rad2deg(f['pitch']) for f in frames], np.float32),
        blade_ml=np.array([f['blade_ml'] for f in frames], np.float32),
        coverage=np.array([f['metrics']['coverage'] for f in frames], np.float32),
        rmse_mm=np.array([f['metrics']['rmse_mm'] for f in frames], np.float32),
        waste=np.array([f['metrics']['waste_frac'] for f in frames], np.float32),
        cycle=np.array([f['cycle'] for f in frames], np.int32),
        phase=np.array([phases.index(f['phase']) for f in frames], np.int16),
        face_up_score=np.array([f.get('face_up_score', np.nan) for f in frames], np.float32),
        phase_names=np.array(phases), config=json.dumps(report['config']))
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return out / 'rollout.npz'


def load(path: Path):
    z = np.load(path, allow_pickle=False)
    cfg = CycleConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(str(z['config'])).items()})
    return cfg, z


def colour(h_mm):
    """0 mm dark blue -> 2 mm (target) teal -> >=4 mm amber, same scale as the other players."""
    t = float(np.clip(h_mm / 4.0, 0, 1))
    if t < .5:
        a = t * 2; return np.array([.08 + .10 * a, .15 + .65 * a, .24 + .47 * a, 1.0])
    a = (t - .5) * 2; return np.array([.18 + .82 * a, .80 - .11 * a, .71 - .45 * a, 1.0])


class Stage:
    """Scene + drawing helpers shared by the interactive viewer and offline rendering."""

    def __init__(self, cfg: CycleConfig):
        self.cfg = cfg
        self.ex = ArmExecutor(cfg)
        self.model, self.data = self.ex.model, self.ex.data
        c = cfg
        nv, nu = c.wall_shape
        self.u = np.linspace(-c.width_m / 2 + c.cell_m / 2, c.width_m / 2 - c.cell_m / 2, nu)
        self.v = np.linspace(c.cell_m / 2, c.height_m - c.cell_m / 2, nv)

    def pose(self, q):
        self.data.qpos[:] = 0; self.data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.data)

    def draw(self, scn, wall_mm):
        """Mortar cells + work-square outline as display-only geoms."""
        c, fr = self.cfg, self.ex.frame
        R = fr.R.flatten()
        half = c.cell_m / 2 * 0.98
        for i, j in np.argwhere(wall_mm > 0.02):
            if scn.ngeom >= scn.maxgeom:
                break
            h = float(wall_mm[i, j]) / 1000
            p = fr.to_world(self.ex._uvn(self.u[j], self.v[i], -h / 2 - 0.0002))
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([half, half, h / 2]), p, R, colour(wall_mm[i, j]).astype(np.float32))
            scn.ngeom += 1
        corners = [(-c.width_m / 2, 0), (c.width_m / 2, 0), (c.width_m / 2, c.height_m), (-c.width_m / 2, c.height_m)]
        for k in range(4):
            if scn.ngeom >= scn.maxgeom:
                break
            a = fr.to_world(self.ex._uvn(*corners[k], -0.001)); b = fr.to_world(self.ex._uvn(*corners[(k + 1) % 4], -0.001))
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                                np.array([1.0, .82, .25, 1.0], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0012, a, b)
            scn.ngeom += 1

    def draw_blade(self, scn, blade_mm, phi_deg):
        """Mortar on the trowel face, cell by cell, at its real thickness (display only).
        Mortar grid rows run along e_w=(-sin phi, cos phi); in the blade body frame that is -z
        (tool_rotation: blade z = x_w cross n), columns run along blade x."""
        if blade_mm is None or not np.any(blade_mm > 0.05):
            return
        b = self.ex.blade_body
        R = self.data.xmat[b].reshape(3, 3); o = self.data.xpos[b]
        th = self.ex.scene.trowel_thickness
        nw, nx = blade_mm.shape
        cell = self.cfg.blade_cell_m
        zscale = self.ex.hw * 2 / (nw * cell)          # mortar grid is 30 mm wide, model blade 27.5 mm
        for r in range(nw):
            for c_ in range(nx):
                h = float(blade_mm[r, c_]) / 1000
                if h < 5e-5 or scn.ngeom >= scn.maxgeom:
                    continue
                x = self.ex.xc + (c_ - (nx - 1) / 2) * cell
                z = -(r - (nw - 1) / 2) * cell * zscale
                pos = o + R[:, 0] * x + R[:, 2] * z + R[:, 1] * (th + h / 2)
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                                    np.array([cell / 2 * .96, h / 2, cell * zscale / 2 * .96]), pos,
                                    R.flatten(), np.array([.62, .58, .52, 1.0], np.float32))
                scn.ngeom += 1

    def heatmap(self, wall_mm, size=220):
        nv, nu = wall_mm.shape
        img = np.zeros((nv, nu, 3), np.uint8)
        for i in range(nv):
            for j in range(nu):
                img[nv - 1 - i, j] = (colour(wall_mm[i, j])[:3] * 255).astype(np.uint8)
        ys = (np.arange(size) * nv // size); xs = (np.arange(size) * nu // size)
        return img[ys][:, xs]

    def camera(self, cam):
        # from the robot's front-left, looking along +X towards the wall face
        cam.lookat[:] = self.ex.frame.to_world([0, 0, -0.12])
        cam.distance, cam.azimuth, cam.elevation = 0.72, 50, -18


def text_lines(z, i, speed, paused):
    names = [str(x) for x in z['phase_names']]
    ph = names[int(z['phase'][i])]
    left = '\n'.join(['phase', 'cycle', 'trowel pitch', 'load on trowel', 'coverage', 'thickness RMSE',
                      'waste', 'frame', 'speed'])
    right = '\n'.join([PHASE_TEXT.get(ph, ph), str(int(z['cycle'][i])), f"{float(z['pitch_deg'][i]):.1f} deg",
                       f"{float(z['blade_ml'][i]):.1f} mL", f"{float(z['coverage'][i]) * 100:.1f} %",
                       f"{float(z['rmse_mm'][i]):.3f} mm", f"{float(z['waste'][i]) * 100:.1f} %",
                       f"{i + 1}/{len(z['q'])}", f"{speed:g}x" + ('  PAUSED' if paused else '')])
    return left, right


SPEED_LEVELS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
FRAME_DT_S = 0.05     # recorded frames are shown at 20 fps at 1x


def play(path: Path, speed=1.0):
    import mujoco.viewer
    cfg, z = load(path)
    st = Stage(cfg)
    speed = float(min(SPEED_LEVELS, key=lambda s: abs(s - speed)))
    # 'pos' is a fractional frame index advanced by wall-clock time, so N x speed really is
    # N x even when drawing one frame takes longer than 50 ms (frames are skipped, not slowed).
    state = {'i': 0, 'pos': 0.0, 'paused': False, 'speed': speed, 'step': 0}

    def shift_speed(d):
        k = SPEED_LEVELS.index(state['speed']) + d
        state['speed'] = SPEED_LEVELS[int(np.clip(k, 0, len(SPEED_LEVELS) - 1))]

    def key(k):
        if k == 32: state['paused'] = not state['paused']
        elif k == 262: state['step'] = 1
        elif k == 263: state['step'] = -1
        elif k == 93: shift_speed(+1)
        elif k == 91: shift_speed(-1)
        elif k == 82: state['i'] = 0; state['pos'] = 0.0

    with mujoco.viewer.launch_passive(st.model, st.data, key_callback=key,
                                      show_left_ui=False, show_right_ui=False) as v:
        st.camera(v.cam)
        n = len(z['q'])
        last = time.monotonic()
        while v.is_running():
            t0 = time.monotonic()
            i = state['i']
            with v.lock():
                st.data.qpos[:] = 0; st.data.qpos[:6] = z['q'][i]
                mujoco.mj_forward(st.model, st.data)
                v.user_scn.ngeom = 0
                st.draw(v.user_scn, z['wall_mm'][i].astype(float))
                if 'blade_mm' in z:
                    st.draw_blade(v.user_scn, z['blade_mm'][i].astype(float), float(z['blade_angle_deg'][i]))
            left, right = text_lines(z, i, state['speed'], state['paused'])
            v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
                         (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                          'space pause   <- -> step   [ slower  ] faster (0.25x-10x)   R restart',
                          'L1 simulation, not hardware')])
            vp = v.viewport
            if vp is not None and vp.width > 400:
                s = 220
                v.set_images((mujoco.MjrRect(vp.width - s - 10, 10, s, s), st.heatmap(z['wall_mm'][i].astype(float), s)))
            v.sync()
            now = time.monotonic(); elapsed = now - last; last = now
            if state['step']:
                state['i'] = int(np.clip(i + state['step'], 0, n - 1)); state['step'] = 0; state['paused'] = True
                state['pos'] = float(state['i'])
            elif not state['paused']:
                # at least one frame per redraw, otherwise as many as wall-clock time x speed asks for
                state['pos'] = max(state['pos'] + elapsed * state['speed'] / FRAME_DT_S, state['i'] + 1.0)
                if state['pos'] >= n: state['pos'] = 0.0
                state['i'] = int(state['pos'])
            else:
                state['pos'] = float(i)
            time.sleep(max(0.0, FRAME_DT_S / state['speed'] - (time.monotonic() - t0)))


def render_png(path: Path, out: Path, indices, size=(960, 640)):
    """Offline check / thumbnails with the same drawing code (no window needed)."""
    cfg, z = load(path)
    st = Stage(cfg)
    st.model.vis.global_.offwidth = max(st.model.vis.global_.offwidth, size[0])
    st.model.vis.global_.offheight = max(st.model.vis.global_.offheight, size[1])
    r = mujoco.Renderer(st.model, height=size[1], width=size[0], max_geom=10000)
    cam = mujoco.MjvCamera(); st.camera(cam)
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for i in indices:
        st.pose(z['q'][i]); r.update_scene(st.data, camera=cam)
        st.draw(r.scene, z['wall_mm'][i].astype(float))
        if 'blade_mm' in z:
            st.draw_blade(r.scene, z['blade_mm'][i].astype(float), float(z['blade_angle_deg'][i]))
        f = out / f'frame_{i:05d}.png'; Image.fromarray(r.render()).save(f); files.append(f)
    r.close()
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--record', type=Path, help='play an existing rollout.npz')
    ap.add_argument('--policy', type=Path, help='run this policy and play it')
    ap.add_argument('--teacher', choices=('technique', 'flat', 'legacy_transport'),
                    help='run the hand-written teacher instead')
    ap.add_argument('--v05', action='store_true', help='use v0.5 lab tool, loading and transport physics')
    ap.add_argument('--v06', action='store_true', help='use v0.6 pose-derived loading and continuous trajectory')
    ap.add_argument('--v08', action='store_true', help='v0.6 physics with the v0.8 recipe (recipes.py)')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', type=Path, default=Path('experiments/00_initial_debug/runs/wall_cycle_unclassified/replay'))
    ap.add_argument('--no-window', action='store_true', help='only record (and write report.json)')
    ap.add_argument('--speed', type=float, default=1.0,
                    help='initial playback speed, snapped to 0.25/0.5/1/2/3/5/10x; [ and ] change it live')
    ap.add_argument('--mixed-initial', action='store_true',
                    help='use the same bare/partial initial-state distribution as evaluation')
    a = ap.parse_args(argv)
    path = a.record
    if path is None:
        from .area import load_work_area
        from .recipes import make_config
        if a.v08:
            cfg=make_config('v0.8')
        elif a.v06:
            cfg=make_config('v0.7')          # identical to the inline v0.6 config used before v0.8
        elif a.v05:
            cfg=CycleConfig(physics='v0.5',tool_profile='lab_20260922',lift_wall_fraction=.75)
        else:
            cfg=CycleConfig()
        cfg = load_work_area(cfg)
        frames, report = record(cfg, a.seed, a.policy, a.teacher or 'technique',
                                initial_mix=a.mixed_initial)
        path = save(frames, report, a.out)
        print(json.dumps({k: v for k, v in report.items() if k != 'config'}, ensure_ascii=False, indent=1))
    if not a.no_window:
        play(path, a.speed)


if __name__ == '__main__':
    main()
