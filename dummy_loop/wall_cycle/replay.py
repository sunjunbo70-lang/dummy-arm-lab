"""Export a trained v0.2 whole-cycle policy to the offline HTML stick-figure replay.

KEPT FOR THE v0.2 RECORD ONLY (experiments/2026-09-22_wall_cycle_rl). It is pinned to
CycleConfig(physics='v0.2') so the recorded 11-dim policies still load. v0.3 replays use
dummy_loop/wall_cycle/view.py: the real Dummy V2 MuJoCo model in MuJoCo's own window,
driven by the co-simulated joint angles (see docs/changes/2026-09-22_wall_cycle_v0.3.md).
"""
import argparse, hashlib, json
from pathlib import Path

import mujoco
import numpy as np

from ..wall.ppo import PPO, PPOConfig
from ..wall.scene import (SceneConfig, build_scene, joint_limits, tool_frame_matrix,
                          wall_frame)
from ..wall.controller import ToolIK, WallFrame
from .config import CycleConfig
from .env import WallCycleEnv

ROOT = Path(__file__).resolve().parents[2]


def _jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    return x


def load_agent(path, env):
    cfg = PPOConfig(hidden=(128, 128), lr=2e-5, clip=.1, init_log_std=-3)
    return PPO(env.obs_dim, env.act_dim, cfg).load(path)


class ArmKinematics:
    def __init__(self):
        self.cfg = SceneConfig()
        self.model, _ = build_scene(self.cfg); self.data = mujoco.MjData(self.model)
        lo, hi = joint_limits(self.cfg); lo[5] = max(lo[5], -np.pi); hi[5] = min(hi[5], np.pi)
        self.ik = ToolIK(self.model, lo, hi, tool_frame_matrix(self.cfg), w_roll=.1)
        self.frame = WallFrame(*wall_frame(self.cfg)); self.q = np.zeros(6)
        self.names = ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6']

    def solve(self, uv, n, psi):
        target = [uv[0], uv[1], n]
        p, R = self.frame.to_world(target), self.frame.tool_rotation(psi)
        q, ok, _, _ = self.ik.solve(p, R, self.q, iters=140)
        if not ok:
            rng=np.random.default_rng(17); best=None
            for seed in [np.zeros(6)]+[self.ik.lo+(self.ik.hi-self.ik.lo)*rng.random(6) for _ in range(8)]:
                candidate, good, ep, et=self.ik.solve(p,R,seed,iters=240)
                if good and (best is None or ep+et<best[0]): best=(ep+et,candidate)
            if best is not None: q,ok=best[1],True
        if ok: self.q = q
        return ok

    def home(self, blend=.25):
        self.q *= (1-blend)

    def points(self, q=None):
        self.data.qpos[:6] = self.q if q is None else q; mujoco.mj_forward(self.model, self.data)
        pts = [self.data.xpos[self.model.body(n).id].copy() for n in self.names]
        pts.append(self.data.site_xpos[self.model.site('tcp').id].copy())
        return np.asarray(pts)


def export(policy_path: Path, out: Path, seed=2209, max_cycles=80):
    out.mkdir(parents=True, exist_ok=False)
    cfg = CycleConfig(physics='v0.2'); env = WallCycleEnv(cfg, seed=seed, initial_mix=False, record=True)
    agent = load_agent(policy_path, env); obs = env.reset(); done = False; total = 0.
    while not done and env.cycles < max_cycles:
        action = agent.act(obs, deterministic=True)[0]
        obs, reward, done, info = env.step(action); total += reward

    arm = ArmKinematics(); frames=[]; last_action=None; last_uv=(0., cfg.height_m/2)
    wall_phases={'WALL_APPROACH','CONTACT_ACQUIRE','WORK_STEP','WORK','LIFT','RETREAT'}
    display_q=arm.q.copy()
    for ev in env.events:
        phase=ev['phase']; action=ev.get('action', last_action)
        if action is not None: last_action=action
        ok=True
        if phase in wall_phases and last_action is not None:
            if phase=='WORK_STEP': last_uv=tuple(ev['tool_center_uv_m'])
            elif phase in ('WALL_APPROACH','CONTACT_ACQUIRE'): last_uv=tuple(last_action['start_uv_m'])
            else: last_uv=tuple(last_action['end_uv_m'])
            n={'WALL_APPROACH':-.07,'CONTACT_ACQUIRE':-.008,'LIFT':-.025,'RETREAT':-.08}.get(phase,-.002)
            ok=arm.solve((last_uv[0], last_uv[1]-cfg.height_m/2), n,
                         np.deg2rad(last_action['blade_angle_deg']))
        elif phase in ('LOAD_APPROACH','DISPENSE','TOOL_INSPECT','SCAN_RETURN','SCAN'):
            arm.home(.35)
        # Keep every fourth work sample; phase boundaries are always retained.
        if phase=='WORK_STEP' and ev.get('sample',0)%4: continue
        wall=np.asarray(ev['wall']); blade=np.asarray(ev['blade'])
        frame={'phase':phase,'cycle':ev['cycle'],'control_steps':ev['control_steps'],
                       'arm':np.round(arm.points(),5),'q_deg':np.round(np.rad2deg(arm.q),2),
                       'tool_uv':np.round(last_uv,5),'kinematic_ok':ok,
                       'wall_mm':np.round(wall*1000,3),'blade_mm':np.round(blade*1000,3),
                       'metrics':ev['metrics'],'action':last_action}
        # Preserve visible continuity between skill endpoints. These are FK-valid
        # joint interpolation frames, explicitly labelled rather than hidden cuts.
        joint_jump=float(np.max(np.abs(np.rad2deg(arm.q-display_q))))
        n_transit=min(36,max(1,int(np.ceil(joint_jump/4))))
        if frames and n_transit>1:
            for j in range(1,n_transit):
                alpha=j/n_transit; qi=(1-alpha)*display_q+alpha*arm.q
                middle={**frame,'phase':'TRANSIT','arm':np.round(arm.points(qi),5),
                        'q_deg':np.round(np.rad2deg(qi),2)}
                frames.append(middle)
        frames.append(frame)
        display_q=arm.q.copy()
    decisions=[x['action'] for x in env.events if x['phase']=='DECIDE' and 'action' in x]
    mode_counts={name:sum(a['mode']==name for a in decisions) for name in
                 ('DEPOSIT','REUSE','LEVEL','RESCAN','FINISH')}
    path_angles=[]
    for a in decisions:
        delta=np.asarray(a['end_uv_m'])-a['start_uv_m']
        if np.linalg.norm(delta)>1e-9: path_angles.append(float(np.rad2deg(np.arctan2(delta[1],delta[0]))))
    action_summary={'mode_counts':mode_counts,'path_angle_deg_min':min(path_angles,default=0),
                    'path_angle_deg_max':max(path_angles,default=0),
                    'non_vertical_fraction':float(np.mean([abs(np.asarray(a['end_uv_m'])[0]-a['start_uv_m'][0])>.005
                                                           for a in decisions]))}
    max_frame_delta=max((float(np.max(np.abs(np.asarray(frames[k]['q_deg'])-
                                              np.asarray(frames[k-1]['q_deg']))))
                         for k in range(1,len(frames))),default=0.)
    report={'title':'整片墙面连续强化学习闭环','evidence_level':'L1','hardware_motion':False,
            'policy':str(policy_path.resolve()),'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            'seed':seed,'return':total,'result':info,'action_summary':action_summary,
            'config':cfg.to_dict(),'frames':len(frames),
            'transit_frames':sum(f['phase']=='TRANSIT' for f in frames),
            'max_display_joint_delta_deg':max_frame_delta,
            'kinematic_fail_frames':sum(not f['kinematic_ok'] for f in frames),
            'note':'Policy and material states are actual rollout data. Arm poses are MuJoCo inverse-kinematic solutions; mortar is a reduced-order 2.5-D proxy.'}
    payload={'report':report,'frames':frames,'wall':{'width':cfg.width_m,'height':cfg.height_m,
             'shape':list(cfg.wall_shape)},'scene':{'wall_x':SceneConfig().wall_distance}}
    template=Path(__file__).with_name('player.html').read_text(encoding='utf-8')
    (out/'index.html').write_text(template.replace('__DATA__',json.dumps(_jsonable(payload),ensure_ascii=False,
                                        separators=(',',':')).replace('</','<\\/')),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(_jsonable(report),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2)); print(f'Open: {out / "index.html"}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=2209)
    a=p.parse_args();export(a.policy,a.out,a.seed)


if __name__=='__main__': main()
