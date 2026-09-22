"""Run v0.6 structural gates. Software only; never imports a hardware backend."""
import argparse, json, time, sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from dummy_loop.wall_cycle.area import load_work_area
from dummy_loop.wall_cycle.arm import ArmExecutor
from dummy_loop.wall_cycle.config import CycleConfig
from dummy_loop.wall_cycle.env import WallCycleEnv


def config():
    return load_work_area(CycleConfig(
        physics='v0.6',tool_profile='lab_20260922',lift_wall_fraction=.75,
        base_steps=12000,max_steps=24000,extension_steps=2000,max_cycles=160,
        max_reload_cycles=60,stall_limit=1,stall_window=12,min_cycles_before_stall=25))


def run(seed, max_cycles=None):
    cfg=config();ex=ArmExecutor(cfg,seed=seed)
    env=WallCycleEnv(cfg,seed,initial_mix=False,record=True,executor=ex);env.teacher_style='technique'
    env.reset();ret=0.;plans=[];trace=[];done=False
    while not done and (max_cycles is None or env.cycles<max_cycles):
        a=env.teacher_action();d=env.decode(a);p=ex.plan(d)
        if p.ok:
            plans.append({'carry_min':p.carry_face_up_min,'approach':p.approach_distances_m})
        _,r,done,info=env.step(a);ret+=r
        for ev in env.events:
            if ev['phase']=='ARM_TRACE' and not ev.get('_counted'):
                trace.extend(ev['trajectory']);ev['_counted']=True
    carry=[x['face_up_score'] for x in trace if x['phase'] in ('FEED_ALIGN_UP','CARRY_FACE_UP')]
    rotate=[x['face_up_score'] for x in trace if x['phase']=='ROTATE_TO_WALL']
    monotonic=all(all(a>=b-5e-4 for a,b in zip(p['approach'],p['approach'][1:])) for p in plans)
    return {'seed':seed,'return':ret,'cycles':env.cycles,'done':done,
            'success':bool(info['success']),'coverage':info['metrics']['coverage'],
            'rmse_mm':info['metrics']['rmse_mm'],'waste_frac':info['metrics']['waste_frac'],
            'carry_frames':len(carry),'carry_min':float(min(carry or [1.])),
            'carry_over_10deg_fraction':float(np.mean(np.asarray(carry)<np.cos(np.deg2rad(10))) if carry else 0),
            'rotation_min':float(min(rotate or [1.])), 'monotonic_approach':monotonic,
            'volume_error_m3':abs(info['volume_balance_m3']),
            'unreachable':info['unreachable_strokes'],'projected':info['projected_strokes']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--episodes',type=int,default=5);ap.add_argument('--seed',type=int,default=41000)
    ap.add_argument('--max-cycles',type=int);a=ap.parse_args();t=time.time()
    rows=[]
    for i in range(a.episodes):
        row=run(a.seed+i,a.max_cycles);rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    gates={'no_face_down':bool(all(r['carry_min']>=np.cos(np.deg2rad(15)) for r in rows)),
           'face_up_995':bool(np.mean([1-r['carry_over_10deg_fraction'] for r in rows])>=.995),
           'monotonic_approach':all(r['monotonic_approach'] for r in rows),
           'mass_conservation':max(r['volume_error_m3'] for r in rows)<1e-9}
    result={'evidence_level':'L1','hardware_motion':False,'config':config().to_dict(),
            'episodes':rows,'gates':gates,'passed':all(gates.values()),'elapsed_s':round(time.time()-t,1)}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'gates':gates,'out':str(a.out)},ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 2)

if __name__=='__main__':main()
