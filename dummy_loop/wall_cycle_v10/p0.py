"""Controlled P0 material counterfactuals. Does not train or command hardware."""
import argparse, concurrent.futures, hashlib, json, math, os, subprocess, time, zipfile
from pathlib import Path
import numpy as np
from ..wall_cycle_v09.config import config
from ..wall_cycle_v09.reward_quality import quality
from ..wall_cycle.mortar import MortarSystem, StrokeStats
ROOT=Path(__file__).resolve().parents[2]

def make_scene(index):
    c=config();m=MortarSystem(c,270000+index);m.reset();rng=np.random.default_rng(270000+index)
    r,s=m._score_rows,m._score_cols;u,v=np.meshgrid(m.u[s],m.v[r]);h=m.wall[r,s]
    amplitude=float(rng.uniform(.0015,.0035));width=float(rng.uniform(.006,.012));share=float(rng.uniform(.6,.9))
    if index<24:
        kind='edge_ridge';side=index%4;offset=float(rng.uniform(.082,.092))
        axis=(u if side<2 else v-.25);sign=1 if side%2 else -1
        h[:]=.002+amplitude*np.exp(-.5*((axis-sign*offset)/width)**2)
        center=np.array([sign*offset,.25]) if side<2 else np.array([0.,.25+sign*offset])
        direction=np.array([0.,1.]) if side<2 else np.array([1.,0.])
    elif index<48:
        kind='high_spot';center=np.array([rng.uniform(-.055,.055),.25+rng.uniform(-.055,.055)])
        h[:]=.002+amplitude*np.exp(-.5*(((u-center[0])/(width*1.5))**2+((v-center[1])/width)**2))
        direction=np.array([0.,1.]) if index%2 else np.array([1.,0.])
    else:
        kind='bare';h[:]=0.;center=np.array([0.,.25]);direction=np.array([0.,1.])
    m.initial_m3=float(m.wall.sum()*m.wall_cell_area);m.p.lift_wall_fraction=share
    return m,dict(index=index,seed=270000+index,kind=kind,amplitude_m=amplitude,width_m=width,share=share,center=center.tolist(),direction=direction.tolist())

def measures(m,initial):
    r,s=m._score_rows,m._score_cols;h=m.wall[r,s];before=initial[r,s];good=(before>=.0015)&(before<=.0025)
    excess=float(np.maximum(h-.0025,0).sum()*m.wall_cell_area*1e6)
    damage=float((good&((h<.0015)|(h>.0025))).sum()/max(good.sum(),1))
    ledger=m.initial_m3+m.supplied_m3-m.wall.sum()*m.wall_cell_area-m.blade_volume_m3-m.dropped_m3-m.outside_m3
    return {**quality(m), 'excess_ml':excess,'damage_fraction_initial_good':damage,'blade_ml':m.blade_volume_m3*1e6,'supplied_ml':m.supplied_m3*1e6,'dropped_ml':m.dropped_m3*1e6,'outside_ml':m.outside_m3*1e6,'ledger_error_ml':float(ledger*1e6)}

def run_case(job,keep=False):
    index,load,length,spacing,pitch_deg,force=job;m,scene=make_scene(index);c=m.cfg;initial=m.wall.copy();before=measures(m,initial)
    feed=None
    if load:feed=m.feed(load,c.feed_normal_force_N,c.feed_scoop_depth_m,c.feed_scoop_distance_m,c.feed_scoop_speed_m_s)
    direction=np.array(scene['direction']);center=np.array(scene['center']);start=center-direction*length/2;end=center+direction*length/2
    phi=float(math.atan2(direction[1],direction[0])-math.pi/2);pitch=math.radians(pitch_deg)
    n=int(math.ceil(length/spacing))+1;c.stroke_samples=n;t=time.perf_counter();stats=StrokeStats()
    m.begin_stroke(phi,direction);m.air(pitch,stats)
    for a in np.linspace(0,1,n):m.contact(start+(end-start)*a,pitch,.06,force_N=force,stats=stats)
    m.end_stroke(stats);m.air(0.,stats);after=measures(m,initial)
    reduction=(before['excess_ml']-after['excess_ml'])/max(before['excess_ml'],1e-12)
    result={'scene':scene,'requested_ml':load,'length_m':length,'spacing_m':spacing,'pitch_deg':pitch_deg,'force_N':force,'speed_m_s':.06,'start':start.tolist(),'end':end.tolist(),'phi':phi,'contact_samples':n,'before':before,'after':after,'feed':feed,'excess_reduction_fraction':reduction,'quality_improved':bool(after['J']<before['J']),'material_gate':bool(before['excess_ml']>0 and reduction>=.2 and after['damage_fraction_initial_good']<=.02),'wall_seconds':time.perf_counter()-t,'evidence':'nominal material only; reach/physical execution not established'}
    if keep:return result,initial,m.wall.copy(),m.blade.copy()
    return result

def snapshot(out):
    git=['git','-c',f'safe.directory={ROOT.as_posix()}','-C',str(ROOT)]
    (out/'source_diff.patch').write_bytes(subprocess.check_output(git+['diff','HEAD','--binary']))
    (out/'git_status.txt').write_bytes(subprocess.check_output(git+['status','--short']))
    hashes={}
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for parent in [ROOT/'dummy_loop',ROOT/'models',ROOT/'configs']:
            for f in parent.rglob('*'):
                if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.json','.xml','.yaml','.yml'):
                    data=f.read_bytes();name=f.relative_to(ROOT).as_posix();hashes[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    return subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);commit=snapshot(a.out)
    protocol={'stage':'P0_nominal_counterfactual','source_head':commit,'seed_base':270000,'scenes':60,'loads_ml':[0,6,12,18],'lengths_m':[.06,.12,.20],'spacing_m':.0025,'pitch_deg':4,'force_N':2,'workers':a.workers,'gate':{'fraction':.8,'excess_reduction':.2,'max_initial_good_damage':.02},'logging':'scalar per branch; initial/final arrays for representative branches only','status':'running','started_utc':time.time()}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    # Save initial scene definitions before testing; no adaptation to results.
    (a.out/'scenes.json').write_text(json.dumps([make_scene(i)[1] for i in range(60)],indent=2),encoding='utf8')
    np.savez_compressed(a.out/'initial_scenes.npz',walls=np.array([make_scene(i)[0].wall for i in range(60)]))
    jobs=[(i,q,L,.0025,4.,2.) for i in range(60) for q in [0.,6.,12.,18.] for L in [.06,.12,.20]]
    rows=[];t=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool, (a.out/'cases.jsonl').open('w',encoding='utf8',buffering=1) as f:
        for result in pool.map(run_case,jobs,chunksize=3):
            rows.append(result);f.write(json.dumps(result)+'\n')
            if len(rows)%60==0:print('cases',len(rows),'/',len(jobs),'seconds',round(time.perf_counter()-t,1),flush=True)
    bykind={}
    for kind,inds in [('edge_ridge',range(24)),('high_spot',range(24,48)),('bare',range(48,60))]:
        valid=[r for r in rows if r['scene']['index'] in inds];passed=sum(any(r['material_gate'] and r['requested_ml']==0 for r in valid if r['scene']['index']==i) for i in inds)
        bykind[kind]={'scenes':len(inds),'empty_tool_gate_scenes':passed,'fraction':passed/len(inds),'by_length':[{'length_m':L,'mean_delta_J':float(np.mean([r['after']['J']-r['before']['J'] for r in valid if r['requested_ml']==0 and r['length_m']==L]))} for L in [.06,.12,.20]]}
    result={'status':'nominal_complete','cases':len(rows),'wall_seconds':time.perf_counter()-t,'by_task':bykind,'max_abs_ledger_error_ml':max(abs(r['after']['ledger_error_ml']) for r in rows),'empty_tool_material_gate_pass':bykind['edge_ridge']['fraction']>=.8,'physical_gate':'pending','training_started':False,'note':'Passing material gate does not establish robot executability.'}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

