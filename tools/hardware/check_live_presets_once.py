"""One explicitly requested, supervised home/fold check using GUI controller."""
import json
import argparse
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dummy_loop.live_control import LiveController, HOME

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Execute the supervised physical home/fold cycle')
    if not parser.parse_args().execute:
        parser.error('Physical motion requires --execute')
    path=ROOT/'outputs'/f'preset_recheck_{time.time_ns()}'
    c=LiveController(str(path.with_suffix('.jsonl')))
    report={'stages':[]}
    def tick():
        c.heartbeat()
        c.expire_feedback()
        s=c.snapshot()
        if not c.thread.is_alive(): raise RuntimeError(s['status'])
        time.sleep(.05)
        return s
    try:
        c.connect()
        deadline=time.monotonic()+20
        while not c.snapshot()['connected']:
            tick()
            if time.monotonic()>deadline: raise TimeoutError('Connection timeout')
        report['initial_deg']=c.snapshot()['q'].tolist()
        c.set_speed(5)
        for kind,target in [('home',HOME),('fold',np.array([0,-75,180,0,0,0]))]:
            print('BEGIN '+kind,flush=True)
            c.request_pose(kind)
            deadline=time.monotonic()+150
            last=0
            while True:
                s=tick()
                if not s['connected']: raise RuntimeError(s['status'])
                now=time.monotonic()
                if now-last>3:
                    print(json.dumps({'pose':kind,'q':np.round(s['q'],3).tolist(),'progress':s['auto_progress'],'age':round(now-s['rx_at'],3)},ensure_ascii=False),flush=True)
                    last=now
                if '完成' in s['auto_progress'] and not s['active'] and not s['automatic']:
                    error=float(np.max(np.abs(s['q']-target)))
                    if error>=.15: raise RuntimeError('Final feedback mismatch')
                    stage={'pose':kind,'final_deg':s['q'].tolist(),'max_error_deg':error}
                    report['stages'].append(stage)
                    print('COMPLETE '+json.dumps(stage),flush=True)
                    break
                if now>deadline: raise TimeoutError(kind+' timeout')
        report['result']='both_presets_feedback_reached'
    except BaseException as exc:
        report['error']=str(exc)
        raise
    finally:
        c.close()
        if c.thread: c.thread.join(5)
        report['worker_exited']=not c.thread.is_alive()
        report['final_status']=c.snapshot()['status']
        path.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('REPORT '+str(path.with_suffix('.report.json')),flush=True)

if __name__=='__main__': main()
