"""Experiment resource telemetry. Run with a psutil-equipped Python, no CUDA context."""
import argparse, json, os, subprocess, time
from pathlib import Path
import psutil

def monitor(out, pid, interval=5):
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    root=psutil.Process(pid); identity=root.create_time(); procs={}; psutil.cpu_percent(None)
    with out.open('a',encoding='utf-8',buffering=1) as f:
        while True:
            try:
                if not root.is_running() or root.create_time()!=identity: break
                children=[root]+root.children(recursive=True)
            except psutil.Error: break
            row={'utc_s':time.time(),'root_pid':pid,'cpu_system_pct':psutil.cpu_percent(None),
                 'cpu_per_core_pct':psutil.cpu_percent(None,percpu=True),'memory':psutil.virtual_memory()._asdict(),
                 'disk_io':psutil.disk_io_counters()._asdict(),'processes':[]}
            for p in children:
                try:
                    key=(p.pid,p.create_time()); p=procs.setdefault(key,p)
                    row['processes'].append({'pid':p.pid,'name':p.name(),'cpu_pct_one_core':p.cpu_percent(None),
                        'rss_bytes':p.memory_info().rss,'threads':p.num_threads(),'io':p.io_counters()._asdict()})
                except psutil.Error: pass
            for name,args in [('gpu',['--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu','--format=csv,noheader,nounits']),
                              ('gpu_processes',['--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'])]:
                try:
                    r=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=4,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    row[name]={'returncode':r.returncode,'csv':r.stdout.strip(),'error':r.stderr.strip()}
                except Exception as e: row[name]={'error':str(e)}
            f.write(json.dumps(row)+'\n'); time.sleep(interval)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--interval',type=float,default=5)
    a=p.parse_args();monitor(a.out,a.pid,a.interval)
