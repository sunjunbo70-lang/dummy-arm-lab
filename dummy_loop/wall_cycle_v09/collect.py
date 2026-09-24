"""v0.9 collection entry point. 400 episodes with the approved task strata."""
import argparse,json,os,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from .workers import collect_teacher

def main():
 p=argparse.ArgumentParser();p.add_argument('--reach',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--episodes',type=int,default=400);p.add_argument('--buffer',type=float,default=.03);p.add_argument('--decisions',type=int,default=2000);p.add_argument('--controlled',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 from .provenance import capture
 capture(a.out/'config')
 subprocess.Popen([r'C:\Users\28017\anaconda3\python.exe','-m','dummy_loop.wall_cycle_v09.telemetry','--out',str(a.out/'resources.jsonl'),'--pid',str(os.getpid())],creationflags=0x08000000)
 tasks=(['bare','rough','edge_ridge','corner','high_low','finish']*20 if a.controlled else ['bare','rough']*75+['edge_ridge','corner','high_low']*50+['finish']*100)
 jobs=[(i,60200+i,tasks[i%len(tasks)],a.reach,str(a.out),a.buffer,a.decisions) for i in range(a.episodes)]
 with ProcessPoolExecutor(a.workers) as pool:
  for f in as_completed([pool.submit(collect_teacher,j) for j in jobs]):
   r=f.result()
   with (a.out/'episodes.jsonl').open('a',encoding='utf8') as log:log.write(json.dumps(r)+'\n')
   print(r['index'],r['samples'],r['final']['metrics']['coverage'],flush=True)
if __name__=='__main__':main()
