"""Drag a checkpoint onto the launcher; run a new full episode, then native replay."""
import sys,subprocess,datetime
from pathlib import Path
root=Path(__file__).resolve().parents[2]
if __name__=='__main__':
 stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out=root/'outputs/wall_cycle/v09_r12'/f'interactive_{stamp}'
 subprocess.run([sys.executable,'-m','dummy_loop.wall_cycle_v09.evaluate_run','--checkpoint',sys.argv[1],'--reach',str(root/'outputs/wall_cycle/v09_r12/reach_a1_001/reach.npz'),'--out',str(out),'--seed','60090','--task','edge_ridge','--record','--view'],check=True,cwd=root)
