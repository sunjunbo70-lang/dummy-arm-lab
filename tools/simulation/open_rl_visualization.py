"""Open the offline PPO player, rendering saved trained policies if needed."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rebuild', action='store_true')
    a = p.parse_args()
    base = ROOT / 'outputs/wall'
    candidates = sorted(base.glob('ppo_visualization_*/index.html'), key=lambda f: f.stat().st_mtime)
    if a.rebuild or not candidates:
        out = base / ('ppo_visualization_' + time.strftime('%Y%m%d_%H%M%S'))
        subprocess.run([sys.executable, '-m', 'dummy_loop.wall.visualize', '--out', str(out)], cwd=ROOT, check=True)
        target = out / 'index.html'
    else:
        target = candidates[-1]
    print(target)
    webbrowser.open(target.as_uri())


if __name__ == '__main__':
    main()
