"""Compatibility entry for current handoff packages; Linux remains unverified."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    print('Building current runtime/history packages. See docs/SETUP.md.')
    runpy.run_path(str(Path(__file__).resolve().parent / 'maintenance/package_release.py'), run_name='__main__')
