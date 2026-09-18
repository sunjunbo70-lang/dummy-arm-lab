"""Compatibility entrypoint; implementation: tools/hardware/commission_native_j1.py."""
from pathlib import Path
import runpy
import sys
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))
globals().update(runpy.run_path(str(_ROOT/"tools/hardware/commission_native_j1.py"),run_name=__name__))
