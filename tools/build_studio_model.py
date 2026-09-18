"""Compatibility entrypoint; implementation: tools/modeling/build_studio_model.py."""
from pathlib import Path
import runpy
import sys
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))
globals().update(runpy.run_path(str(_ROOT/"tools/modeling/build_studio_model.py"),run_name=__name__))
