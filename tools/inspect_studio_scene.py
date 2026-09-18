"""Compatibility entrypoint; implementation: tools/diagnostics/inspect_studio_scene.py."""
from pathlib import Path
import runpy
import sys
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))
globals().update(runpy.run_path(str(_ROOT/"tools/diagnostics/inspect_studio_scene.py"),run_name=__name__))
