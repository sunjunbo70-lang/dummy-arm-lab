"""Immutable source/dependency identity for each experimental run."""
import hashlib,json,subprocess,zipfile
from pathlib import Path

def capture(out):
 root=Path(__file__).resolve().parents[2];out=Path(out);out.mkdir(parents=True,exist_ok=True);hashes={}
 with zipfile.ZipFile(out/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
  for folder in ('dummy_loop/wall_cycle','dummy_loop/wall_cycle_v09'):
   for p in (root/folder).rglob('*'):
    if p.is_file() and '__pycache__' not in str(p):
     rel=str(p.relative_to(root));hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest();z.write(p,rel)
 cmd=['git','-c',f'safe.directory={root.as_posix()}','-C',str(root)]
 head=subprocess.run(cmd+['rev-parse','HEAD'],capture_output=True,text=True).stdout.strip();diff=subprocess.run(cmd+['diff','--binary'],capture_output=True).stdout;(out/'working_tree.patch').write_bytes(diff)
 (out/'source_manifest.json').write_text(json.dumps({'head':head,'files':hashes},indent=2),encoding='utf8')
