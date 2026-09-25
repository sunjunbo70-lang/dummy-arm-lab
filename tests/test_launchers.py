"""Versioned experiment entry points: inventory and hardware separation."""
import json,re,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class LauncherContracts(unittest.TestCase):
 def entries(self):return json.loads((ROOT/'experiments/catalog.json').read_text(encoding='utf8'))
 def test_catalog_complete_and_unique(self):
  listed=[(ROOT/x['path']).resolve() for x in self.entries()]
  actual=[p.resolve() for p in (ROOT/'experiments').rglob('*.cmd') if 'scripts' in p.parts]
  self.assertEqual(len(listed),len(set(listed)));self.assertEqual(set(listed),set(actual))
  for p in listed:self.assertTrue(p.is_file())
 def test_nonhardware_entries_cannot_enable_hardware(self):
  for x in self.entries():
   if x['kind']=='hardware':continue
   s=(ROOT/x['path']).read_text(encoding='utf-8-sig').lower()
   self.assertNotIn('--enable',s);self.assertNotIn('serial_backend',s)
 def test_modules_and_repo_root(self):
  for x in self.entries():
   p=ROOT/x['path'];s=p.read_text(encoding='utf-8-sig')
   for module in re.findall(r'-m ([\w.]+)',s):
    target=ROOT.joinpath(*module.split('.'));self.assertTrue(target.with_suffix('.py').exists() or (target/'__main__.py').exists(),str(p))
   for suffix in re.findall(r'%~dp0((?:\.\.\\?)+)',s):
    self.assertEqual((p.parent/suffix.replace('\\','/')).resolve(),ROOT)
if __name__=='__main__':unittest.main()
