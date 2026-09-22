"""Static safety and inventory checks for categorized Windows launchers."""
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ROOT / 'launchers'


class LauncherContracts(unittest.TestCase):
    def test_catalog_entries_exist_and_are_unique(self):
        catalog=json.loads((LAUNCHERS/'catalog.json').read_text(encoding='utf-8'))
        listed=[]
        for category,names in catalog['categories'].items():
            for name in names:
                path=LAUNCHERS/category/name
                self.assertTrue(path.is_file(),path)
                listed.append(path.resolve())
        self.assertEqual(len(listed),len(set(listed)))

    def test_nonhardware_launchers_cannot_enable_hardware(self):
        for category in ('simulation','training','replay','archive'):
            for path in (LAUNCHERS/category).glob('*.cmd'):
                text=path.read_text(encoding='utf-8-sig',errors='replace').lower()
                self.assertNotIn('--enable',text,path)
                self.assertNotIn('serial_backend',text,path)

    def test_python_module_entries_resolve(self):
        for category in ('simulation','training','replay'):
            for path in (LAUNCHERS/category).glob('*.cmd'):
                text=path.read_text(encoding='utf-8-sig',errors='replace')
                pieces=text.split('-m ')[1:]
                for piece in pieces:
                    module=piece.split()[0].strip('"')
                    target=ROOT.joinpath(*module.split('.'))
                    self.assertTrue(target.with_suffix('.py').is_file() or
                                    (target/'__main__.py').is_file(),f'{path}: {module}')


if __name__ == '__main__': unittest.main()
