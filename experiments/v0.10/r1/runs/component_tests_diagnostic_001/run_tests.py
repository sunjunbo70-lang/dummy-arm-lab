import faulthandler,unittest
from pathlib import Path
p=Path(__file__).parent
with (p/'trace.log').open('w') as trace, (p/'tests.log').open('w') as log:
 faulthandler.enable(trace);faulthandler.dump_traceback_later(60,repeat=True,file=trace)
 suite=unittest.defaultTestLoader.discover('D:/VLA/dummy_arm/dummy_loop/wall_cycle_v10/r1',pattern='test_*.py',top_level_dir='D:/VLA/dummy_arm')
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 faulthandler.cancel_dump_traceback_later()
 (p/'result.txt').write_text(str(dict(tests=result.testsRun,errors=len(result.errors),failures=len(result.failures))))
