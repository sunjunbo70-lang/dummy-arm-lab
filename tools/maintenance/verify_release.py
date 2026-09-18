"""Verify release hashes, extract runtime to a new path, and run offline checks."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--extract-to', type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    with archive.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    expected = archive.with_suffix('.zip.sha256').read_text().split()[0]
    if actual != expected:
        raise SystemExit('Archive SHA-256 mismatch')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('dummy-experiment/RELEASE_MANIFEST.json'))
        for row in manifest['files']:
            with z.open('dummy-experiment/' + row['path']) as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != row['sha256']:
                    raise SystemExit('Content mismatch: ' + row['path'])
        if args.extract_to:
            if manifest['kind'] != 'runtime':
                raise SystemExit('Only runtime can be execution-tested')
            dest = args.extract_to.resolve()
            if dest.exists():
                raise SystemExit('Extraction directory must not exist')
            for name in z.namelist():
                if not (dest / name).resolve().is_relative_to(dest):
                    raise SystemExit('Unsafe archive path')
            z.extractall(dest)
    report = {'archive': archive.name, 'sha256': actual, 'verified_files': len(manifest['files']), 'checks': [],
              'scope': 'Offline checks using current interpreter; not a clean install, Linux test or hardware test'}
    if args.extract_to:
        root = dest / 'dummy-experiment'
        for command in [
            ['tools/maintenance/verify_evidence.py'],
            ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
            ['tools/live_mujoco.py', '--smoke-test'],
            ['tools/gui/live_mujoco.py', '--smoke-test'],
            ['-m', 'dummy_loop', '--help'],
        ]:
            r = subprocess.run([sys.executable, *command], cwd=root, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
            report['checks'].append({'command': command, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
    reportpath = archive.with_suffix('.verification.json')
    reportpath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if any(r['returncode'] for r in report['checks']):
        raise SystemExit(f'FAILED: see {reportpath}')
    print(f'RELEASE_OK: {len(manifest["files"])} files; {len(report["checks"])} offline checks; {reportpath}')

if __name__ == '__main__':
    main()
