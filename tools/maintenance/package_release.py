"""Build portable runtime and historical evidence archives without overwriting releases."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]
# 2026-09-18 重构后，历史资料不在基线内，而在与基线并列的 _archive/。
ARCHIVE = ROOT.parent / '_archive'
SKIP = {'__pycache__', '.git', '.venv-loop'}

def collect(folders, base=None):
    base = ROOT if base is None else base
    return [p for folder in folders for p in (base / folder).rglob('*')
            if p.is_file() and not SKIP.intersection(p.relative_to(base).parts) and p.suffix != '.pyc']

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='2026-09-18-workspace')
    parser.add_argument('--kind', choices=['runtime', 'history', 'both'], default='both')
    args = parser.parse_args()
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.tag):
        parser.error('tag must contain only letters, digits, hyphens or underscores')
    out = ROOT / 'releases'
    out.mkdir(exist_ok=True)
    targets = {kind: out / f'dummy-experiment-{kind}-{args.tag}.zip' for kind in (('runtime', 'history') if args.kind == 'both' else (args.kind,))}
    for p in targets.values():
        if p.exists():
            raise SystemExit(f'Refusing to overwrite {p}; use --tag')
    runtime = collect(['dummy_loop', 'configs', 'models', 'vendor', 'requirements', 'tools', 'tests', 'docs', 'experiments'])
    runtime += [p for p in ROOT.iterdir() if p.is_file() and (p.suffix in {'.md', '.cmd'} or p.name in {'requirements-loop.txt', '.gitignore', '.gitattributes'})]
    runtime += [ROOT / 'outputs' / n for n in ['README.md', 'policy.npz', 'teacher.npz', 'rollout.summary.json'] if (ROOT / 'outputs' / n).exists()]
    history = collect([d.name for d in ARCHIVE.iterdir() if d.is_dir()], base=ARCHIVE) \
        if ARCHIVE.is_dir() else []
    groups = {'runtime': runtime, 'history': history}
    for kind in targets:
        files = groups[kind]
        rows = []
        target = targets[kind]
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
            for p in sorted(set(files)):
                base = ROOT if kind == 'runtime' else ARCHIVE
                relative = p.relative_to(base).as_posix()
                rows.append({'path': relative, 'bytes': p.stat().st_size, 'sha256': digest(p)})
                z.write(p, 'dummy-experiment/' + relative, compress_type=zipfile.ZIP_STORED if p.suffix == '.zip' else zipfile.ZIP_DEFLATED)
            manifest = json.dumps({'kind': kind, 'tag': args.tag, 'files': rows}, ensure_ascii=False, indent=2)
            z.writestr('dummy-experiment/RELEASE_MANIFEST.json', manifest)
        target.with_suffix('.manifest.json').write_text(manifest, encoding='utf-8')
        target.with_suffix('.zip.sha256').write_text(f'{digest(target)}  {target.name}\n', encoding='ascii')
        print(f'{kind}: {len(rows)} files, {target.stat().st_size / 1024**2:.1f} MiB, {target}', flush=True)

if __name__ == '__main__':
    main()
