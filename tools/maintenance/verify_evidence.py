"""Verify immutable experiment files against their SHA-256 manifest."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]

def main():
    rows = json.loads((ROOT / 'experiments/2026-09-16_baseline/manifest.json').read_text(encoding='utf-8'))
    errors = []
    for row in rows:
        p = ROOT / row['path']
        if not p.is_file() or p.stat().st_size != row['bytes']:
            errors.append(row['path'])
            continue
        with p.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != row['sha256']:
                errors.append(row['path'])
    if errors:
        raise SystemExit('FAILED: ' + '\n'.join(errors))
    print(f'EVIDENCE_OK: {len(rows)} files')

if __name__ == '__main__':
    main()
