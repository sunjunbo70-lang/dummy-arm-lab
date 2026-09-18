"""Decode passive DummyStudio capture; never opens a serial port."""
import argparse
import base64
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tail', type=int, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / 'outputs/studio_wire_capture.tsv'
    lines = source.read_text(encoding='utf-8-sig').splitlines()
    decoded = []
    counts = Counter()
    for line in lines:
        if not line.strip():
            continue
        timestamp, direction, encoded = line.lstrip('\ufeff').split('\t', 2)
        payload = base64.b64decode(encoded, validate=True)
        counts[direction] += 1
        decoded.append(f'{timestamp} {direction:8} {payload!r}')
    (root / 'outputs/studio_wire_capture_decoded.txt').write_text(
        '\n'.join(decoded) + '\n', encoding='utf-8')
    print(dict(counts))
    print('\n'.join(decoded[-args.tail:]))


if __name__ == '__main__':
    main()
