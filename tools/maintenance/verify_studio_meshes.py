"""校验 models/studio_meshes/ 是否齐全、是否与记录一致（逐文件 SHA-256）。只读，不改任何文件。

Studio 外观网格因上游无许可声明不随仓库分发。换机器后从原开发机复制
  <原基线>/models/studio_meshes/  →  本仓库 models/studio_meshes/
再运行：
  python tools/maintenance/verify_studio_meshes.py
全部一致时退出码 0；缺失或哈希不符时列出并以 1 退出。
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    prov = json.loads((ROOT / 'models' / 'studio_provenance.json').read_text(encoding='utf-8'))
    meshdir = ROOT / 'models' / 'studio_meshes'
    expected = {name: h for part in prov['parts'] for name, h in part.get('stl_sha256', {}).items()}
    missing, mismatch, ok = [], [], 0
    for name, h in sorted(expected.items()):
        f = meshdir / name
        if not f.is_file():
            missing.append(name); continue
        if hashlib.sha256(f.read_bytes()).hexdigest() != h:
            mismatch.append(name)
        else:
            ok += 1
    print(json.dumps({'meshdir': str(meshdir), 'expected': len(expected), 'ok': ok,
                      'missing': missing, 'hash_mismatch': mismatch}, ensure_ascii=False, indent=2))
    return 0 if not missing and not mismatch else 1


if __name__ == '__main__':
    sys.exit(main())
