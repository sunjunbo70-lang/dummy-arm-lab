"""历史取证脚本：对一份原始资料目录做清单、文本提取与页面渲染。

这不是日常开发或实机运行的依赖。本脚本在 2026-09-16 的资料审计中使用过，
产物已归档在 docs/history/：文件清单.json、V3.0.7手册_文本提取.txt、
手册_p14.png、手册_p15.png。那些是历史证据，本脚本不再写入该目录。

原版本把扫描根目录硬编码为 D:/VLA，并直接写进 docs/history/。2026-09-18
工作区重构后该路径不再成立，现改为必须显式传入 --root，输出默认落到
outputs/inspect_sources/（不进版本控制，也不覆盖既有证据）。

重跑本脚本需要提供对应的上游资料目录；没有那份资料时它无事可做。

用法：
    python tools/diagnostics/inspect_sources.py --root <资料目录> inventory
    python tools/diagnostics/inspect_sources.py --root <资料目录> read <相对路径> [...]
    python tools/diagnostics/inspect_sources.py --root <资料目录> pdf <相对路径>
    python tools/diagnostics/inspect_sources.py --root <资料目录> render <相对路径> <页码> [...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, type=Path,
                        help='要检查的原始资料目录（原先硬编码为 D:/VLA）')
    parser.add_argument('--out', type=Path, default=REPO / 'outputs' / 'inspect_sources',
                        help='输出目录；默认 outputs/inspect_sources/，不写 docs/history/')
    parser.add_argument('mode', choices=['inventory', 'read', 'pdf', 'render'])
    parser.add_argument('args', nargs='*')
    return parser


def resolve_root(root):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f'ERROR: 资料目录不存在: {root}')
    return root


def safe(root, relative):
    """把相对路径解析到 root 下，拒绝越界访问。"""
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f'路径越出资料目录: {relative}')
    return p


def run_inventory(root, out):
    rows = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in {'.git', '__pycache__', 'node_modules'} and not d.startswith('.venv')]
        for f in files:
            p = Path(base) / f
            rows.append({'path': str(p.relative_to(root)), 'bytes': p.stat().st_size})
    (out / '文件清单.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    keep = ['agents.md', 'protocol', 'interface', 'hardware_interface',
            'robot.cpp', 'command', 'usb', 'readme', 'urdf', 'mjcf']
    skip = ['Drivers', 'Middlewares', 'esp32-iot', 'Dummy 源码注释']
    for r in rows:
        p = r['path']
        if any(s in p.lower() for s in keep) and not any(s in p for s in skip):
            print(p)
    print('TOTAL', len(rows))


def run_read(root, targets):
    for s in targets:
        p = safe(root, s)
        print('\nFILE', s)
        for i, line in enumerate(p.read_text(encoding='utf-8-sig', errors='replace').splitlines(), 1):
            print(f'{i}: {line}')


def run_pdf(root, out, relative):
    from pypdf import PdfReader
    reader = PdfReader(safe(root, relative))
    text = '\n'.join(f'\nPAGE {i + 1}\n' + (page.extract_text() or '')
                     for i, page in enumerate(reader.pages))
    (out / (Path(relative).stem + '_文本提取.txt')).write_text(text, encoding='utf-8')
    print(text)


def run_render(root, out, relative, pages):
    import subprocess
    source = safe(root, relative)
    for n in map(int, pages):
        target = out / f'{Path(relative).stem}_p{n}'
        subprocess.run(['pdftoppm', '-f', str(n), '-l', str(n), '-scale-to', '1500',
                        '-singlefile', '-png', str(source), str(target)], check=True)
        print(str(target) + '.png')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    args = build_parser().parse_args()
    root = resolve_root(args.root)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == 'inventory':
        run_inventory(root, out)
    elif args.mode == 'read':
        if not args.args:
            raise SystemExit('ERROR: read 模式需要至少一个相对路径')
        run_read(root, args.args)
    elif args.mode == 'pdf':
        if len(args.args) != 1:
            raise SystemExit('ERROR: pdf 模式需要且仅需要一个相对路径')
        run_pdf(root, out, args.args[0])
    elif args.mode == 'render':
        if len(args.args) < 2:
            raise SystemExit('ERROR: render 模式需要一个相对路径和至少一个页码')
        run_render(root, out, args.args[0], args.args[1:])


if __name__ == '__main__':
    main()
