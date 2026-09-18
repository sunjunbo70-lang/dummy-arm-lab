#!/usr/bin/env bash
# Run from Ubuntu terminal after copying the extracted project.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/gui.txt
.venv/bin/python -m unittest discover -s tests -v
printf '\nNext: source .venv/bin/activate\nThen follow README.md\n'
