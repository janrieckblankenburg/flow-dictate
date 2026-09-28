#!/bin/zsh
# One-time setup: create venv + install dependencies.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt
echo
echo "Setup done. Start with:  ./run.sh"
