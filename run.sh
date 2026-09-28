#!/bin/zsh
# Start Flow Dictate. ANTHROPIC_API_KEY comes from the environment or
# from a local .env file (KEY=value lines; needed when started via launchd).
cd "$(dirname "$0")"
if [ -f .env ]; then
  set -a; source .env; set +a
fi
exec .venv/bin/python -u dictate.py
