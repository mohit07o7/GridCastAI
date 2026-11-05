#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  echo 'Missing project Python environment. See docs/command_center.md.' >&2
  exit 1
fi
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/launch_command_center.py" "$@"
