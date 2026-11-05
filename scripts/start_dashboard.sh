#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  echo 'Missing project .venv. See docs/dashboard_demo.md for setup.' >&2
  exit 1
fi
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/launch_dashboard.py" "$@"
