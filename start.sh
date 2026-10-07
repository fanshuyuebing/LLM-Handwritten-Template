#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo '请先按 OFFLINE.md 恢复 Python 环境。'
  exit 1
fi
exec .venv/bin/python web/server.py
