#!/usr/bin/env bash
# One-time setup:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

need() { command -v "$1" >/dev/null 2>&1 || { echo "✗ Missing $1: $2"; exit 1; }; }
need node "install Node 20+ from https://nodejs.org"
need python3 "install Python 3.10+ from https://python.org"
need git "install git"
node -e 'process.exit(+process.versions.node.split(".")[0] >= 20 ? 0 : 1)' || { echo "✗ Node 20+ required (found $(node -v))"; exit 1; }

echo "→ 1/4  JavaScript dependencies"
npm ci --no-audit --no-fund

echo "→ 2/4  Python environment (.venv) for the oracle and browser testing"
# Pick the newest Python available (override with PYTHON=/path/to/python). Python 3.10.0 has a
# dataclass(slots=True) bug that crashes pytest >= 9.2, so anything older than 3.10.2 is rejected.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
ok_py() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10, 2) else 1)' 2>/dev/null; }
ok_py "$PY" || { echo "✗ $("$PY" -V 2>&1) is too old. Install Python 3.11+ (e.g. brew install python@3.12) or set PYTHON=/path/to/python"; exit 1; }
if [ -x .venv/bin/python ] && ! ok_py .venv/bin/python; then
  echo "   rebuilding .venv: $(.venv/bin/python -V 2>&1) is too old"
  rm -rf .venv
fi
[ -x .venv/bin/python ] || "$PY" -m venv .venv
echo "   using $(.venv/bin/python -V)"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r python/requirements.txt
.venv/bin/python -m playwright install chromium

echo "→ 4/4  Verifying the engine against the oracle"
npm run verify

echo "→ token-xray: CPU PyTorch, transformers and TRL in .venv (a few minutes the first time)"
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -e "xray[dev,trl]"

echo "→ token-xray: transformers 4.45.2 (before the Oct 2024 accumulation fix) in .venv-hf445, for the hero plate"
[ -x .venv-hf445/bin/python ] || "$PY" -m venv .venv-hf445
.venv-hf445/bin/pip install -q --upgrade pip
.venv-hf445/bin/pip install -q -r xray/requirements-hf445.txt
node scripts/xray.mjs test

echo ""
echo "✓ Ready. Run: npm run dev"
