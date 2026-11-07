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
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r python/requirements.txt
.venv/bin/python -m playwright install chromium

echo "→ 3/4  Agent skills for design, three.js and visual QA (.claude/skills, not committed)"
fetch_skills() { # repo, subdir, names...
  local repo="$1" sub="$2"; shift 2
  local tmp; tmp="$(mktemp -d)"
  if git clone -q --depth 1 --filter=blob:none --sparse "https://github.com/$repo" "$tmp" 2>/dev/null; then
    (cd "$tmp" && git sparse-checkout set "$sub") && mkdir -p .claude/skills
    for name in "$@"; do
      if [ -d "$tmp/$sub/$name" ]; then rm -rf ".claude/skills/$name"; cp -R "$tmp/$sub/$name" .claude/skills/; echo "   ✓ $name"; fi
    done
  else
    echo "   ! could not fetch $repo (skipping)"
  fi
  rm -rf "$tmp"
}
fetch_skills anthropics/skills skills frontend-design webapp-testing
fetch_skills cloudai-x/threejs-skills skills threejs-fundamentals threejs-geometry threejs-materials \
  threejs-shaders threejs-postprocessing threejs-interaction threejs-performance threejs-animation

echo "→ 4/4  Verifying the engine against the oracle"
npm run verify

echo ""
echo "✓ Ready. Run: npm run dev"
