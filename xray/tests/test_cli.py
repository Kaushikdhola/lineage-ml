"""
The command a pip user runs first: `token-xray report` must work on a plain install, without TRL and
without the transformers 4.45.2 side environment (which only `export`, the site's data, requires).
"""
import os
import subprocess
import sys

from token_xray.scenarios import isolated_pythonpath

BLOCK_TRL = "import sys; sys.modules['trl'] = None; from token_xray.cli import main; sys.exit(main(['report']))"


def test_report_runs_without_trl_or_the_side_environment():
    env = {**os.environ, "XRAY_HF445_PYTHON": "/nonexistent/python"}
    p = subprocess.run([sys.executable, "-c", BLOCK_TRL], env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "Skipped the transformers 4.45.2 run" in p.stdout
    assert "4.45.2 Trainer" not in p.stdout  # the panel is left out, not faked
    assert "Packed, position_ids reset, model default use_cache" in p.stdout  # the rest still runs


def test_side_interpreter_sees_only_this_package(tmp_path):
    want = str(tmp_path / "path")
    d = isolated_pythonpath(want)
    assert d == want  # a fresh directory, never the package's own parent (site-packages for a pip install)
    assert sorted(os.listdir(d)) == ["token_xray"]
    assert os.path.exists(os.path.join(d, "token_xray", "scenarios.py"))
