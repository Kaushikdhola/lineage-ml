"""
Mutation test for token-xray: does every check catch a fault, and is every check caught by one?

Applies a fixed list of faults one at a time to a COPY of xray/ (the working tree is never touched),
runs the full test suite against each, and prints which tests caught each fault. Two kinds of fault:
  * source: a corruption of the instrument itself (readout, recorder, probe, closed forms, leak harness,
    runners, API, data-parallel launchers), applied as an exact text replacement;
  * library: a realistic bug injected into the installed transformers or TRL, through XRAY_FAULT
    (see xray/tests/conftest.py).

Exit status 0 only if the baseline passes, every fault is caught by at least one test, and every test
fails under at least one fault.

    node scripts/py.mjs scripts/mutate.py            # all faults (about 25 minutes on a laptop CPU)
    node scripts/py.mjs scripts/mutate.py readout-weight ga-reverted
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "src/token_xray/"

# (name, file under xray/, text to replace, replacement)  — source faults
SOURCE = [
    ("readout-weight", PKG + "readout.py", "wr.append(gr / rr)", "wr.append(gr / rr * 1.001)"),
    ("readout-cosine", PKG + "readout.py", "cr.append(1.0 if gn == 0.0 else gr / (gn * rr**0.5))", "cr.append(1.0)"),
    ("recorder-overwrites", PKG + "readout.py",
     "self.grads[tok] = flat_g[k].clone() if prev is None else prev + flat_g[k]", "self.grads[tok] = flat_g[k].clone()"),
    ("padding-sink-off", PKG + "readout.py", "self.pad_sq += float(flat_g[k].norm()) ** 2", "pass"),
    ("unit-gradient-mean", PKG + "readout.py", 'loss = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum")',
     'loss = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="mean")'),
    ("pair-first-target-only", PKG + "readout.py", "w = torch.linalg.lstsq(U, g.unsqueeze(1)).solution[:, 0]",
     "w = torch.linalg.lstsq(U[:, :1], g.unsqueeze(1)).solution[:, 0].repeat(U.shape[1])"),
    ("pair-unit-gradient", PKG + "readout.py", "logits = model(inputs_embeds=emb).logits[:, 0]",
     "logits = model(inputs_embeds=emb * 1.01).logits[:, 0]"),
    ("pair-identifiability-guard", PKG + "readout.py", "if U.shape[1] > U.shape[0] - 1:", "if U.shape[1] > U.shape[0]:"),
    ("row-readout-scale", PKG + "readout.py",
     "return {int(t): g[int(t)].detach().clone() for t in token_ids if int(t) != PAD_ID}",
     "return {int(t): 2 * g[int(t)].detach().clone() for t in token_ids if int(t) != PAD_ID}"),
    ("probe-attention-live", PKG + "probe.py", '    if mode == "aggregation":', "    if False:"),
    ("probe-leak-mode-dead", PKG + "probe.py", '    if mode == "aggregation":', "    if True:"),
    ("closed-form-mean-of-means", PKG + "intended.py", "out += [[N / (k * Ng)] * ni for ni in g]",
     "out += [[N / ((k + 1) * Ng)] * ni for ni in g]"),
    ("closed-form-first-microbatch", PKG + "intended.py",
     "return [[1.0 if i < micro else 0.0] * ni for i, ni in enumerate(n)]",
     "return [[1.0 if i <= micro else 0.0] * ni for i, ni in enumerate(n)]"),
    ("closed-form-dpo-sign", PKG + "intended.py", "k = [beta * _sigmoid(-beta * d) / P for d in deltas]",
     "k = [beta * _sigmoid(beta * d) / P for d in deltas]"),
    ("closed-form-no-clip", PKG + "intended.py",
     "clipped = (A[i] > 0 and r > 1 + eh) or (A[i] < 0 and r < 1 - epsilon_low)", "clipped = False"),
    ("closed-form-std-ddof0", PKG + "intended.py", "sd = math.sqrt(sum(x * x for x in c) / (G - 1))",
     "sd = math.sqrt(sum(x * x for x in c) / G)"),
    ("closed-form-per-rank", PKG + "intended.py", "return [[N / (world * per[r])] * ni for ni, r in zip(n, rank_of)]",
     "return [[N / (world * per[r] + 1)] * ni for ni, r in zip(n, rank_of)]"),
    ("closed-form-multiplicity", PKG + "intended.py", "return [[c * N / seen] * ni for c, ni in zip(copies, n)]",
     "return [[N / seen] * ni for c, ni in zip(copies, n)]"),
    ("closed-form-pad-eos", PKG + "intended.py", "return [[N / (N - B)] * (ni - 1) + [0.0] for ni in n]",
     "return [[1.0] * (ni - 1) + [0.0] for ni in n]"),
    ("leak-labels-all-documents", PKG + "leak.py", "                lab[i] = -100", "                pass"),
    ("leak-boundary-target", PKG + "leak.py", "    lab[0, start] = -100", "    pass"),
    ("custom-loop-ignores-cache", PKG + "scenarios.py", 'kw = {} if use_cache is None else {"use_cache": use_cache}',
     'kw = {"use_cache": False}'),
    ("report-requires-legacy", PKG + "scenarios.py",
     "*([run_legacy_ga()] if require_legacy or legacy_available() else []),", "run_legacy_ga(),"),
    ("report-builds-trl", PKG + "cli.py", '"leak": {"panels": leak_panels()}}',
     '"leak": {"panels": leak_panels()}, "trl": __import__("token_xray.scenarios").scenarios.trl_section()}'),
    ("side-path-exposes-site-packages", PKG + "scenarios.py",
     'os.symlink(Path(__file__).resolve().parent, os.path.join(d, "token_xray"))\n    return d',
     'return str(Path(__file__).resolve().parents[1])'),
    ("legacy-runs-current-library", PKG + "scenarios.py",
     'return os.environ.get("XRAY_HF445_PYTHON", str(Path(__file__).resolve().parents[3] / ".venv-hf445/bin/python"))',
     'return __import__("sys").executable'),
    ("reference-normaliser", PKG + "runners.py", '                reduction="sum",\n            ) / N\n            loss.backward()',
     '                reduction="sum",\n            ) / (N + 1)\n            loss.backward()'),
    ("left-padding-labels", PKG + "runners.py",
     "ids.append(pad + x), att.append([0] * len(pad) + a), lab.append([-100] * len(pad) + y)",
     "ids.append(pad + x), att.append([0] * len(pad) + a), lab.append(pad + y)"),
    ("api-tolerance", PKG + "api.py", "if dev > tol:", "if dev > tol * 1e9:"),
    ("sft-padding-free-off", PKG + "trl_runners.py",
     "kw = dict(padding_free=padding_free, packing=packing, max_length=max_length, bf16=bf16)",
     "kw = dict(padding_free=False, packing=packing, max_length=max_length, bf16=bf16)"),
    ("sft-leak-labels-everywhere", PKG + "trl_runners.py",
     'row["labels"] = list(s) if i == labels_only else [-100] * len(s)', 'row["labels"] = list(s)'),
    ("dpo-margin-sign", PKG + "trl_runners.py", "deltas.append(float((lc - rc) - (lr - rr)))",
     "deltas.append(float((lc - rc) + (lr - rr)))"),
    ("grpo-ratios-swapped", PKG + "trl_runners.py", "new, old = probe_at(new_state, pb), probe_at(old_state, pb)",
     "old, new = probe_at(new_state, pb), probe_at(old_state, pb)"),
    ("ddp-per-rank-mean", PKG + "ddp.py", "loss = ce / sum(n_all) * world", "loss = ce / n_local"),
    ("ddp-sampler-drops-last", PKG + "ddp.py", "shuffle=False, drop_last=False)", "shuffle=False, drop_last=True)"),
    ("ddp-split-swapped", PKG + "ddp.py", "return [list(s) for s in split]", "return [list(s) for s in reversed(split)]"),
    ("trainer-ddp-never-averages", PKG + "ddp_trainer_entry.py",
     'average_tokens_across_devices=spec["average_tokens_across_devices"],', "average_tokens_across_devices=False,"),
    ("trainer-ddp-always-averages", PKG + "ddp_trainer_entry.py",
     'average_tokens_across_devices=spec["average_tokens_across_devices"],', "average_tokens_across_devices=True,"),
]

# library faults, by XRAY_FAULT name (xray/tests/conftest.py)
LIBRARY = ["ga-reverted", "label-smoothing-off", "sft-boundary-targets", "sft-normaliser-1pct", "dpo-normalizer",
           "grpo-clip-removed", "cache-never-created", "collator-keeps-pad-labels"]

FAILED = re.compile(r"^(FAILED|ERROR) (tests/\S+)")


def python() -> str:
    venv = ROOT / ".venv/bin/python"
    return str(venv) if venv.exists() else sys.executable


def run_suite(copy: Path, fault_env: str = "") -> tuple[set[str], int]:
    env = {**os.environ, "PYTHONPATH": str(copy / "src"),
           "XRAY_HF445_PYTHON": os.environ.get("XRAY_HF445_PYTHON", str(ROOT / ".venv-hf445/bin/python"))}
    env.pop("XRAY_FAULT", None)
    if fault_env:
        env["XRAY_FAULT"] = fault_env
    p = subprocess.run([python(), "-m", "pytest", "-q", "-rfE", "--tb=no", "-p", "no:cacheprovider", "tests"],
                       cwd=copy, env=env, capture_output=True, text=True)
    failed = {m.group(2).split(" ")[0] for line in p.stdout.splitlines() if (m := FAILED.match(line))}
    return failed, p.returncode


def collect(copy: Path) -> list[str]:
    env = {**os.environ, "PYTHONPATH": str(copy / "src")}
    p = subprocess.run([python(), "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", "tests"],
                       cwd=copy, env=env, capture_output=True, text=True)
    return [line.strip() for line in p.stdout.splitlines() if "::" in line]


def main(argv: list[str]) -> int:
    only = set(argv)
    with tempfile.TemporaryDirectory() as d:
        copy = Path(d) / "xray"
        shutil.copytree(ROOT / "xray", copy, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info"))
        tests = collect(copy)
        t0 = time.time()
        base, code = run_suite(copy)
        if code != 0 or base:
            print(f"baseline is not green ({sorted(base)}); fix the suite first")
            return 2
        print(f"baseline: {len(tests)} tests pass ({time.time() - t0:.0f} s per run)\n")

        caught_by: dict[str, set[str]] = {}
        for name, rel, old, new in SOURCE:
            if only and name not in only:
                continue
            path = copy / rel
            text = path.read_text()
            if text.count(old) != 1:
                print(f"STALE fault {name}: target text found {text.count(old)} times in {rel}")
                return 2
            path.write_text(text.replace(old, new))
            try:
                caught_by[name] = run_suite(copy)[0]
            finally:
                path.write_text(text)
            print(f"  {name:32s} source   caught by {len(caught_by[name]):2d}: {', '.join(sorted(caught_by[name]))[:150]}")
        for name in LIBRARY:
            if only and name not in only:
                continue
            caught_by[name] = run_suite(copy, name)[0]
            print(f"  {name:32s} library  caught by {len(caught_by[name]):2d}: {', '.join(sorted(caught_by[name]))[:150]}")

    missed = [f for f, c in caught_by.items() if not c]
    covered = set().union(*caught_by.values()) if caught_by else set()
    never = [t for t in tests if t not in covered]
    print(f"\nfaults caught: {len(caught_by) - len(missed)}/{len(caught_by)}")
    if missed:
        print("NOT caught:", ", ".join(missed))
    if not only:
        print(f"tests that caught at least one fault: {len(tests) - len(never)}/{len(tests)}")
        for t in never:
            print("  never failed:", t)
    return 0 if not missed and (only or not never) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
