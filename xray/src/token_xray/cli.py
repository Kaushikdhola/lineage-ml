"""token-xray command line.

    token-xray report                  print the frozen scenarios as a table
    token-xray export public/xray.json write measured results for the website
    token-xray legacy-ga out.json      measure the gradient-accumulation step with the INSTALLED
                                       transformers (run in .venv-hf445 for the pre-fix 4.45.2 Trainer)
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="token-xray")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    e = sub.add_parser("export")
    e.add_argument("out")
    g = sub.add_parser("legacy-ga")
    g.add_argument("out")
    a = p.parse_args(argv)

    if a.cmd == "legacy-ga":
        from .scenarios import legacy_ga_panel

        with open(a.out, "w") as f:
            json.dump(legacy_ga_panel(), f)
        return 0

    if a.cmd == "export":
        from .scenarios import export

        data = export()  # the site's data: every panel, including the 4.45.2 run, or an error
        with open(a.out, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"wrote {a.out}")
        return 0

    # report: only what it prints, so a plain `pip install token-xray-ml` (no TRL, no side environment) works
    from .scenarios import _versions, aggregation_panels, leak_panels, legacy_available, legacy_python

    data = {"versions": _versions(), "aggregation": {"panels": aggregation_panels(require_legacy=False)},
            "leak": {"panels": leak_panels()}}
    print("versions:", data["versions"])
    print("\nAggregation (max |w*N - intended|, min cosine)")
    for pnl in data["aggregation"]["panels"]:
        print(f"  [{pnl['kind']:9s}] {pnl['title']:52s} {pnl['maxDeviation']:.2e}  {pnl['minCos']:.6f}")
    print("\nLeaks (max off-diagonal)")
    for pnl in data["leak"]["panels"]:
        print(f"  [{pnl['kind']:9s}] {pnl['title']:52s} {pnl['maxLeak']:.2e}")
    if not legacy_available():
        print("\nSkipped the transformers 4.45.2 run (the Trainer before the October 2024 fix). It needs a separate\n"
              "Python: pip install transformers==4.45.2 accelerate==0.34.2 torch==2.4.1 'numpy<2.1', then set\n"
              f"XRAY_HF445_PYTHON to that interpreter (looked for {legacy_python()}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
