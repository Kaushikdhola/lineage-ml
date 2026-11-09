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

    from .scenarios import export

    data = export()
    if a.cmd == "export":
        with open(a.out, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"wrote {a.out}")
        return 0
    print("versions:", data["versions"])
    print("\nAggregation (max |w*N - intended|, min cosine)")
    for pnl in data["aggregation"]["panels"]:
        print(f"  [{pnl['kind']:9s}] {pnl['title']:52s} {pnl['maxDeviation']:.2e}  {pnl['minCos']:.6f}")
    print("\nLeaks (max off-diagonal)")
    for pnl in data["leak"]["panels"]:
        print(f"  [{pnl['kind']:9s}] {pnl['title']:52s} {pnl['maxLeak']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
