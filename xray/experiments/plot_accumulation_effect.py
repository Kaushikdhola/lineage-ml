"""Chart for accumulation_effect.py: distance from the best achievable loss, per data source and arm.

    .venv/bin/python xray/experiments/plot_accumulation_effect.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "accumulation_effect.json"
OUT = HERE / "results" / "accumulation_effect.png"

# reference data-viz palette, light mode: categorical slots 1-3, text tokens, surface
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARMS = {
    "fixed": dict(color="#2a78d6", ls="-", label="Fixed Trainer (transformers 5.22.2)"),
    "buggy": dict(color="#eb6834", ls="-", label="Buggy Trainer (transformers 4.45.2)"),
    "control": dict(color="#1baf7a", ls=(0, (4, 3)), label="Control: 4.45.2 without accumulation (lies on the fixed line)"),
}


def series(runs, arm, src):
    rs = [r for r in runs if r["arm"] == arm]
    steps = [c["step"] for c in rs[0]["curve"]]
    ex = [[r["curve"][i][src] - r["floor"][src] for r in rs] for i in range(len(steps))]
    return steps, [st.mean(e) for e in ex], [min(e) for e in ex], [max(e) for e in ex]


def main():
    d = json.loads(DATA.read_text())
    runs, share = d["runs"], d["token_share"]["short_intended"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2})
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120, sharey=True, facecolor=SURFACE)
    panels = [("short", f"Short sequences: {100 * share:.0f}% of tokens"),
              ("long", f"Long sequences: {100 * (1 - share):.0f}% of tokens")]
    for ax, (src, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for arm, sty in ARMS.items():
            steps, mean, lo, hi = series(runs, arm, src)
            k = steps.index(25)
            steps, mean, lo, hi = steps[k:], mean[k:], lo[k:], hi[k:]
            if arm != "control":
                ax.fill_between(steps, lo, hi, color=sty["color"], alpha=0.12, linewidth=0)
            ax.plot(steps, mean, color=sty["color"], ls=sty["ls"], lw=2.2 if arm != "control" else 1.6,
                    label=sty["label"], zorder=3 if arm != "control" else 4)
            if arm != "control":  # direct labels at the line ends, pushed apart where the lines meet
                dy = 0 if src == "long" else (9 if arm == "buggy" else -9)
                ax.annotate("buggy" if arm == "buggy" else "fixed", (steps[-1], mean[-1]), xytext=(8, dy),
                            textcoords="offset points", va="center", color=TEXT, fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", color=TEXT, fontsize=16, pad=12)
        ax.set_xlabel("Optimizer step")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.grid(True, which="minor", axis="y", color=GRID, lw=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(25, 330)
    axes[0].set_ylabel("Loss above the best achievable (log scale, lower is better)")

    # the measured gap on the long data at the last step
    _, fm, _, _ = series(runs, "fixed", "long")
    _, bm, _, _ = series(runs, "buggy", "long")
    x = 288  # a bracket spanning the gap between the two lines at the last step
    axes[1].annotate("", xy=(x, fm[-1]), xytext=(x, bm[-1]),
                     arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", color=TEXT_2, lw=1.2))
    axes[1].annotate(f"{bm[-1] / fm[-1]:.1f}× further from the\nbest achievable loss", (x, bm[-1]),
                     xytext=(-6, 16), textcoords="offset points", ha="right", va="bottom", color=TEXT, fontsize=14)
    fig.suptitle("The 2024 gradient-accumulation bug starved the data that made up most of the batch",
                 x=0.06, ha="left", y=0.975, fontsize=21, color=TEXT, fontweight="bold")
    fig.text(0.06, 0.915, f"Tiny LLaMA, two synthetic data sources, micro-batches grouped by length. The bug gave the short data 50% "
             f"of each step's gradient instead of {100 * share:.0f}%.\nReal runs of the transformers Trainer; mean of 3 seeds, "
             "bands show min to max. A toy: the direction is the point, not the sizes.",
             ha="left", va="top", fontsize=13, color=TEXT_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=3, frameon=False, fontsize=13,
               labelcolor=TEXT)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.80, bottom=0.17, wspace=0.08)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
