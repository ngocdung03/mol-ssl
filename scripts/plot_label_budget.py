#!/usr/bin/env python
"""Plot the headline figure: metric vs label budget, one line per arm, shaded +/-1 std.

Reads only a sweep metrics JSON, so the figure cannot disagree with the recorded numbers.
The shaded band is the point of the plot -- a gap smaller than the bands is not a result, and the
figure should make that visible rather than hide it behind two confident-looking lines.

Usage: python scripts/plot_label_budget.py --metrics results/metrics_<run_tag>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_LABEL = {
    "none": "supervised (random init)",
    "pretrain": "attribute-mask pretrained",      # legacy alias for 'node'
    "node": "node-level (attribute mask)",
    "graph": "graph-level (PCBA supervised)",
    "node_graph": "node + graph (Hu sequential)",
}
ARM_COLOR = {
    "none": "#B0413E", "pretrain": "#2B6C8F", "node": "#2B6C8F",
    "graph": "#E8A33D", "node_graph": "#4C8C4A",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = json.loads(Path(args.metrics).read_text())
    curve, key = m["curve"], m["metric"]
    out = args.out or f"results/label_budget_{m['dataset'].lower()}_{m['run_tag']}.png"

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    for arm, series in curve.items():
        budgets = sorted(float(b) for b in series)
        means = [series[_k(series, b)]["mean"] for b in budgets]
        stds = [series[_k(series, b)]["std"] for b in budgets]
        xs = [b * 100 for b in budgets]
        color = ARM_COLOR.get(arm, None)
        ax.plot(xs, means, marker="o", label=ARM_LABEL.get(arm, arm), color=color, zorder=3)
        ax.fill_between(xs,
                        [mu - s for mu, s in zip(means, stds)],
                        [mu + s for mu, s in zip(means, stds)],
                        alpha=0.18, color=color, zorder=2)

    ax.set_xscale("log")
    ax.set_xticks([b * 100 for b in sorted(float(b) for b in next(iter(curve.values())))])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("labeled fraction of the scaffold-train split (%)")
    ax.set_ylabel(f"test {key.upper()} (scaffold split)")
    ax.set_title(f"{m['dataset']}: {key.upper()} vs label budget\n"
                 f"mean ± 1 std over {len(m['seeds'])} seeds", fontsize=11)
    ax.grid(alpha=0.25, zorder=1)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


def _k(series: dict, b: float) -> str:
    """Budget keys are stringified floats; match on value rather than formatting."""
    for k in series:
        if abs(float(k) - b) < 1e-12:
            return k
    raise KeyError(b)


if __name__ == "__main__":
    raise SystemExit(main())
