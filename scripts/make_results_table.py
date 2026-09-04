#!/usr/bin/env python
"""Render the results table from metrics JSONs -- the only permitted source of a reported number.

Hard rule 3 says every number in the README traces to a `results/metrics_{run_tag}.json`. This
script is the mechanism: it reads that directory and nothing else, so a number can only appear in
the table if a run actually produced it. Published literature values are kept in a separate,
explicitly labeled column and are never mixed with measured ones.

Usage: python scripts/make_results_table.py > results/RESULTS.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Published scaffold-split numbers, cited not reproduced (see README.md, Method sources).
# Source: Yang et al. 2019 (Chemprop, D-MPNN), MoleculeNet scaffold-split protocol.
PUBLISHED = {
    ("Tox21", "auroc"): ("0.759", "Chemprop D-MPNN, Yang et al. 2019 (published, not reproduced)"),
    ("BBBP", "auroc"): ("0.913", "Chemprop D-MPNN, Yang et al. 2019 (published, not reproduced)"),
    ("BACE", "auroc"): ("0.852", "Chemprop D-MPNN, Yang et al. 2019 (published, not reproduced)"),
    ("Lipo", "rmse"): ("0.555", "Chemprop D-MPNN, Yang et al. 2019 (published, not reproduced)"),
}


def load_runs() -> list[dict]:
    runs = []
    for p in sorted(RESULTS.glob("metrics_*.json")):
        m = json.loads(p.read_text())
        if m.get("script") in {"run_baseline", "run_ecfp_baseline"}:
            m["_path"] = p.name
            runs.append(m)
    return runs


def fmt(agg: dict | None, digits: int = 4) -> str:
    if not agg or agg.get("n", 0) == 0:
        return "—"
    return f"{agg['mean']:.{digits}f} ± {agg['std']:.{digits}f}"


def main() -> int:
    runs = load_runs()
    if not runs:
        print("No completed baseline runs yet. No numbers to report.")
        return 0

    print("# Results\n")
    print("Every number below is read from a `results/metrics_{run_tag}.json` produced by an actual")
    print("run. Mean ± standard deviation over the seed list; never a best seed, never a best epoch.\n")
    print("## Supervised baselines, Bemis–Murcko scaffold splits\n")
    print("| Dataset | Model | Metric | Result (mean ± std) | Seeds | ECE | Run tag |")
    print("|---|---|---|---|---|---|---|")

    for m in sorted(runs, key=lambda r: (r["dataset"], r.get("model", "gine"))):
        key = "auroc" if m["task"] == "classification" else "rmse"
        model = m.get("model", "GINE (this work)")
        ece = fmt(m.get("test_ece")) if m.get("test_ece") else "—"
        print(f"| {m['dataset']} | {model} | {key.upper()} | {fmt(m.get(f'test_{key}'))} "
              f"| {len(m['seeds'])} | {ece} | `{m['run_tag']}` |")

    print("\n## Published reference values\n")
    print("Cited from the literature, **not reproduced here**. Listed for orientation only; a gap")
    print("against these is expected and is explained in the README rather than tuned away.\n")
    print("| Dataset | Metric | Published | Source |")
    print("|---|---|---|---|")
    for (ds, key), (val, src) in PUBLISHED.items():
        print(f"| {ds} | {key.upper()} | {val} | {src} |")

    print("\n## Split-inflation study\n")
    infl = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("metrics_*.json"))]
    infl = [m for m in infl if m.get("script") == "run_split_comparison"]
    if not infl:
        print("Not run yet.")
    else:
        print("| Dataset | Metric | Scaffold | Random | Inflation | Seeds | Run tag |")
        print("|---|---|---|---|---|---|---|")
        for m in infl:
            key = "auroc" if m["task"] == "classification" else "rmse"
            print(f"| {m['dataset']} | {key.upper()} | {fmt(m.get(f'scaffold_{key}'))} "
                  f"| {fmt(m.get(f'random_{key}'))} | {fmt(m.get('inflation'))} "
                  f"| {len(m['seeds'])} | `{m['run_tag']}` |")

    print("\n## Label-budget sweeps (pretraining arms)\n")
    sweeps = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("metrics_*.json"))]
    sweeps = [m for m in sweeps if m.get("script") == "run_label_budget_sweep"]
    if not sweeps:
        print("Not run yet.")
    else:
        print("Arms are encoder-initialization conditions; split, seeds and labeled subsample are")
        print("held identical within a run. See `results/PRECLAIM_node_vs_graph.md` for the")
        print("pre-registered thresholds.\n")
        print("| Dataset | Metric | Arm | Budget | Value | Seeds | Run tag |")
        print("|---|---|---|---|---|---|---|")
        for m in sweeps:
            key = m.get("metric", "auroc")
            for arm, series in m["curve"].items():
                for b in sorted(series, key=float):
                    agg = series[b]
                    pct = f"{float(b) * 100:g}%"
                    print(f"| {m['dataset']} | {key.upper()} | {arm} | {pct} "
                          f"| {fmt(agg)} | {len(m['seeds'])} | `{m['run_tag']}` |")

        for m in sweeps:
            if not m.get("interaction"):
                continue
            print(f"\n### 2x2 interaction, `{m['run_tag']}`\n")
            print("`(node_graph - graph) - (node - none)`, equivalently")
            print("`node_graph - (graph + node) + none`: how far the combined arm exceeds the")
            print("additive prediction. Near zero means the two objectives contribute additively.")
            print("This contrast is this study's construction -- the 2x2 layout is Hu et al.'s but")
            print("they never compute an interaction. Their own Table 1 averages give +0.2, so a")
            print("near-zero value replicates their result rather than contradicting it.\n")
            print("| Budget | Interaction | Pooled std | Inside seed noise | Clears 2x pooled |")
            print("|---|---|---|---|---|")
            for b in sorted(m["interaction"], key=float):
                d = m["interaction"][b]
                print(f"| {float(b) * 100:g}% | {d['interaction']:+.4f} | {d['pooled_std']:.4f} "
                      f"| {'yes' if d['inside_seed_noise'] else 'no'} "
                      f"| {'yes' if d['exceeds_widened_threshold'] else 'no'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
