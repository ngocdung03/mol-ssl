#!/usr/bin/env python
"""Run a supervised baseline over the config's full seed list.

Usage:
  python scripts/run_baseline.py --config configs/baseline_bbbp_gine.yaml --kw bbbp_gine_sup

Reports mean +/- std across every seed in the config. There is no flag to report a single seed or
a best seed: hard rule 4 exists because that flag is how optimistic numbers get published.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.eval import aggregate_over_seeds
from src.runlog import append_ledger, git_rev, make_run_tag, resolve_keyword, write_metrics
from src.train import train_one_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kw", default=None, help="run keyword (required)")
    ap.add_argument("--label-budget", type=float, default=None,
                    help="override the config's label budget (for sweeps)")
    args = ap.parse_args()

    kw = resolve_keyword(args.kw)
    run_tag = make_run_tag(kw)
    cfg = load_config(args.config)
    if args.label_budget is not None:
        cfg["label_budget"] = args.label_budget

    task = cfg["task"]
    key = "auroc" if task == "classification" else "rmse"
    seeds = cfg["seeds"]
    print(f"[{run_tag}] {cfg['dataset']} {task} budget={cfg.get('label_budget', 1.0)} seeds={seeds}")

    per_seed = []
    for seed in seeds:
        print(f"seed {seed} ...")
        per_seed.append(train_one_seed(cfg, seed=seed))

    test_scores = [r["test"].get(key) for r in per_seed]
    agg = aggregate_over_seeds(test_scores)
    extra = {}
    if task == "classification":
        extra["test_ece"] = aggregate_over_seeds([r["test"].get("ece") for r in per_seed])

    metrics = {
        "run_tag": run_tag,
        "script": "run_baseline",
        "config": cfg["config_path"],
        "dataset": cfg["dataset"],
        "task": task,
        "split": cfg["split"],
        "split_manifest": cfg["split_manifest"],
        "label_budget": cfg.get("label_budget", 1.0),
        "ssl_method": cfg.get("ssl", {}).get("method", "none"),
        "seeds": seeds,
        f"test_{key}": agg,
        **extra,
        "per_seed": per_seed,
        "git": git_rev(),
    }
    write_metrics(run_tag, metrics)

    append_ledger({
        "run_tag": run_tag,
        "script": "run_baseline",
        "dataset": cfg["dataset"],
        "task": task,
        "split": cfg["split"],
        "label_budget": cfg.get("label_budget", 1.0),
        "ssl_method": metrics["ssl_method"],
        "n_seeds": len(seeds),
        f"test_{key}_mean": agg["mean"],
        f"test_{key}_std": agg["std"],
        "test_ece_mean": extra.get("test_ece", {}).get("mean", ""),
        "n_train": per_seed[0]["data"]["n_train"],
        "git": git_rev(),
    })

    print(f"\ntest_{key}: {agg['mean']:.4f} +/- {agg['std']:.4f} over {agg['n']} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
