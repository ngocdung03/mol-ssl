#!/usr/bin/env python
"""The headline experiment: metric vs label budget, supervised against pretrained.

For each budget, train both arms on the SAME labeled subset (same seed -> same subsample), so the
only difference between them is whether the encoder was pretrained. Anything else would confound
the comparison with a data difference.

The claim this is designed to support is "here is what pretraining buys you per label, measured
properly" -- not "pretraining wins". If the gap sits inside the seed noise, that is the result.

Usage:
  python scripts/run_label_budget_sweep.py --config configs/labelbudget_tox21.yaml \
      --checkpoint artifacts/encoder_pretrained.pt --kw tox21_sweep
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import budget_list, load_config
from src.eval import aggregate_over_seeds
from src.runlog import append_ledger, git_rev, make_run_tag, resolve_keyword, write_metrics
from src.train import train_one_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="pretrained encoder from run_pretrain.py")
    ap.add_argument("--kw", default=None)
    ap.add_argument("--arms", default="none,pretrain", help="comma-separated: none,pretrain")
    args = ap.parse_args()

    kw = resolve_keyword(args.kw)
    run_tag = make_run_tag(kw)
    cfg = load_config(args.config)
    task = cfg["task"]
    key = "auroc" if task == "classification" else "rmse"
    budgets = budget_list(cfg)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"[{run_tag}] {cfg['dataset']} sweep: budgets={budgets} arms={arms} seeds={cfg['seeds']}")

    curve = {}
    for arm in arms:
        curve[arm] = {}
        for budget in budgets:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["label_budget"] = budget
            run_cfg.setdefault("ssl", {})
            # Same seed -> same labeled subsample in both arms; only the encoder init differs.
            run_cfg["ssl"]["checkpoint"] = args.checkpoint if arm == "pretrain" else None

            scores, n_train = [], None
            for seed in cfg["seeds"]:
                r = train_one_seed(run_cfg, seed=seed, verbose=False)
                scores.append(r["test"].get(key))
                n_train = r["data"]["n_train"]
            agg = aggregate_over_seeds(scores)
            curve[arm][str(budget)] = {**agg, "n_train": n_train}
            print(f"  {arm:9s} budget {budget:>5} n_train={n_train:5d} "
                  f"{key} {agg['mean']:.4f} +/- {agg['std']:.4f}", flush=True)

    # The comparison is only meaningful with the seed spread alongside it, so carry both.
    deltas = {}
    if "none" in curve and "pretrain" in curve:
        for b in curve["none"]:
            base, pre = curve["none"][b], curve["pretrain"][b]
            gap = pre["mean"] - base["mean"]
            pooled = (base["std"] ** 2 + pre["std"] ** 2) ** 0.5
            deltas[b] = {
                "delta": gap,
                "pooled_std": pooled,
                "inside_seed_noise": abs(gap) < pooled,
            }
            verdict = "inside seed noise" if deltas[b]["inside_seed_noise"] else "exceeds seed noise"
            print(f"  delta @ {b}: {gap:+.4f} (pooled std {pooled:.4f}) -- {verdict}")

    metrics = {
        "run_tag": run_tag, "script": "run_label_budget_sweep",
        "dataset": cfg["dataset"], "task": task, "metric": key,
        "config": cfg["config_path"], "checkpoint": args.checkpoint,
        "budgets": budgets, "arms": arms, "seeds": cfg["seeds"],
        "curve": curve, "deltas": deltas, "git": git_rev(),
    }
    write_metrics(run_tag, metrics)
    for arm in arms:
        for b, agg in curve[arm].items():
            append_ledger({
                "run_tag": f"{run_tag}_{arm}_{b}", "script": "run_label_budget_sweep",
                "dataset": cfg["dataset"], "task": task, "split": "scaffold",
                "label_budget": b, "ssl_method": arm, "n_seeds": len(cfg["seeds"]),
                f"test_{key}_mean": agg["mean"], f"test_{key}_std": agg["std"],
                "n_train": agg["n_train"], "git": git_rev(),
            })
    print(f"\nwrote results/metrics_{run_tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
