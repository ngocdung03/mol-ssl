#!/usr/bin/env python
"""ECFP4 + RandomForest / XGBoost on the same committed scaffold splits as the GNN.

Usage:
  python scripts/run_ecfp_baseline.py --config configs/baseline_tox21_gine.yaml --model rf --kw tox21_ecfp_rf
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import featurize_many, fit_predict_multitask
from src.config import load_config
from src.datamodule import label_budget_subsample
from src.eval import aggregate_over_seeds, expected_calibration_error, masked_auroc, masked_rmse, selective_accuracy
from src.runlog import append_ledger, git_rev, make_run_tag, resolve_keyword, write_metrics
from src.splits import load_split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=["rf", "xgb"], required=True)
    ap.add_argument("--kw", default=None)
    ap.add_argument("--label-budget", type=float, default=None)
    args = ap.parse_args()

    kw = resolve_keyword(args.kw)
    run_tag = make_run_tag(kw)
    cfg = load_config(args.config)
    if args.label_budget is not None:
        cfg["label_budget"] = args.label_budget

    task = cfg["task"]
    key = "auroc" if task == "classification" else "rmse"
    manifest = load_split(cfg["split_manifest"])
    idx = manifest["indices"]

    from src.datamodule import load_moleculenet
    ds = load_moleculenet(cfg["dataset"])
    smiles = [d.smiles for d in ds]
    Y = np.stack([d.y.view(-1).numpy() for d in ds]).astype(float)

    budget = float(cfg.get("label_budget", 1.0))
    print(f"[{run_tag}] {cfg['dataset']} {args.model} budget={budget} seeds={cfg['seeds']}")

    scores, eces, per_seed = [], [], []
    for seed in cfg["seeds"]:
        tr_idx = label_budget_subsample(idx["train"], budget, seed) if budget < 1.0 else idx["train"]
        X_tr, keep_tr = featurize_many([smiles[i] for i in tr_idx])
        X_te, keep_te = featurize_many([smiles[i] for i in idx["test"]])
        Y_tr = Y[[tr_idx[i] for i in keep_tr]]
        Y_te = Y[[idx["test"][i] for i in keep_te]]

        pred = fit_predict_multitask(args.model, task, seed, X_tr, Y_tr, X_te)

        if task == "classification":
            mean, _per, skipped = masked_auroc(Y_te, np.nan_to_num(pred, nan=0.5))
            ece = expected_calibration_error(Y_te, np.nan_to_num(pred, nan=0.5))
            sel = selective_accuracy(Y_te, np.nan_to_num(pred, nan=0.5))
            per_seed.append({"seed": seed, "test": {"auroc": mean, "ece": ece,
                                                    "n_tasks_skipped": skipped,
                                                    "selective_accuracy": sel},
                             "n_train": int(X_tr.shape[0])})
            scores.append(mean); eces.append(ece)
            print(f"  seed {seed}: auroc {mean:.4f} ece {ece:.4f}")
        else:
            rmse = masked_rmse(Y_te, pred)
            per_seed.append({"seed": seed, "test": {"rmse": rmse}, "n_train": int(X_tr.shape[0])})
            scores.append(rmse)
            print(f"  seed {seed}: rmse {rmse:.4f}")

    agg = aggregate_over_seeds(scores)
    metrics = {
        "run_tag": run_tag, "script": "run_ecfp_baseline", "model": f"ecfp4_{args.model}",
        "config": cfg["config_path"], "dataset": cfg["dataset"], "task": task,
        "split": cfg["split"], "split_manifest": cfg["split_manifest"],
        "label_budget": budget, "ssl_method": "none", "seeds": cfg["seeds"],
        f"test_{key}": agg, "per_seed": per_seed, "git": git_rev(),
    }
    if eces:
        metrics["test_ece"] = aggregate_over_seeds(eces)
    write_metrics(run_tag, metrics)

    append_ledger({
        "run_tag": run_tag, "script": "run_ecfp_baseline", "dataset": cfg["dataset"],
        "task": task, "split": cfg["split"], "label_budget": budget,
        "ssl_method": f"none_ecfp4_{args.model}", "n_seeds": len(cfg["seeds"]),
        f"test_{key}_mean": agg["mean"], f"test_{key}_std": agg["std"],
        "test_ece_mean": metrics.get("test_ece", {}).get("mean", ""),
        "n_train": per_seed[0]["n_train"], "git": git_rev(),
    })
    print(f"\ntest_{key}: {agg['mean']:.4f} +/- {agg['std']:.4f} over {agg['n']} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
