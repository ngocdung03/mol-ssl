#!/usr/bin/env python
"""Phase 2: how much does a random split inflate the score, versus a scaffold split?

Identical model, identical data, identical seeds -- only the split changes. The gap is the
inflation, and it is the finding that does not depend on semi-supervised learning working.

Usage:
  python scripts/run_split_comparison.py --config configs/baseline_tox21_gine.yaml --kw tox21_splitgap

The random split is produced by the QUARANTINED `src/splits_random_for_comparison` module and is
never written into `data/splits/`, never reaches an unlabeled pool, and is refused by the
production loader. See that module's docstring.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import load_config
from src.datamodule import dataset_graphs, split_subset
from src.eval import aggregate_over_seeds
from src.runlog import append_ledger, git_rev, make_run_tag, resolve_keyword, write_metrics
from src.splits import load_split
from src.splits_random_for_comparison import random_split, save_random_split
from src.train import evaluate
from src.models.gnn import PropertyPredictor


def train_on_indices(cfg, graphs, idx, seed, device):
    """Same training recipe as src.train, driven by an explicit index dict rather than a manifest."""
    from torch_geometric.loader import DataLoader

    from src.train import masked_bce, masked_mse

    torch.manual_seed(seed)
    np.random.seed(seed)
    task = cfg["task"]
    tcfg, mcfg = cfg.get("train", {}), cfg.get("model", {})
    bs = int(tcfg.get("batch_size", 32))

    tr = DataLoader(split_subset(graphs, idx["train"]), batch_size=bs, shuffle=True)
    va = DataLoader(split_subset(graphs, idx["val"]), batch_size=bs)
    te = DataLoader(split_subset(graphs, idx["test"]), batch_size=bs)

    model = PropertyPredictor(
        n_tasks=int(cfg.get("n_tasks", 1)), hidden=int(mcfg.get("hidden", 300)),
        layers=int(mcfg.get("layers", 5)), dropout=float(mcfg.get("dropout", 0.1)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 1e-3)))
    use_amp = str(tcfg.get("amp", "fp16")) == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = masked_bce if task == "classification" else masked_mse
    better = (lambda a, b: a > b) if task == "classification" else (lambda a, b: a < b)
    key = "auroc" if task == "classification" else "rmse"

    best, best_state, since = (-float("inf") if task == "classification" else float("inf")), None, 0
    patience = int(tcfg.get("patience", 20))
    for _epoch in range(int(tcfg.get("epochs", 100))):
        model.train()
        for batch in tr:
            batch = batch.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(batch), batch.y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        score = evaluate(model, va, task, device).get(key, float("nan"))
        if not np.isnan(score) and better(score, best):
            best, since = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, te, task, device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kw", default=None)
    args = ap.parse_args()

    kw = resolve_keyword(args.kw)
    run_tag = make_run_tag(kw)
    cfg = load_config(args.config)
    task = cfg["task"]
    key = "auroc" if task == "classification" else "rmse"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    graphs, smiles, dropped = dataset_graphs(cfg["dataset"])
    scaffold_idx = load_split(cfg["split_manifest"])["indices"]
    print(f"[{run_tag}] {cfg['dataset']} split comparison, seeds={cfg['seeds']}, dropped={dropped}")

    scaf_scores, rand_scores, per_seed = [], [], []
    for seed in cfg["seeds"]:
        # Quarantined random split: seeded, kept in a temp file, never in data/splits/.
        rnd = random_split(len(smiles), 0.8, 0.1, 0.1, seed=seed)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            save_random_split(fh.name, cfg["dataset"], rnd, seed=seed)

        s = train_on_indices(cfg, graphs, scaffold_idx, seed, device).get(key)
        r = train_on_indices(cfg, graphs, rnd, seed, device).get(key)
        scaf_scores.append(s)
        rand_scores.append(r)
        per_seed.append({"seed": seed, f"scaffold_{key}": s, f"random_{key}": r})
        print(f"  seed {seed}: scaffold {s:.4f} | random {r:.4f} | gap {r - s:+.4f}")

    scaf, rand = aggregate_over_seeds(scaf_scores), aggregate_over_seeds(rand_scores)
    gap = aggregate_over_seeds([r - s for r, s in zip(rand_scores, scaf_scores)])

    metrics = {
        "run_tag": run_tag, "script": "run_split_comparison", "dataset": cfg["dataset"],
        "task": task, "seeds": cfg["seeds"], "config": cfg["config_path"],
        f"scaffold_{key}": scaf, f"random_{key}": rand, "inflation": gap,
        "per_seed": per_seed, "n_dropped_unparseable": dropped, "git": git_rev(),
    }
    write_metrics(run_tag, metrics)
    append_ledger({
        "run_tag": run_tag, "script": "run_split_comparison", "dataset": cfg["dataset"],
        "task": task, "split": "scaffold_vs_random", "label_budget": 1.0,
        "ssl_method": "none", "n_seeds": len(cfg["seeds"]),
        f"test_{key}_mean": scaf["mean"], f"test_{key}_std": scaf["std"],
        f"random_{key}_mean": rand["mean"], f"inflation_{key}_mean": gap["mean"],
        "git": git_rev(),
    })

    print(f"\nscaffold {key}: {scaf['mean']:.4f} +/- {scaf['std']:.4f}")
    print(f"random   {key}: {rand['mean']:.4f} +/- {rand['std']:.4f}")
    print(f"inflation:      {gap['mean']:+.4f} +/- {gap['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
