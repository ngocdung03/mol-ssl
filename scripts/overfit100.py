#!/usr/bin/env python
"""Phase-0 deliverable: a GINE must overfit 100 molecules to near-zero loss.

Catches featurization and batching bugs that no downstream metric will reveal.
Usage: python scripts/overfit100.py --kw overfit_check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.datamodule import load_moleculenet
from src.featurize import mol_to_data
from src.models.gnn import PropertyPredictor
from src.runlog import append_ledger, git_rev, make_run_tag, resolve_keyword, write_metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kw", default=None, help="run keyword (required)")
    ap.add_argument("--dataset", default="BBBP")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kw = resolve_keyword(args.kw)
    run_tag = make_run_tag(kw)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_moleculenet(args.dataset)
    graphs = []
    for d in ds:
        g = mol_to_data(d.smiles, y=[float(d.y.view(-1)[0])])
        if g is not None and not torch.isnan(g.y).any():
            graphs.append(g)
        if len(graphs) >= args.n:
            break

    model = PropertyPredictor(n_tasks=1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(graphs, batch_size=32, shuffle=True)

    loss = float("nan")
    for step in range(args.steps):
        model.train()
        for batch in loader:
            batch = batch.to(dev)
            opt.zero_grad()
            loss_t = F.binary_cross_entropy_with_logits(model(batch), batch.y)
            loss_t.backward()
            opt.step()
        loss = float(loss_t)
        if step % 50 == 0:
            print(f"step {step:4d} loss {loss:.5f}")

    print(f"final loss {loss:.6f}")
    metrics = {
        "run_tag": run_tag, "script": "overfit100", "dataset": args.dataset,
        "n_molecules": len(graphs), "steps": args.steps, "seed": args.seed,
        "final_train_loss": loss, "passed": loss < 0.05, "git": git_rev(),
    }
    write_metrics(run_tag, metrics)
    append_ledger(metrics)
    print("OVERFIT_OK" if metrics["passed"] else "OVERFIT_FAIL")
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
