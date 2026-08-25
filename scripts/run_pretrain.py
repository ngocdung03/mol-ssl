#!/usr/bin/env python
"""Phase 3 step 1: pretrain a GINE encoder by attribute masking on the unlabeled pool.

Writes an encoder checkpoint that `run_baseline.py --pretrained` can load. No labels are used here.

LEAKAGE: the pool must already have been scaffold-filtered against every downstream val/test split
(hard rule 2). This script re-asserts that invariant before training rather than trusting the
upstream step, because a pool built once and reused silently is exactly how leakage survives.

Usage:
  python scripts/run_pretrain.py --pool data/pool/zinc250k_filtered.txt --kw pretrain_zinc
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.datamodule import filter_pool_by_scaffold
from src.featurize import mol_to_data
from src.models.gnn import GINEEncoder
from src.runlog import append_ledger, git_rev, make_run_tag, resolve_keyword, write_metrics
from src.splits import load_split

DOWNSTREAM_MANIFESTS = [
    "data/splits/tox21_scaffold.json",
    "data/splits/bbbp_scaffold.json",
    "data/splits/bace_scaffold.json",
    "data/splits/lipophilicity_scaffold.json",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="one SMILES per line, already scaffold-filtered")
    ap.add_argument("--kw", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/encoder_pretrained.pt")
    args = ap.parse_args()

    kw = resolve_keyword(args.kw)
    run_tag = make_run_tag(kw)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pool = [s.strip() for s in Path(args.pool).read_text().splitlines() if s.strip()]
    print(f"[{run_tag}] pool={len(pool)} molecules from {args.pool}")

    # Re-assert rule 2 rather than trusting that the pool file is clean.
    manifests = [load_split(p) for p in DOWNSTREAM_MANIFESTS if Path(p).exists()]
    kept, dropped = filter_pool_by_scaffold(pool, manifests)
    if dropped:
        print(f"REFUSING: pool contained {dropped} molecules with held-out scaffolds. "
              f"Rebuild the pool with the filter before pretraining.")
        return 1
    print(f"leakage re-check passed: 0 held-out scaffolds in pool (checked vs {len(manifests)} manifests)")

    graphs = [g for g in (mol_to_data(s) for s in kept) if g is not None]
    print(f"featurized {len(graphs)}/{len(kept)} molecules")

    from torch_geometric.loader import DataLoader

    from src.ssl.pretrain import build

    encoder = GINEEncoder(hidden=args.hidden, layers=args.layers).to(device)
    masker = build({"encoder": encoder, "mask_rate": args.mask_rate}).to(device)
    opt = torch.optim.Adam(masker.parameters(), lr=args.lr)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)

    history = []
    for epoch in range(args.epochs):
        masker.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = masker.loss(batch)
            loss.backward()
            opt.step()
            total += float(loss) * batch.num_graphs
            n += batch.num_graphs
        mean = total / max(1, n)
        history.append(mean)
        print(f"  epoch {epoch:3d} mask_loss {mean:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "hidden": args.hidden, "layers": args.layers,
        "run_tag": run_tag, "mask_rate": args.mask_rate,
        "n_pool": len(graphs), "git": git_rev(),
    }, args.out)

    metrics = {
        "run_tag": run_tag, "script": "run_pretrain", "pool": args.pool,
        "n_pool_input": len(pool), "n_pool_featurized": len(graphs),
        "epochs": args.epochs, "mask_rate": args.mask_rate, "seed": args.seed,
        "final_mask_loss": history[-1] if history else None,
        "mask_loss_history": history, "checkpoint": args.out, "git": git_rev(),
    }
    write_metrics(run_tag, metrics)
    append_ledger({
        "run_tag": run_tag, "script": "run_pretrain", "dataset": "pool",
        "ssl_method": "pretrain_attrmask", "n_seeds": 1,
        "n_train": len(graphs), "final_mask_loss": metrics["final_mask_loss"],
        "git": git_rev(),
    })
    print(f"\nsaved encoder -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
