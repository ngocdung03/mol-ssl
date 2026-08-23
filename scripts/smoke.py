#!/usr/bin/env python
"""End-to-end smoke test: SMILES -> graph -> GINE forward pass on GPU.

The PLAN.md §2 anchor: run this before trusting anything else in the repo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.loader import DataLoader

from src.featurize import mol_to_data
from src.models.gnn import PropertyPredictor

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",           # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",    # caffeine
    "C",                               # methane, no bonds
]


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device {dev} | gpus {torch.cuda.device_count()}")
    if dev == "cuda":
        print("gpu0:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))

    graphs = [mol_to_data(s, y=[0.0]) for s in SMILES]
    assert all(g is not None for g in graphs), "featurization returned None"
    for s, g in zip(SMILES, graphs):
        print(f"  {s:34s} atoms={g.x.shape[0]:3d} edges={g.edge_index.shape[1]:3d}")

    model = PropertyPredictor(n_tasks=1).to(dev).eval()
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs))))
    with torch.no_grad():
        out = model(batch.to(dev))
    print("forward ok, output shape", tuple(out.shape))
    assert out.shape == (len(SMILES), 1)
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
