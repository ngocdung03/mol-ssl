#!/usr/bin/env python
"""Generate the committed scaffold split manifests -- the single source of truth for partitions.

Splits are deterministic and seedless by design: the partition is a fixed artifact of the dataset,
and seeds vary the model and the labeled subset, never the data. Run once; commit the output.

Usage: python scripts/make_splits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datamodule import load_moleculenet
from src.splits import save_split, scaffold_split

DATASETS = {
    "Tox21": "data/splits/tox21_scaffold.json",
    "BBBP": "data/splits/bbbp_scaffold.json",
    "BACE": "data/splits/bace_scaffold.json",
    "Lipo": "data/splits/lipophilicity_scaffold.json",
}


def main() -> int:
    for name, out in DATASETS.items():
        ds = load_moleculenet(name)
        smiles = [d.smiles for d in ds]
        split = scaffold_split(smiles, 0.8, 0.1, 0.1)
        save_split(out, name, split, smiles)
        n = {k: len(v) for k, v in split.items()}
        total = sum(n.values())
        print(f"{name:12s} n={len(smiles):6d} kept={total:6d} "
              f"train={n['train']:6d} val={n['val']:5d} test={n['test']:5d} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
