"""QUARANTINED: random splitting, for measuring split inflation ONLY.

This module exists for exactly one purpose: to quantify how much a random split inflates a
molecular benchmark score relative to a Bemis-Murcko scaffold split. That inflation is the finding.

It is deliberately kept out of `src/splits.py` so the production path has no random-split function
to reach for, and it writes manifests tagged `random_for_comparison_only` -- which
`src.splits.load_split` REFUSES to load. A random split can therefore never reach the training
pipeline, the unlabeled pool, or an SSL run by accident.

Rules for anything in this file:
  * Never feed its output to an unlabeled pool or any pretraining step.
  * Never report a number from it except side by side with the scaffold number.
  * Never relax `require_scaffold_split` to accommodate it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

MANIFEST_TYPE = "random_for_comparison_only"


def random_split(
    n_total: int,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 0,
) -> dict[str, list[int]]:
    """A seeded random partition of indices. Seeded so the inflation measurement is reproducible."""
    if abs(frac_train + frac_val + frac_test - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")

    idx = list(range(n_total))
    random.Random(seed).shuffle(idx)
    n_train = int(frac_train * n_total)
    n_val = int((frac_train + frac_val) * n_total)
    return {
        "train": sorted(idx[:n_train]),
        "val": sorted(idx[n_train:n_val]),
        "test": sorted(idx[n_val:]),
    }


def save_random_split(path: str | Path, dataset: str, split: dict[str, list[int]], seed: int) -> None:
    """Write a manifest that the production loader will refuse.

    The `split_type` is not `scaffold_bemis_murcko`, so `src.splits.load_split` raises on it. That
    refusal is the safety mechanism: this artifact is readable only by the comparison script.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dataset": dataset,
        "split_type": MANIFEST_TYPE,
        "seed": seed,
        "warning": "RANDOM SPLIT. Comparison only. Never use for SSL, pools, or reported results.",
        "indices": split,
    }, indent=2))


def load_random_split(path: str | Path) -> dict:
    """Load a comparison manifest, refusing anything that is not explicitly a random one."""
    payload = json.loads(Path(path).read_text())
    if payload.get("split_type") != MANIFEST_TYPE:
        raise ValueError(f"not a comparison manifest: {payload.get('split_type')!r}")
    return payload
