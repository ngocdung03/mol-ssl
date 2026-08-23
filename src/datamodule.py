"""Dataset loading, split enforcement, label-budget subsampling, unlabeled pool.

Two hard gates live here:
  * `require_scaffold_split` — a config that omits or randomizes the split cannot run.
  * `filter_pool_by_scaffold` — the unlabeled pool is filtered against held-out scaffolds before
    it is written to disk. `tests/test_leakage.py` proves this has teeth.
"""
from __future__ import annotations

import random
from pathlib import Path

from src.splits import heldout_scaffolds, scaffold_smiles

ALLOWED_SPLIT = "scaffold"


def require_scaffold_split(cfg: dict) -> str:
    """Rule 1, enforced in code and not only in CLAUDE.md."""
    split = cfg.get("split")
    if split is None:
        raise ValueError("config has no 'split': refusing to guess (see CLAUDE.md rule 1)")
    if split != ALLOWED_SPLIT:
        raise ValueError(f"split={split!r} forbidden; only {ALLOWED_SPLIT!r} is permitted")
    return split


def label_budget_subsample(train_idx: list[int], budget: float, seed: int) -> list[int]:
    """Take `budget` fraction of the labeled training indices, reproducibly.

    Subsamples *within* the scaffold train split, so the low-label regime never gains access to
    val/test chemotypes. Seed varies the labeled subset; the split itself stays fixed.
    """
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    n = max(1, round(len(train_idx) * budget))
    rng = random.Random(seed)
    return sorted(rng.sample(list(train_idx), n))


def filter_pool_by_scaffold(pool_smiles: list[str], manifests: list[dict]) -> tuple[list[str], int]:
    """Drop any pool molecule whose Bemis-Murcko scaffold appears in a val/test split.

    Returns (kept, n_dropped). Unparseable pool SMILES are dropped too — an unknown scaffold cannot
    be proven clean.
    """
    banned: set[str] = set()
    for m in manifests:
        banned |= heldout_scaffolds(m)

    kept: list[str] = []
    dropped = 0
    for smi in pool_smiles:
        scaf = scaffold_smiles(smi)
        if scaf is None or scaf in banned:
            dropped += 1
            continue
        kept.append(smi)
    return kept, dropped


def load_moleculenet(name: str, root: str | Path = "data/raw"):
    """MoleculeNet via PyG (avoids a deepchem dependency). Import is local so tests stay light."""
    from torch_geometric.datasets import MoleculeNet

    return MoleculeNet(root=str(root), name=name)
