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
    """Rule 1 of the methodological rules in README.md, enforced in code."""
    split = cfg.get("split")
    if split is None:
        raise ValueError("config has no 'split': refusing to guess (see README.md, rule 1)")
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


def filter_pool_indices(pool_smiles: list[str], manifests: list[dict]) -> tuple[list[int], int]:
    """Indices of pool molecules whose scaffold appears in NO val/test split.

    Index-returning sibling of `filter_pool_by_scaffold`, so a *labeled* corpus (e.g. the PCBA
    graph-level pretraining panel) can be filtered without its labels having to round-trip through
    a SMILES list. Same rule, same banned set: unparseable SMILES are dropped too, because an
    unknown scaffold cannot be proven clean.

    Returns (kept_indices, n_dropped).
    """
    banned: set[str] = set()
    for m in manifests:
        banned |= heldout_scaffolds(m)

    kept: list[int] = []
    dropped = 0
    for i, smi in enumerate(pool_smiles):
        scaf = scaffold_smiles(smi)
        if scaf is None or scaf in banned:
            dropped += 1
            continue
        kept.append(i)
    return kept, dropped


def filter_pool_by_scaffold(pool_smiles: list[str], manifests: list[dict]) -> tuple[list[str], int]:
    """Drop any pool molecule whose Bemis-Murcko scaffold appears in a val/test split.

    Returns (kept, n_dropped). Thin wrapper over `filter_pool_indices` so the SMILES path and the
    labeled path cannot drift apart on what counts as clean.
    """
    kept_idx, dropped = filter_pool_indices(pool_smiles, manifests)
    return [pool_smiles[i] for i in kept_idx], dropped


def load_moleculenet(name: str, root: str | Path = "data/raw"):
    """MoleculeNet via PyG (avoids a deepchem dependency). Import is local so tests stay light."""
    from torch_geometric.datasets import MoleculeNet

    return MoleculeNet(root=str(root), name=name)


def dataset_graphs(name: str, root: str | Path = "data/raw") -> tuple[list, list[str], int]:
    """MoleculeNet -> (graphs, smiles, n_dropped), index-aligned with the split manifest.

    Index alignment is the whole contract here: manifests store integer indices into the PyG
    dataset order, so an unparseable molecule must keep its slot rather than shifting every index
    after it. Dropped entries become None and are filtered per-split, and the count is returned so
    the caller reports it instead of losing it.
    """
    from src.featurize import mol_to_data

    ds = load_moleculenet(name, root=root)
    graphs, smiles, dropped = [], [], 0
    for d in ds:
        g = mol_to_data(d.smiles, y=d.y.view(-1).tolist())
        if g is None:
            dropped += 1
        graphs.append(g)
        smiles.append(d.smiles)
    return graphs, smiles, dropped


def split_subset(graphs: list, indices: list[int]) -> list:
    """Graphs for a set of manifest indices, skipping any that failed to featurize."""
    return [graphs[i] for i in indices if graphs[i] is not None]


def build_loaders(cfg: dict, seed: int, root: str | Path = "data/raw"):
    """Config + seed -> (train_loader, val_loader, test_loader, info).

    The label budget is applied to the *training* indices only, so a low-label run never gains
    access to validation or test chemotypes -- the budget shrinks supervision, not the split.
    """
    from torch_geometric.loader import DataLoader

    from src.splits import load_split

    require_scaffold_split(cfg)
    manifest = load_split(cfg["split_manifest"])
    graphs, _smiles, dropped = dataset_graphs(cfg["dataset"], root=root)

    idx = manifest["indices"]
    budget = float(cfg.get("label_budget", 1.0))
    train_idx = label_budget_subsample(idx["train"], budget, seed) if budget < 1.0 else idx["train"]

    train = split_subset(graphs, train_idx)
    val = split_subset(graphs, idx["val"])
    test = split_subset(graphs, idx["test"])

    bs = int(cfg.get("train", {}).get("batch_size", 32))
    info = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_dropped_unparseable": dropped, "label_budget": budget,
        "n_train_pool": len(idx["train"]),
    }
    return (
        DataLoader(train, batch_size=bs, shuffle=True),
        DataLoader(val, batch_size=bs),
        DataLoader(test, batch_size=bs),
        info,
    )
