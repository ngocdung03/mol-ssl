"""Bemis-Murcko scaffold splitting and split manifests.

Rule 1 of the project: scaffold splits only. Random splits leak scaffolds across train/test and
inflate every downstream number. There is deliberately no random-split function in this module.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def scaffold_smiles(smiles: str, include_chirality: bool = False) -> str | None:
    """Bemis-Murcko scaffold as canonical SMILES; None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_groups(smiles_list: list[str]) -> dict[str, list[int]]:
    """scaffold -> indices. Unparseable SMILES are excluded (caller should count them)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaf = scaffold_smiles(smi)
        if scaf is not None:
            groups[scaf].append(idx)
    return dict(groups)


def scaffold_split(
    smiles_list: list[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
) -> dict[str, list[int]]:
    """Deterministic scaffold split: largest scaffold groups fill train first.

    Deterministic (no seed) by design — the split is a fixed artifact, and seeds vary the model, not
    the data. Matches the DeepChem/Chemprop 'scaffold' (non-balanced) protocol.
    """
    if abs(frac_train + frac_val + frac_test - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")

    groups = scaffold_groups(smiles_list)
    # sort by group size desc, then by scaffold SMILES for a stable tie-break
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    n_total = sum(len(v) for v in groups.values())
    n_train_max = frac_train * n_total
    n_val_max = (frac_train + frac_val) * n_total

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for _scaf, idxs in ordered:
        if len(train) + len(idxs) <= n_train_max:
            train += idxs
        elif len(train) + len(val) + len(idxs) <= n_val_max:
            val += idxs
        else:
            test += idxs
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def save_split(path: str | Path, dataset: str, split: dict[str, list[int]], smiles_list: list[str]) -> None:
    """Write a split manifest: the single source of truth for this dataset's partition.

    Stores the scaffold sets alongside the indices so leakage checks never have to re-derive them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "split_type": "scaffold_bemis_murcko",
        "n_total": len(smiles_list),
        "indices": split,
        "scaffolds": {
            part: sorted({s for s in (scaffold_smiles(smiles_list[i]) for i in idxs) if s})
            for part, idxs in split.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def load_split(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("split_type") != "scaffold_bemis_murcko":
        raise ValueError(f"refusing non-scaffold split manifest: {payload.get('split_type')!r}")
    return payload


def heldout_scaffolds(manifest: dict) -> set[str]:
    """Every scaffold the unlabeled pool must NOT contain: val + test."""
    scafs = manifest["scaffolds"]
    return set(scafs.get("val", [])) | set(scafs.get("test", []))
