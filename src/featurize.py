"""RDKit SMILES -> PyG graph tensors.

Atom/bond feature sets follow the standard MoleculeNet/Chemprop-style scheme. Kept explicit and
boring on purpose: a featurization bug is invisible downstream and silently ruins every SSL claim.
"""
from __future__ import annotations

import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

ATOM_ELEMENTS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B",
    "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu",
    "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

ATOM_FEAT_DIM = (
    len(ATOM_ELEMENTS) + 1 + len(DEGREES) + 1 + len(FORMAL_CHARGES) + 1
    + len(NUM_HS) + 1 + len(HYBRIDIZATIONS) + 1 + 2
)
BOND_FEAT_DIM = len(BOND_TYPES) + 1 + 2


def _one_hot(value, choices: list, allow_other: bool = True) -> list[float]:
    """One-hot with an explicit trailing 'other' slot, so unseen values never silently map to 0."""
    vec = [0.0] * (len(choices) + (1 if allow_other else 0))
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        if allow_other:
            vec[-1] = 1.0
        else:
            raise
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_ELEMENTS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )


def mol_to_data(smiles: str, y=None) -> Data | None:
    """SMILES -> PyG Data, or None if RDKit cannot parse it.

    Returns None rather than raising: MoleculeNet contains unparseable SMILES, and dropping them
    must be counted and logged by the caller, not swallowed here.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, edge_attr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_attr += [feats, feats]

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    edge_attr = (
        torch.tensor(edge_attr, dtype=torch.float) if edge_attr
        else torch.empty((0, BOND_FEAT_DIM), dtype=torch.float)
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if y is not None:
        data.y = torch.as_tensor(y, dtype=torch.float).view(1, -1)
    return data
