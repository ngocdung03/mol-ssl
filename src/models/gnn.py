"""GINE encoder + prediction head.

GINE (GIN with edge features) is the standard backbone for the SSL molecular literature
(Hu et al. 2020, MolCLR), which keeps the pretraining comparisons honest.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

from src.featurize import ATOM_FEAT_DIM, BOND_FEAT_DIM


class GINEEncoder(nn.Module):
    def __init__(self, hidden: int = 300, layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.atom_lin = nn.Linear(ATOM_FEAT_DIM, hidden)
        self.bond_lins = nn.ModuleList(nn.Linear(BOND_FEAT_DIM, hidden) for _ in range(layers))
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden)))
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, data) -> torch.Tensor:
        h = self.atom_lin(data.x)
        for conv, norm, bond_lin in zip(self.convs, self.norms, self.bond_lins):
            e = bond_lin(data.edge_attr)
            h = self.dropout(torch.relu(norm(conv(h, data.edge_index, e))))
        return global_mean_pool(h, data.batch)


class PropertyPredictor(nn.Module):
    """Encoder + linear head. n_tasks>1 for multi-task sets like Tox21 (12 tasks)."""

    def __init__(self, n_tasks: int = 1, hidden: int = 300, layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.encoder = GINEEncoder(hidden=hidden, layers=layers, dropout=dropout)
        self.head = nn.Linear(self.encoder.out_dim, n_tasks)

    def forward(self, data) -> torch.Tensor:
        return self.head(self.encoder(data))
