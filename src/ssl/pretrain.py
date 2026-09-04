"""Self-supervised pretraining by attribute masking (Hu et al. 2020).

The idea: hide a fraction of atoms' identities and make the encoder reconstruct them from graph
context. Chemistry is highly constrained -- the neighbours of an atom narrow down what it can be --
so this forces the encoder to learn local chemical environments without using a single assay label.

Only the encoder is pretrained. The prediction head is discarded and re-initialized for the
downstream task, which is what makes the encoder weights transferable across property datasets.

Scope note: attribute masking was the first self-supervised method implemented here.
Contrastive, consistency, and pseudo-labeling remain stubs behind the same interface;
graph-level supervised pretraining is implemented in supervised_graph.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.featurize import ATOM_ELEMENTS

# Atom-type one-hot occupies the first slots of the atom feature vector (see src/featurize.py:
# atom_features starts with _one_hot(symbol, ATOM_ELEMENTS)), plus one trailing "other" slot.
N_ATOM_TYPES = len(ATOM_ELEMENTS) + 1


class AttributeMasking(nn.Module):
    """Mask atom types, predict them back from the encoder's node representations.

    The masked atoms' type one-hot is zeroed in the input, so the model cannot read the answer off
    the feature it is asked to predict. Everything else about the atom (degree, charge, ring
    membership) is left intact -- the task is "which element sits in this environment", not
    "reconstruct the whole atom".
    """

    def __init__(self, encoder, mask_rate: float = 0.15, hidden: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.mask_rate = float(mask_rate)
        dim = hidden or encoder.out_dim
        self.atom_head = nn.Linear(dim, N_ATOM_TYPES)
        self.ce = nn.CrossEntropyLoss()

    def node_representations(self, data) -> torch.Tensor:
        """Per-node embeddings -- the encoder's forward pools, so recompute without pooling."""
        h = self.encoder.atom_lin(data.x)
        for conv, norm, bond_lin in zip(self.encoder.convs, self.encoder.norms, self.encoder.bond_lins):
            e = bond_lin(data.edge_attr)
            h = self.encoder.dropout(torch.relu(norm(conv(h, data.edge_index, e))))
        return h

    def loss(self, batch, model=None) -> torch.Tensor:
        """Cross-entropy on the masked atoms' true element. `model` is unused (interface parity)."""
        x = batch.x
        n_nodes = x.size(0)
        if n_nodes == 0:
            return x.sum() * 0.0

        n_mask = max(1, int(round(self.mask_rate * n_nodes)))
        idx = torch.randperm(n_nodes, device=x.device)[:n_mask]

        # True label is the argmax over the atom-type block, before it is erased.
        target = x[idx, :N_ATOM_TYPES].argmax(dim=1)

        masked_x = x.clone()
        masked_x[idx, :N_ATOM_TYPES] = 0.0
        masked = batch.clone()
        masked.x = masked_x

        h = self.node_representations(masked)
        return self.ce(self.atom_head(h[idx]), target)


def build(cfg: dict):
    """Interface entry point. `cfg` carries the encoder plus method hyperparameters."""
    encoder = cfg["encoder"]
    return AttributeMasking(encoder, mask_rate=float(cfg.get("mask_rate", 0.15)))
