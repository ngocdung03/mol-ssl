"""Graph-level supervised multi-task pretraining (Hu et al. 2019, arXiv 1905.12265, section 3.3).

Hu et al.'s second stage: after node-level self-supervision, fit a linear head on the POOLED graph
embedding against a large multi-task assay panel -- here PCBA, 128 binary bioassays. The head is
discarded afterwards; only the encoder transfers downstream.

Why this exists: the repository's first headline result was a null for attribute masking
(results/FINDING_label_budget.md), which is a NODE-level objective. Hu et al. Table 1 reports that
node-level-only pretraining is exactly what underperforms (71.4 avg ROC-AUC) while node-level plus
graph-level reaches 73.5, against a 67.0 non-pretrained baseline. This module supplies the
graph-level factor so all four cells of the 2x2 can be measured.

Sibling of pretrain.AttributeMasking: same build()/loss() contract. The difference is that
GINEEncoder.forward already ends in global_mean_pool, which IS the graph-level vector this
objective wants -- so unlike AttributeMasking there is no un-pooled recomputation here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.train import masked_bce

N_PCBA_TASKS = 128


class SupervisedGraphPretrain(nn.Module):
    """Linear head on the pooled graph embedding -> n_tasks binary assay logits.

    Masked BCE over observed labels only. PCBA is 39.4% missing (77.6 of 128 assays observed per
    molecule, measured on the scaffold-filtered 419,118-molecule corpus), so treating a missing
    assay as a negative would pretrain the encoder on roughly 6.4 million fabricated negatives. `masked_bce` is imported from the downstream training loop
    rather than reimplemented: the pretext and downstream objectives must treat missing labels
    identically, or the transfer comparison is confounded by a loss-definition difference.

    A linear head only, matching Hu et al.'s "Supervised" arm. Hidden layers would be an
    unregistered deviation from the strategy being replicated.
    """

    def __init__(self, encoder, n_tasks: int = N_PCBA_TASKS):
        super().__init__()
        self.encoder = encoder
        self.n_tasks = int(n_tasks)
        self.head = nn.Linear(encoder.out_dim, self.n_tasks)

    def forward(self, data) -> torch.Tensor:
        return self.head(self.encoder(data))

    def loss(self, batch, model=None) -> torch.Tensor:
        """Masked BCE on the assay panel. `model` is unused (interface parity with pretrain.build).

        The explicit view is defensive: PyG already collates y to (num_graphs, n_tasks) because
        mol_to_data stores it as view(1, -1). Reshaping here turns a silent broadcast against a
        wrong label count into a loud RuntimeError.
        """
        y = batch.y.view(batch.num_graphs, self.n_tasks)
        return masked_bce(self(batch), y)


def build(cfg: dict):
    """Interface entry point. `cfg` carries the encoder plus method hyperparameters."""
    encoder = cfg["encoder"]
    return SupervisedGraphPretrain(encoder, n_tasks=int(cfg.get("n_tasks", N_PCBA_TASKS)))
