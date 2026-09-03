"""Tests for the graph-level supervised pretraining objective.

The failure this file is mainly guarding against is a graph-level arm that runs, converges, and
means nothing -- because it trained only the head, or masked its labels wrongly, or was
accidentally wired to node embeddings. Any of those produces a plausible loss curve and a
worthless checkpoint, which the label-budget sweep cannot distinguish from a working one.
"""
import math

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.featurize import mol_to_data
from src.models.gnn import GINEEncoder
from src.ssl.supervised_graph import build

SMILES = ["c1ccccc1C", "CCCCO", "C1CCCCC1", "CC(=O)NCC"]
N_TASKS = 6


def _batch(smiles=None, labels=None, n_tasks=N_TASKS):
    smiles = smiles or SMILES
    if labels is None:
        labels = [[1.0, 0.0] + [float("nan")] * (n_tasks - 2) for _ in smiles]
    graphs = [mol_to_data(s, y=y) for s, y in zip(smiles, labels)]
    assert all(g is not None for g in graphs), "fixture SMILES must all parse"
    return next(iter(DataLoader(graphs, batch_size=len(graphs))))


def _model(n_tasks=N_TASKS, hidden=32, layers=2):
    return build({"encoder": GINEEncoder(hidden=hidden, layers=layers), "n_tasks": n_tasks})


def test_untrained_loss_is_near_ln2():
    """Binary multi-task BCE at ~zero logits is ln 2. Far off means a wiring bug."""
    torch.manual_seed(0)
    loss = float(_model().loss(_batch()))
    assert abs(loss - math.log(2)) < 0.25, f"expected ~{math.log(2):.4f}, got {loss:.4f}"


def test_nan_labels_are_excluded_from_the_loss():
    """The load-bearing test: loss must equal BCE over observed entries alone.

    Task 0 is observed, tasks 1..n are missing. If NaN were treated as a negative, the loss would
    average in n-1 fabricated targets and diverge from the hand-computed value.
    """
    torch.manual_seed(0)
    labels = [[1.0] + [float("nan")] * (N_TASKS - 1) for _ in SMILES]
    batch = _batch(labels=labels)
    # eval() is required, not cosmetic: GINEEncoder carries BatchNorm1d, whose running statistics
    # update on every forward pass, so in train mode the reference forward below would not see the
    # same normalization as loss() did.
    model = _model().eval()

    got = model.loss(batch)
    with torch.no_grad():
        logits = model(batch)
    want = F.binary_cross_entropy_with_logits(
        logits[:, 0], torch.ones(len(SMILES))
    )
    assert torch.allclose(got, want, atol=1e-6), f"masked loss {got} != observed-only loss {want}"


def test_all_nan_batch_returns_finite_zero_loss():
    """A molecule with no measured assay must not poison the batch with NaN.

    Real PCBA rows are 39.4% missing, so sparse and all-missing batches are reachable.
    """
    labels = [[float("nan")] * N_TASKS for _ in SMILES]
    loss = _model().loss(_batch(labels=labels))
    assert torch.isfinite(loss), "all-missing batch produced a non-finite loss"
    assert float(loss) == 0.0


def test_gradient_reaches_the_encoder():
    """Pretraining that trains only the head transfers nothing."""
    torch.manual_seed(0)
    model = _model()
    model.loss(_batch()).backward()
    total = sum(
        float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None
    )
    assert total > 0.0, "no gradient reached the encoder -- only the head would be trained"


def test_operates_on_pooled_graph_embedding():
    """Output is one row per GRAPH, not per node, and is invariant to batch composition.

    Catches an accidental node-level wiring, which would emit one row per atom.
    """
    torch.manual_seed(0)
    model = _model().eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (len(SMILES), N_TASKS), f"expected graph-level rows, got {tuple(out.shape)}"
    assert out.shape[0] != batch.x.size(0), "output row count equals node count -- node-level wiring"

    # The same molecule scored alongside different neighbours must score identically.
    with torch.no_grad():
        alone = model(_batch(smiles=[SMILES[0]], labels=[[1.0, 0.0] + [float("nan")] * (N_TASKS - 2)]))
    assert torch.allclose(out[0], alone[0], atol=1e-5), "per-graph output depends on the batch"


def test_loss_decreases_when_overfitting_one_batch():
    """The objective must be learnable, not merely differentiable."""
    torch.manual_seed(0)
    model = _model()
    batch = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = float(model.loss(batch))
    for _ in range(30):
        opt.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        opt.step()
    assert float(loss) < first, f"loss did not fall: {first:.4f} -> {float(loss):.4f}"


def test_wrong_label_count_raises_loudly():
    """A label-count mismatch must not silently broadcast."""
    labels = [[1.0, 0.0, float("nan")] for _ in SMILES]  # 3 tasks, model expects N_TASKS
    batch = _batch(labels=labels, n_tasks=3)
    with pytest.raises(RuntimeError):
        _model(n_tasks=N_TASKS).loss(batch)
