"""Attribute-masking pretraining (Hu et al. 2020).

The failure mode to guard is subtle: if the masked atom's type is still visible in its own input
features, the task is trivial and the encoder learns nothing while the loss still drops. These
tests pin the masking down rather than only checking that a number comes out.
"""
import math

import torch
from torch_geometric.loader import DataLoader

from src.featurize import mol_to_data
from src.models.gnn import GINEEncoder
from src.ssl.pretrain import N_ATOM_TYPES, AttributeMasking, build

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "c1ccccc1", "CCO"]


def _batch():
    graphs = [mol_to_data(s) for s in SMILES]
    return next(iter(DataLoader(graphs, batch_size=len(graphs))))


def test_untrained_loss_is_near_uniform_cross_entropy():
    """An untrained 44-way classifier should sit near ln(44); far off means a wiring bug."""
    torch.manual_seed(0)
    m = build({"encoder": GINEEncoder(hidden=64, layers=3), "mask_rate": 0.25})
    loss = float(m.loss(_batch()))
    assert abs(loss - math.log(N_ATOM_TYPES)) < 1.5


def test_gradient_reaches_the_encoder():
    """Pretraining is worthless if it only trains the auxiliary head."""
    enc = GINEEncoder(hidden=64, layers=3)
    m = AttributeMasking(enc, mask_rate=0.5)
    m.loss(_batch()).backward()
    total = sum(p.grad.abs().sum() for p in enc.parameters() if p.grad is not None)
    assert float(total) > 0


def test_masking_erases_the_answer_from_the_input():
    """The masked atom's type block must be zeroed, or the task leaks its own label."""
    enc = GINEEncoder(hidden=32, layers=2)
    m = AttributeMasking(enc, mask_rate=1.0)  # mask everything, so the check is total
    batch = _batch()
    x = batch.x.clone()

    captured = {}
    original = m.node_representations

    def spy(data):
        captured["x"] = data.x.clone()
        return original(data)

    m.node_representations = spy
    m.loss(batch)

    assert captured["x"][:, :N_ATOM_TYPES].abs().sum() == 0, "atom types still visible after masking"
    # Non-type features must survive: the task is element identity, not full reconstruction.
    assert captured["x"][:, N_ATOM_TYPES:].abs().sum() > 0
    # The caller's batch must not be mutated in place.
    assert torch.equal(batch.x, x)


def test_mask_rate_controls_how_many_atoms_are_masked():
    enc = GINEEncoder(hidden=32, layers=2)
    batch = _batch()
    n_nodes = batch.x.size(0)
    for rate in (0.1, 0.5):
        m = AttributeMasking(enc, mask_rate=rate)
        captured = {}
        original = m.node_representations
        m.node_representations = lambda d, o=original, c=captured: (
            c.__setitem__("n_masked", int((d.x[:, :N_ATOM_TYPES].sum(dim=1) == 0).sum())) or o(d)
        )
        m.loss(batch)
        assert abs(captured["n_masked"] - round(rate * n_nodes)) <= 1


def test_loss_decreases_when_overfitting_one_batch():
    """The objective must be learnable, not merely differentiable."""
    torch.manual_seed(0)
    enc = GINEEncoder(hidden=64, layers=3, dropout=0.0)
    m = AttributeMasking(enc, mask_rate=0.15)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    batch = _batch()

    torch.manual_seed(0)
    first = float(m.loss(batch))
    for _ in range(60):
        opt.zero_grad()
        loss = m.loss(batch)
        loss.backward()
        opt.step()
    torch.manual_seed(0)
    last = float(m.loss(batch))
    assert last < first
