"""Loading a pretrained encoder into the downstream model.

The SSL comparison is only meaningful if the pretrained weights actually arrive and the prediction
head does not. A silent no-op here would make "pretrained" and "supervised" the same run with
different labels -- the exact way a fake SSL gain gets published.
"""
import pytest
import torch

from src.models.gnn import GINEEncoder, PropertyPredictor
from src.train import load_pretrained_encoder


def _save_ckpt(tmp_path, hidden=64, layers=3):
    enc = GINEEncoder(hidden=hidden, layers=layers)
    # Perturb so the weights are distinguishable from a fresh init.
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(1.0)
    path = tmp_path / "enc.pt"
    torch.save({"encoder_state": enc.state_dict(), "hidden": hidden, "layers": layers}, path)
    return path, enc


def test_weights_actually_transfer(tmp_path):
    path, saved = _save_ckpt(tmp_path)
    model = PropertyPredictor(n_tasks=1, hidden=64, layers=3)

    before = model.encoder.atom_lin.weight.clone()
    state = load_pretrained_encoder(str(path), {"hidden": 64, "layers": 3})
    model.encoder.load_state_dict(state)

    assert not torch.equal(model.encoder.atom_lin.weight, before)
    assert torch.equal(model.encoder.atom_lin.weight, saved.atom_lin.weight)


def test_prediction_head_is_not_transferred(tmp_path):
    """The head must stay randomly initialized; the checkpoint has no head to give it."""
    path, _ = _save_ckpt(tmp_path)
    model = PropertyPredictor(n_tasks=1, hidden=64, layers=3)
    head_before = model.head.weight.clone()
    model.encoder.load_state_dict(load_pretrained_encoder(str(path), {"hidden": 64, "layers": 3}))
    assert torch.equal(model.head.weight, head_before)


def test_architecture_mismatch_raises(tmp_path):
    """A mismatched encoder must fail loudly, not be coerced into place."""
    path, _ = _save_ckpt(tmp_path, hidden=64, layers=3)
    with pytest.raises(ValueError, match="hidden"):
        load_pretrained_encoder(str(path), {"hidden": 300, "layers": 3})
    with pytest.raises(ValueError, match="layers"):
        load_pretrained_encoder(str(path), {"hidden": 64, "layers": 5})


def _save_graph_ckpt(tmp_path, name="enc_graph.pt", hidden=64, layers=3, init_from=None, scale=2.0):
    """A checkpoint in run_pretrain_graph.py's schema, provenance keys included.

    The point of this fixture is that load_pretrained_encoder must accept the richer schema with no
    change to src/train.py, since it reads only encoder_state/hidden/layers.
    """
    enc = GINEEncoder(hidden=hidden, layers=layers)
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(scale)
    path = tmp_path / name
    torch.save({
        "encoder_state": enc.state_dict(),
        "hidden": hidden,
        "layers": layers,
        "run_tag": "0000_0000_test",
        "pretrain_stage": "graph_supervised",
        "arm": "node_graph" if init_from else "graph",
        "init_from": init_from,
        "corpus": "PCBA",
        "n_tasks": 128,
        "n_pool": 123,
        "epochs": 20,
        "final_loss": 0.01,
        "git": "deadbeef",
    }, path)
    return path, enc


def test_graph_checkpoint_loads_unchanged(tmp_path):
    """The graph-level schema must load through the existing loader untouched."""
    path, saved = _save_graph_ckpt(tmp_path)
    state = load_pretrained_encoder(str(path), {"hidden": 64, "layers": 3})
    model = PropertyPredictor(n_tasks=1, hidden=64, layers=3)
    model.encoder.load_state_dict(state)
    assert torch.equal(model.encoder.atom_lin.weight, saved.atom_lin.weight)


def test_node_graph_checkpoint_differs_from_node_only(tmp_path):
    """The 2x2's integrity claim: the arms must be genuinely different checkpoints.

    If graph-level training were a no-op, `graph` and `node_graph` would be the same run under two
    names, and the factorial would be fictitious.
    """
    node_path, node_enc = _save_ckpt(tmp_path, hidden=64, layers=3)
    graph_path, graph_enc = _save_graph_ckpt(tmp_path, name="g.pt", scale=2.0)
    ng_path, ng_enc = _save_graph_ckpt(
        tmp_path, name="ng.pt", init_from=str(node_path), scale=3.0
    )

    for p in (graph_path, ng_path):
        load_pretrained_encoder(str(p), {"hidden": 64, "layers": 3})

    assert not torch.equal(ng_enc.atom_lin.weight, node_enc.atom_lin.weight), \
        "node+graph is identical to node-only -- graph-level stage did nothing"
    assert not torch.equal(ng_enc.atom_lin.weight, graph_enc.atom_lin.weight), \
        "node+graph is identical to graph-only -- node-level init did nothing"

    ng = torch.load(ng_path, map_location="cpu", weights_only=False)
    assert ng["arm"] == "node_graph"
    assert ng["init_from"] == str(node_path), "provenance must record which node ckpt seeded it"


def test_graph_checkpoint_architecture_mismatch_raises(tmp_path):
    """The architecture guard is not bypassed by the richer schema."""
    path, _ = _save_graph_ckpt(tmp_path, hidden=64, layers=3)
    with pytest.raises(ValueError, match="hidden"):
        load_pretrained_encoder(str(path), {"hidden": 300, "layers": 3})
