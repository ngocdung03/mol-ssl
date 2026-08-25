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
