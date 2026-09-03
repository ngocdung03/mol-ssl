"""Arm-to-checkpoint wiring for the multi-arm sweep.

The sweep's whole validity rests on each arm loading the checkpoint it claims to. A silent
mis-mapping would label one encoder as another arm, which no downstream metric can detect.
Backward compatibility matters too: the committed label-budget run used a bare --checkpoint, and
that command must keep working or the earlier result stops being reproducible.
"""
import pytest

from scripts.run_label_budget_sweep import parse_checkpoint_map


def test_bare_path_maps_to_pretrain_for_backward_compatibility():
    """The original committed invocation must still work verbatim."""
    assert parse_checkpoint_map(["artifacts/encoder_pretrained.pt"]) == {
        "pretrain": "artifacts/encoder_pretrained.pt"
    }


def test_named_arms_parse():
    got = parse_checkpoint_map([
        "graph=artifacts/encoder_graph.pt",
        "node_graph=artifacts/encoder_node_graph.pt",
    ])
    assert got == {
        "graph": "artifacts/encoder_graph.pt",
        "node_graph": "artifacts/encoder_node_graph.pt",
    }


def test_named_and_bare_can_mix():
    got = parse_checkpoint_map(["enc.pt", "graph=g.pt"])
    assert got == {"pretrain": "enc.pt", "graph": "g.pt"}


def test_duplicate_arm_is_rejected():
    """Two checkpoints for one arm is ambiguous; guessing which wins would be silent corruption."""
    with pytest.raises(SystemExit, match="twice"):
        parse_checkpoint_map(["graph=a.pt", "graph=b.pt"])


def test_empty_path_is_rejected():
    with pytest.raises(SystemExit, match="empty path"):
        parse_checkpoint_map(["graph="])


def test_no_checkpoints_is_empty_not_an_error():
    """An all-random-init sweep (--arms none) needs no checkpoint at all."""
    assert parse_checkpoint_map([]) == {}
