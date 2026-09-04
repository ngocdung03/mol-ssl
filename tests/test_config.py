"""Config validation must refuse to guess. Every raise here is a rule from README.md."""
import pytest
import yaml

from src.config import budget_list, load_config

BASE = {
    "dataset": "BBBP", "task": "classification", "n_tasks": 1,
    "split": "scaffold", "split_manifest": "data/splits/bbbp_scaffold.json",
    "seeds": [0, 1, 2, 3, 4],
}


def _write(tmp_path, cfg):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_valid_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, BASE))
    assert cfg["dataset"] == "BBBP" and cfg["split"] == "scaffold"


def test_random_split_config_is_rejected(tmp_path):
    bad = dict(BASE, split="random")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))


def test_missing_split_is_rejected(tmp_path):
    bad = {k: v for k, v in BASE.items() if k != "split"}
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))


def test_fewer_than_five_seeds_is_rejected(tmp_path):
    """Five seeds is the submitted commitment, so three cannot be allowed to slip through."""
    bad = dict(BASE, seeds=[0, 1, 2])
    with pytest.raises(ValueError, match="seeds"):
        load_config(_write(tmp_path, bad))


def test_unknown_dataset_and_task_rejected(tmp_path):
    with pytest.raises(ValueError, match="dataset"):
        load_config(_write(tmp_path, dict(BASE, dataset="QM9")))
    with pytest.raises(ValueError, match="task"):
        load_config(_write(tmp_path, dict(BASE, task="ranking")))


def test_out_of_range_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        load_config(_write(tmp_path, dict(BASE, label_budget=1.5)))
    with pytest.raises(ValueError, match="budget"):
        load_config(_write(tmp_path, dict(BASE, label_budget=0.0)))


def test_budget_list_handles_single_and_sweep():
    assert budget_list({"label_budget": 0.25}) == [0.25]
    assert budget_list({"label_budgets": [0.05, 1.0]}) == [0.05, 1.0]
    assert budget_list({}) == [1.0]


def test_shipped_configs_are_valid():
    """The two committed configs must actually load -- they were aspirational until now."""
    for p in ("configs/baseline_bbbp_gine.yaml", "configs/labelbudget_tox21.yaml"):
        assert load_config(p)["split"] == "scaffold"


def test_pcba_is_not_a_downstream_dataset(tmp_path):
    """PCBA is a pretraining corpus, not an evaluation set.

    It has no scaffold split manifest, needs no label budget, and is never evaluated. Whitelisting
    it would force either a fabricated split_manifest or a weakened required-key check --
    KNOWN_DATASETS lists what may be *evaluated on*. run_pretrain_graph.py therefore takes argparse
    flags and never calls load_config, exactly as run_pretrain.py does.
    """
    from src.config import KNOWN_DATASETS

    assert "PCBA" not in KNOWN_DATASETS
    with pytest.raises(ValueError, match="dataset"):
        load_config(_write(tmp_path, dict(BASE, dataset="PCBA")))


def test_graph_pretrain_work_did_not_relax_downstream_invariants(tmp_path):
    """Regression lock: the 2x2 extension must not have loosened any hard rule."""
    # Rule 1: split still mandatory and still scaffold-only.
    no_split = {k: v for k, v in BASE.items() if k != "split"}
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, no_split))
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, dict(BASE, split="random")))
    # split_manifest still required.
    no_manifest = {k: v for k, v in BASE.items() if k != "split_manifest"}
    with pytest.raises(ValueError, match="split_manifest"):
        load_config(_write(tmp_path, no_manifest))
    # Rule 4: still >=5 seeds.
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, dict(BASE, seeds=[0, 1, 2])))
