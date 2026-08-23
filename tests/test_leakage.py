"""The leakage regression test. Ported in spirit from Care/NNUNet/tests/test_lifs_leakage.py.

A leakage test that can only pass is not a test. `test_filter_has_teeth` deliberately feeds a
contaminated pool and asserts the filter catches it; `test_unfiltered_pool_would_leak` asserts the
contamination is real in the first place, so a no-op filter cannot make the suite green.
"""
import pytest

from src.datamodule import filter_pool_by_scaffold, label_budget_subsample, require_scaffold_split
from src.splits import heldout_scaffolds, save_split, scaffold_smiles, scaffold_split

SMILES = ["c1ccccc1C", "c1ccccc1CC", "c1ccc2ccccc2c1", "c1ccc2ccccc2c1C", "CCCCO", "C1CCCCC1"]


@pytest.fixture
def manifest(tmp_path):
    from src.splits import load_split

    split = scaffold_split(SMILES, 0.5, 0.25, 0.25)
    path = tmp_path / "toy.json"
    save_split(path, "toy", split, SMILES)
    return load_split(path)


def _contaminated_pool(manifest):
    """A pool containing molecules that share a scaffold with the held-out sets."""
    banned = sorted(heldout_scaffolds(manifest))
    assert banned, "fixture produced no held-out scaffolds — test would be vacuous"
    leaks = [s for s in SMILES if scaffold_smiles(s) in set(banned)]
    assert leaks, "fixture produced no leaking molecules — test would be vacuous"
    clean = ["CCCCCCN", "CC(=O)NCC"]
    return leaks + clean, len(leaks)


def test_unfiltered_pool_would_leak(manifest):
    """Proves the contamination exists, so the next test cannot pass trivially."""
    pool, n_leaks = _contaminated_pool(manifest)
    banned = heldout_scaffolds(manifest)
    overlap = [s for s in pool if scaffold_smiles(s) in banned]
    assert len(overlap) == n_leaks


def test_filter_has_teeth(manifest):
    pool, n_leaks = _contaminated_pool(manifest)
    kept, dropped = filter_pool_by_scaffold(pool, [manifest])
    assert dropped == n_leaks
    banned = heldout_scaffolds(manifest)
    assert all(scaffold_smiles(s) not in banned for s in kept)


def test_pool_filter_drops_unparseable(manifest):
    kept, dropped = filter_pool_by_scaffold(["CCCCCCN", "not_a_mol[["], [manifest])
    assert kept == ["CCCCCCN"]
    assert dropped == 1


def test_random_split_config_is_rejected():
    with pytest.raises(ValueError):
        require_scaffold_split({"split": "random"})
    with pytest.raises(ValueError):
        require_scaffold_split({})
    assert require_scaffold_split({"split": "scaffold"}) == "scaffold"


def test_label_budget_stays_inside_train_and_is_seed_reproducible():
    train = list(range(100))
    a = label_budget_subsample(train, 0.05, seed=0)
    b = label_budget_subsample(train, 0.05, seed=0)
    c = label_budget_subsample(train, 0.05, seed=1)
    assert a == b and a != c
    assert len(a) == 5
    assert set(a) <= set(train)
