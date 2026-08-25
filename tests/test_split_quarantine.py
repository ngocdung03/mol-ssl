"""The random split must stay quarantined.

Phase 2 needs a random split to measure how much random splitting inflates a benchmark. Adding one
is exactly the change that could quietly destroy rule 1, so these tests pin the containment:
the production path must still refuse random splits after the comparison module exists.
"""
import pytest

from src.datamodule import require_scaffold_split
from src.splits import load_split
from src.splits_random_for_comparison import (
    MANIFEST_TYPE,
    load_random_split,
    random_split,
    save_random_split,
)


def test_production_loader_refuses_a_random_manifest(tmp_path):
    """The core containment: a random manifest cannot enter the training pipeline."""
    p = tmp_path / "rand.json"
    save_random_split(p, "toy", random_split(100, seed=0), seed=0)
    with pytest.raises(ValueError, match="non-scaffold"):
        load_split(p)


def test_require_scaffold_split_still_rejects_random():
    """Rule 1 must not have been relaxed to accommodate the comparison path."""
    with pytest.raises(ValueError):
        require_scaffold_split({"split": "random"})
    with pytest.raises(ValueError):
        require_scaffold_split({"split": MANIFEST_TYPE})
    assert require_scaffold_split({"split": "scaffold"}) == "scaffold"


def test_splits_module_exposes_no_random_function():
    """There must be no random-split function on the production module to reach for."""
    import src.splits as splits

    assert not [n for n in dir(splits) if "random" in n.lower()]


def test_random_split_is_seeded_and_partitions_completely():
    a = random_split(100, seed=0)
    b = random_split(100, seed=0)
    c = random_split(100, seed=1)
    assert a == b and a != c
    all_idx = sorted(a["train"] + a["val"] + a["test"])
    assert all_idx == list(range(100))
    for x in ("train", "val", "test"):
        for y in ("train", "val", "test"):
            if x != y:
                assert not (set(a[x]) & set(a[y]))


def test_comparison_loader_refuses_a_scaffold_manifest(tmp_path):
    """Containment runs both ways: the comparison path will not silently read a real split."""
    from src.splits import save_split, scaffold_split

    smiles = ["c1ccccc1C", "c1ccccc1CC", "CCCCO", "C1CCCCC1"]
    p = tmp_path / "scaf.json"
    save_split(p, "toy", scaffold_split(smiles, 0.5, 0.25, 0.25), smiles)
    with pytest.raises(ValueError, match="not a comparison manifest"):
        load_random_split(p)


def test_random_manifest_carries_a_warning(tmp_path):
    p = tmp_path / "rand.json"
    save_random_split(p, "toy", random_split(50, seed=3), seed=3)
    payload = load_random_split(p)
    assert payload["split_type"] == MANIFEST_TYPE
    assert "never use" in payload["warning"].lower()
