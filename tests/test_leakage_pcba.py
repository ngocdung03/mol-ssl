"""Leakage tests for the *labeled* pretraining corpus (PCBA, graph-level arm).

`tests/test_leakage.py` covers the unlabeled SMILES pool. The graph-level arm of the node-vs-graph
2x2 pretrains on PCBA, which carries 128 assay labels per molecule, so the filter has to drop
molecules *and* keep the surviving labels aligned with the surviving SMILES.

Misalignment is the dangerous failure here: an off-by-one pairs molecules with other molecules'
assay labels, and pretraining still converges to a plausible-looking loss curve. That produces a
graph-level arm which is meaningless but indistinguishable from a working one by any metric the
sweep reports. `test_labels_survive_the_filter_aligned` is the test for that.

Same discipline as test_leakage.py: `test_unfiltered_corpus_would_leak` proves the contamination is
real first, so a no-op filter cannot make the suite green. `scripts/verify_leakage_teeth.sh` mutates
`filter_pool_indices` and requires this file to fail.
"""
import pytest

from src.datamodule import filter_pool_by_scaffold, filter_pool_indices
from src.splits import heldout_scaffolds, load_split, save_split, scaffold_smiles, scaffold_split

SMILES = ["c1ccccc1C", "c1ccccc1CC", "c1ccc2ccccc2c1", "c1ccc2ccccc2c1C", "CCCCO", "C1CCCCC1"]
N_TASKS = 128


@pytest.fixture
def manifest(tmp_path):
    split = scaffold_split(SMILES, 0.5, 0.25, 0.25)
    path = tmp_path / "toy.json"
    save_split(path, "toy", split, SMILES)
    return load_split(path)


def _labeled_corpus(manifest):
    """A labeled corpus mixing leaking and clean molecules.

    Each molecule gets a distinct sentinel label row (all 128 entries = its own position), so a
    misalignment after filtering is detectable by inspecting any surviving row.
    """
    banned = heldout_scaffolds(manifest)
    assert banned, "fixture produced no held-out scaffolds -- test would be vacuous"
    leaks = [s for s in SMILES if scaffold_smiles(s) in banned]
    assert leaks, "fixture produced no leaking molecules -- test would be vacuous"
    clean = ["CCCCCCN", "CC(=O)NCC", "CCCCCCCC"]
    corpus = leaks + clean
    labels = [[float(i)] * N_TASKS for i in range(len(corpus))]
    return corpus, labels, len(leaks)


def test_unfiltered_corpus_would_leak(manifest):
    """Proves the contamination exists, so the filter tests cannot pass trivially."""
    corpus, _labels, n_leaks = _labeled_corpus(manifest)
    banned = heldout_scaffolds(manifest)
    overlap = [s for s in corpus if scaffold_smiles(s) in banned]
    assert len(overlap) == n_leaks


def test_index_filter_agrees_with_smiles_filter(manifest):
    """The labeled path and the audited SMILES path must drop the same molecules."""
    corpus, _labels, _n = _labeled_corpus(manifest)
    kept_idx, dropped_idx = filter_pool_indices(corpus, [manifest])
    kept_smi, dropped_smi = filter_pool_by_scaffold(corpus, [manifest])
    assert dropped_idx == dropped_smi
    assert [corpus[i] for i in kept_idx] == kept_smi


def test_labels_survive_the_filter_aligned(manifest):
    """Each surviving label row must still belong to its own molecule.

    The sentinel row for molecule i is [i]*128, so row content identifies the molecule it came
    from. If the filter shifted labels relative to SMILES, this fails.
    """
    corpus, labels, _n = _labeled_corpus(manifest)
    kept_idx, _dropped = filter_pool_indices(corpus, [manifest])

    kept_smiles = [corpus[i] for i in kept_idx]
    kept_labels = [labels[i] for i in kept_idx]
    assert len(kept_smiles) == len(kept_labels)

    for smi, row in zip(kept_smiles, kept_labels):
        sentinel = int(row[0])
        assert row == [float(sentinel)] * N_TASKS, "label row was corrupted, not merely reordered"
        assert corpus[sentinel] == smi, f"label row {sentinel} landed on the wrong molecule {smi}"


def test_contaminated_labeled_corpus_is_caught(manifest):
    corpus, _labels, n_leaks = _labeled_corpus(manifest)
    kept_idx, dropped = filter_pool_indices(corpus, [manifest])
    assert dropped == n_leaks
    banned = heldout_scaffolds(manifest)
    assert all(scaffold_smiles(corpus[i]) not in banned for i in kept_idx)


def test_index_filter_drops_unparseable(manifest):
    """An unknown scaffold cannot be proven clean, so it goes -- and takes its labels with it."""
    corpus = ["CCCCCCN", "not_a_mol[[", "CC(=O)NCC"]
    kept_idx, dropped = filter_pool_indices(corpus, [manifest])
    assert dropped == 1
    assert [corpus[i] for i in kept_idx] == ["CCCCCCN", "CC(=O)NCC"]
