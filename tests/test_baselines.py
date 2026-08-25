"""ECFP baseline correctness -- especially the multi-task missing-label handling."""
import numpy as np

from src.baselines import ecfp4, featurize_many, fit_predict_multitask


def test_ecfp4_is_deterministic_and_right_size():
    a = ecfp4("CC(=O)Oc1ccccc1C(=O)O")
    b = ecfp4("CC(=O)Oc1ccccc1C(=O)O")
    assert a.shape == (2048,) and np.array_equal(a, b) and a.sum() > 0


def test_ecfp4_returns_none_on_bad_smiles():
    assert ecfp4("not_a_mol[[") is None


def test_ecfp4_is_canonicalization_invariant():
    """Two SMILES for the same molecule must give the same fingerprint."""
    assert np.array_equal(ecfp4("c1ccccc1"), ecfp4("C1=CC=CC=C1"))


def test_featurize_many_reports_surviving_indices():
    X, keep = featurize_many(["CCO", "not_a_mol[[", "c1ccccc1"])
    assert X.shape == (2, 2048) and keep == [0, 2]


def test_multitask_skips_unfittable_task_as_nan():
    """A task whose training labels are single-class must come back NaN, not a constant guess."""
    rng = np.random.default_rng(0)
    X_tr = rng.integers(0, 2, size=(40, 2048)).astype(np.uint8)
    Y_tr = np.zeros((40, 2))
    Y_tr[:, 0] = rng.integers(0, 2, size=40)   # fittable
    Y_tr[:, 1] = 1.0                            # single class -> unfittable
    X_te = rng.integers(0, 2, size=(10, 2048)).astype(np.uint8)

    pred = fit_predict_multitask("rf", "classification", 0, X_tr, Y_tr, X_te)
    assert pred.shape == (10, 2)
    assert not np.isnan(pred[:, 0]).any()
    assert np.isnan(pred[:, 1]).all()


def test_multitask_ignores_nan_training_labels():
    rng = np.random.default_rng(1)
    X_tr = rng.integers(0, 2, size=(30, 2048)).astype(np.uint8)
    Y_tr = rng.integers(0, 2, size=(30, 1)).astype(float)
    Y_tr[:10, 0] = np.nan   # missing labels must not crash the fit
    X_te = rng.integers(0, 2, size=(5, 2048)).astype(np.uint8)
    pred = fit_predict_multitask("rf", "classification", 0, X_tr, Y_tr, X_te)
    assert pred.shape == (5, 1) and not np.isnan(pred).any()
