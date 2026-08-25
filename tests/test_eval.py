"""Metric correctness, especially masking. An unmasked AUROC silently inflates every table."""
import numpy as np

from src.eval import (
    aggregate_over_seeds,
    expected_calibration_error,
    masked_auroc,
    masked_rmse,
    selective_accuracy,
)


def test_masked_auroc_ignores_nan_labels():
    """NaN entries must not be scored. Treating them as 0 would change the answer."""
    y = np.array([[1.0], [0.0], [np.nan], [1.0], [0.0]])
    s = np.array([[0.9], [0.1], [0.99], [0.8], [0.2]])
    mean, per_task, skipped = masked_auroc(y, s)
    assert skipped == 0
    assert mean == 1.0  # perfect on the 4 labeled rows; the NaN row cannot help or hurt


def test_masked_auroc_skips_single_class_tasks():
    """AUROC is undefined on one class -- skip it, never score it 0.5."""
    y = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    s = np.array([[0.9, 0.5], [0.1, 0.4], [0.8, 0.6]])
    mean, per_task, skipped = masked_auroc(y, s)
    assert skipped == 1 and len(per_task) == 1
    assert not np.isnan(mean)


def test_masked_auroc_differs_from_unmasked():
    """Regression guard: if masking were dropped, this number would move."""
    # The NaN rows carry the highest scores. Scored as negatives they drag AUROC down; ignored,
    # the labeled rows are perfectly separated.
    y = np.array([[1.0], [0.0], [np.nan], [np.nan]])
    s = np.array([[0.9], [0.1], [0.99], [0.98]])
    masked, _, _ = masked_auroc(y, s)
    naive = np.nan_to_num(y, nan=0.0)
    from sklearn.metrics import roc_auc_score
    assert masked == 1.0
    assert roc_auc_score(naive.ravel(), s.ravel()) < masked


def test_masked_rmse_ignores_nan():
    y = np.array([[1.0], [3.0], [np.nan]])
    p = np.array([[1.0], [3.0], [99.0]])
    assert masked_rmse(y, p) == 0.0


def test_ece_zero_for_perfectly_calibrated_certain_predictions():
    y = np.array([[1.0], [0.0], [1.0], [0.0]])
    p = np.array([[1.0], [0.0], [1.0], [0.0]])
    assert expected_calibration_error(y, p) < 1e-9


def test_ece_positive_for_overconfident_wrong_predictions():
    y = np.array([[0.0], [0.0], [0.0], [0.0]])
    p = np.array([[0.99], [0.99], [0.99], [0.99]])
    assert expected_calibration_error(y, p) > 0.9


def test_selective_accuracy_improves_at_lower_coverage():
    """The confident half should beat the full set when confidence means anything."""
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([0.99, 0.98, 0.51, 0.49])  # last one wrong, and least confident
    out = selective_accuracy(y, p, coverages=(0.5, 1.0))
    assert out["0.5"] == 1.0
    assert out["1.0"] < out["0.5"]


def test_aggregate_reports_mean_std_not_best():
    agg = aggregate_over_seeds([0.8, 0.9, 0.7])
    assert abs(agg["mean"] - 0.8) < 1e-9
    assert agg["std"] > 0 and agg["n"] == 3
    assert agg["mean"] != max([0.8, 0.9, 0.7])


def test_aggregate_handles_nan_and_singleton():
    assert aggregate_over_seeds([float("nan")])["n"] == 0
    assert aggregate_over_seeds([0.5])["std"] == 0.0
