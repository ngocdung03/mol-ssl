"""Metrics: masked multi-task AUROC, RMSE, calibration, selective prediction.

Masking is not a detail. Tox21 has 12 tasks and most molecules are labeled for only some of them;
missing labels arrive as NaN. Treating NaN as a negative would inflate every AUROC in the table, so
every metric here computes per-task over labeled entries only, then averages across tasks that had
enough signal to score.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score


def masked_auroc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, list[float], int]:
    """Mean AUROC over tasks, ignoring NaN labels.

    Returns (mean, per_task, n_skipped). A task is skipped when its labeled subset is single-class,
    where AUROC is undefined -- skipped, never silently scored as 0.5.
    """
    y_true = np.atleast_2d(y_true)
    y_score = np.atleast_2d(y_score)
    per_task, skipped = [], 0
    for t in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, t])
        yt = y_true[mask, t]
        if mask.sum() == 0 or len(np.unique(yt)) < 2:
            skipped += 1
            continue
        per_task.append(float(roc_auc_score(yt, y_score[mask, t])))
    mean = float(np.mean(per_task)) if per_task else float("nan")
    return mean, per_task, skipped


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.atleast_2d(y_true).ravel()
    y_pred = np.atleast_2d(y_pred).ravel()
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE over all labeled (molecule, task) pairs, equal-width bins on predicted probability."""
    y_true = np.atleast_2d(y_true).ravel()
    y_prob = np.atleast_2d(y_prob).ravel()
    mask = ~np.isnan(y_true)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) == 0:
        return float("nan")

    conf = np.where(y_prob >= 0.5, y_prob, 1.0 - y_prob)
    correct = (y_prob >= 0.5).astype(float) == y_true
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        ece += (in_bin.sum() / len(conf)) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def selective_accuracy(y_true: np.ndarray, y_prob: np.ndarray, coverages=(0.5, 0.7, 0.9, 1.0)) -> dict:
    """Accuracy when the model is allowed to abstain on its least confident predictions.

    The drug-discovery-real question: if you only act on the confident calls, how good are they?
    """
    y_true = np.atleast_2d(y_true).ravel()
    y_prob = np.atleast_2d(y_prob).ravel()
    mask = ~np.isnan(y_true)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) == 0:
        return {str(c): float("nan") for c in coverages}

    conf = np.where(y_prob >= 0.5, y_prob, 1.0 - y_prob)
    correct = ((y_prob >= 0.5).astype(float) == y_true).astype(float)
    order = np.argsort(-conf)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(order))))
        out[str(c)] = float(correct[order[:k]].mean())
    return out


def aggregate_over_seeds(values: list[float]) -> dict:
    """mean +/- std over the seed list. Never a best seed (hard rule 4)."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }
