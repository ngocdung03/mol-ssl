"""Non-deep baselines: ECFP4 fingerprints + RandomForest / XGBoost.

These exist to keep the graph network honest. A GNN that cannot beat a 2048-bit fingerprint and a
random forest has not earned its complexity, and that comparison is the first thing a reviewer
asks for. Same scaffold splits, same seeds, same metrics as the deep path.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ecfp4(smiles: str) -> np.ndarray | None:
    """ECFP4 (Morgan radius 2), 2048 bits. None if RDKit cannot parse the SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.asarray(_GEN.GetFingerprintAsNumPy(mol), dtype=np.uint8)


def featurize_many(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Fingerprint matrix plus the indices that survived parsing."""
    rows, keep = [], []
    for i, smi in enumerate(smiles_list):
        fp = ecfp4(smi)
        if fp is not None:
            rows.append(fp)
            keep.append(i)
    if not rows:
        return np.empty((0, 2048), dtype=np.uint8), []
    return np.vstack(rows), keep


def build_model(kind: str, task: str, seed: int):
    """RandomForest or XGBoost, classification or regression."""
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(n_estimators=500, n_jobs=-1, random_state=seed)
    if kind == "xgb":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if task == "classification" else XGBRegressor
        return cls(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=seed, tree_method="hist",
            **({"eval_metric": "logloss"} if task == "classification" else {}),
        )
    raise ValueError(f"unknown baseline {kind!r}; expected 'rf' or 'xgb'")


def fit_predict_multitask(kind: str, task: str, seed: int,
                          X_tr: np.ndarray, Y_tr: np.ndarray,
                          X_te: np.ndarray) -> np.ndarray:
    """One independent model per task, since tasks have different missing-label patterns.

    A task whose training labels are all one class cannot be fit; its column is returned as NaN so
    the masked metric skips it rather than scoring a constant.
    """
    Y_tr = np.atleast_2d(Y_tr)
    n_tasks = Y_tr.shape[1]
    out = np.full((X_te.shape[0], n_tasks), np.nan, dtype=float)

    for t in range(n_tasks):
        mask = ~np.isnan(Y_tr[:, t])
        y = Y_tr[mask, t]
        if mask.sum() == 0:
            continue
        if task == "classification" and len(np.unique(y)) < 2:
            continue
        model = build_model(kind, task, seed)
        model.fit(X_tr[mask], y)
        if task == "classification":
            out[:, t] = model.predict_proba(X_te)[:, 1]
        else:
            out[:, t] = model.predict(X_te)
    return out
