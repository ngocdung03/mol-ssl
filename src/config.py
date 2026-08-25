"""Config loading and validation.

Every run is driven by a YAML file. The point is rule 4: a result that cannot be traced back to a
config and a seed is not a result. Validation is strict and refuses to guess -- in particular
`require_scaffold_split` runs on every load, so a config that omits `split:` cannot silently
default to something permissive.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.datamodule import require_scaffold_split

TASK_TYPES = {"classification", "regression"}
KNOWN_DATASETS = {"BBBP", "BACE", "Tox21", "Lipo"}


def load_config(path: str | Path) -> dict:
    """Read a YAML config and validate it. Raises rather than defaulting."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")

    require_scaffold_split(cfg)  # rule 1, on every load

    for key in ("dataset", "task", "split_manifest", "seeds"):
        if key not in cfg:
            raise ValueError(f"{path}: config is missing required key {key!r}")

    if cfg["dataset"] not in KNOWN_DATASETS:
        raise ValueError(f"{path}: unknown dataset {cfg['dataset']!r}; expected one of {sorted(KNOWN_DATASETS)}")
    if cfg["task"] not in TASK_TYPES:
        raise ValueError(f"{path}: task must be one of {sorted(TASK_TYPES)}, got {cfg['task']!r}")

    seeds = cfg["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 5:
        raise ValueError(f"{path}: need >=5 seeds (submitted commitment), got {seeds!r}")

    budgets = budget_list(cfg)
    for b in budgets:
        if not 0.0 < float(b) <= 1.0:
            raise ValueError(f"{path}: label budget must be in (0, 1], got {b}")

    cfg.setdefault("config_path", str(path))
    return cfg


def budget_list(cfg: dict) -> list[float]:
    """Label budgets, whether the config names one (`label_budget`) or a sweep (`label_budgets`)."""
    if "label_budgets" in cfg:
        return [float(b) for b in cfg["label_budgets"]]
    return [float(cfg.get("label_budget", 1.0))]
