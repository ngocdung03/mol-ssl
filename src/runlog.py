"""Run tagging and the append-only experiment ledger.

Ported from ../../../Care/ (`runlog.resolve_keyword()` + `artifacts/experiments.csv`). The point is
that the record of what was run lives outside the model and outside any dashboard: one CSV row per
run, greppable, committed.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "artifacts" / "experiments.csv"
LOG_DIR = ROOT / "artifacts" / "logs"
RESULTS_DIR = ROOT / "results"

_KW_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$")


def resolve_keyword(kw: str | None) -> str:
    """Require a run keyword. Prompt on a TTY, hard-error when detached.

    No default, no auto-generated name: an untagged run cannot be found again, and a run that
    cannot be found again cannot be cited in the README.
    """
    if kw is None:
        if sys.stdin.isatty():
            kw = input("run keyword (--kw): ").strip()
        else:
            raise SystemExit("--kw is required (detached run, cannot prompt)")
    kw = kw.strip().lower()
    if not _KW_RE.match(kw):
        raise SystemExit(f"invalid --kw {kw!r}: use [a-z0-9][a-z0-9_-]{{1,39}}")
    return kw


def make_run_tag(kw: str) -> str:
    return f"{datetime.now().strftime('%m%d_%H%M')}_{kw}"


def log_path(run_tag: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{run_tag}.log"


def write_metrics(run_tag: str, metrics: dict) -> Path:
    """Every number that reaches the README must come from one of these files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"metrics_{run_tag}.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return path


def append_ledger(row: dict) -> None:
    """One row per run, append-only. New keys extend the header; nothing is ever overwritten."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    existing_header: list[str] = []
    rows: list[dict] = []
    if LEDGER.exists():
        with LEDGER.open() as fh:
            reader = csv.DictReader(fh)
            existing_header = list(reader.fieldnames or [])
            rows = list(reader)

    header = existing_header + [k for k in row if k not in existing_header]
    with LEDGER.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for old in rows:
            writer.writerow(old)
        writer.writerow({k: row.get(k, "") for k in header})


def git_rev() -> str:
    out = os.popen("git -C %s rev-parse --short HEAD 2>/dev/null" % ROOT).read().strip()
    return out or "nogit"
