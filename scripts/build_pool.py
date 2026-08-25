#!/usr/bin/env python
"""Build the unlabeled pretraining pool, scaffold-filtered BEFORE it is written to disk.

Hard rule 2: the pool must contain no scaffold appearing in any downstream val or test split. The
filter runs here, in the pipeline, not as a comment and not as a later cleanup step -- a pool file
that has already been written is a pool file someone will reuse unfiltered.

Source: ZINC250k via the PyG MoleculeNet-adjacent download, or any newline-delimited SMILES file
passed with --from-file.

Usage:
  python scripts/build_pool.py --from-file data/raw/zinc250k.smi --max 250000 --kw pool_zinc
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datamodule import filter_pool_by_scaffold
from src.runlog import append_ledger, git_rev, make_run_tag, resolve_keyword, write_metrics
from src.splits import load_split

DOWNSTREAM_MANIFESTS = [
    "data/splits/tox21_scaffold.json",
    "data/splits/bbbp_scaffold.json",
    "data/splits/bace_scaffold.json",
    "data/splits/lipophilicity_scaffold.json",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", required=True, help="newline-delimited SMILES")
    ap.add_argument("--out", default="data/pool/pool_filtered.txt")
    ap.add_argument("--max", type=int, default=250000, help="T4 budget cap")
    ap.add_argument("--kw", default=None)
    args = ap.parse_args()

    kw = resolve_keyword(args.kw)
    run_tag = make_run_tag(kw)

    raw = [s.strip() for s in Path(args.from_file).read_text().splitlines() if s.strip()]
    raw = raw[: args.max]
    print(f"[{run_tag}] read {len(raw)} SMILES (cap {args.max})")

    manifests = [load_split(p) for p in DOWNSTREAM_MANIFESTS if Path(p).exists()]
    if not manifests:
        print("REFUSING: no downstream split manifests found; cannot prove the pool is clean.")
        return 1
    print(f"filtering against {len(manifests)} downstream manifests")

    kept, dropped = filter_pool_by_scaffold(raw, manifests)
    print(f"kept {len(kept)}, dropped {dropped} "
          f"({dropped / max(1, len(raw)):.2%} held-out scaffolds or unparseable)")

    # Verify before writing: the file must be clean at the moment it is created.
    recheck_kept, recheck_dropped = filter_pool_by_scaffold(kept, manifests)
    if recheck_dropped != 0:
        print(f"REFUSING to write: re-check found {recheck_dropped} leaking molecules")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\n")
    print(f"wrote {out}")

    metrics = {
        "run_tag": run_tag, "script": "build_pool", "source": args.from_file,
        "n_input": len(raw), "n_kept": len(kept), "n_dropped": dropped,
        "manifests": DOWNSTREAM_MANIFESTS, "out": str(out),
        "leak_recheck_dropped": recheck_dropped, "git": git_rev(),
    }
    write_metrics(run_tag, metrics)
    append_ledger({
        "run_tag": run_tag, "script": "build_pool", "dataset": "pool",
        "ssl_method": "none", "n_train": len(kept),
        "n_pool_dropped": dropped, "git": git_rev(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
