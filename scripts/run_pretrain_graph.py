#!/usr/bin/env python
"""Graph-level supervised multi-task pretraining on PCBA (Hu et al. 2019, section 3.3, Table 1).

The repository's first headline result was a null for attribute masking, a NODE-level objective
(results/FINDING_label_budget.md). Hu et al. report that node-level-only pretraining is exactly
what underperforms -- 71.4 average ROC-AUC against a 67.0 non-pretrained baseline -- while
node-level plus graph-level reaches 73.5. This script supplies the graph-level factor.

Two arms come out of this one script, selected by --init-from:
  * graph      : random init                              -> Hu's "Supervised" row (68.9)
  * node_graph : --init-from artifacts/encoder_pretrained.pt
                 -> Hu's "Supervised + AttrMasking" row (73.5)
Hu's order is node-level self-supervision FIRST, then graph-level supervised, so --init-from is
what makes this the SECOND stage rather than an alternative to the first.

LEAKAGE (hard rule 2, and Hu section 5.1: "all test graphs used for performance evaluation are
removed from the graph-level supervised pre-training datasets"): PCBA molecules sharing a
Bemis-Murcko scaffold with ANY downstream val/test split are removed before training. Filtered
once into a cache, but the invariant is re-asserted on every run rather than trusted -- a corpus
built once and reused silently is exactly how leakage survives.

The filter runs against all four downstream manifests, not just Tox21, even though only Tox21 is
evaluated here. Rule 2 is written as "any downstream val/test split", and a checkpoint clean for
only one dataset is a trap for whoever reuses it. Per-manifest drop counts go into the metrics JSON
so the more permissive Tox21-only counterfactual stays recoverable.

Usage:
  python scripts/run_pretrain_graph.py --kw pretrain_pcba_graph --out artifacts/encoder_graph.pt
  python scripts/run_pretrain_graph.py --kw pretrain_pcba_nodegraph \
      --init-from artifacts/encoder_pretrained.pt --out artifacts/encoder_node_graph.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.datamodule import filter_pool_indices, load_moleculenet
from src.featurize import mol_to_data
from src.models.gnn import GINEEncoder
from src.runlog import append_ledger, git_rev, make_run_tag, resolve_keyword, write_metrics
from src.splits import heldout_scaffolds, load_split

DOWNSTREAM_MANIFESTS = [
    "data/splits/tox21_scaffold.json",
    "data/splits/bbbp_scaffold.json",
    "data/splits/bace_scaffold.json",
    "data/splits/lipophilicity_scaffold.json",
]


def load_manifests():
    """Downstream manifests, or None if none exist (cannot prove a corpus clean without them)."""
    manifests = [load_split(p) for p in DOWNSTREAM_MANIFESTS if Path(p).exists()]
    if not manifests:
        print("REFUSING: no downstream split manifests found; cannot prove the corpus is clean.")
        return None
    return manifests


def build_cache(cache_path: Path, root: str, manifests, limit: int | None = None) -> dict:
    """Scaffold-filter PCBA once and save the surviving SMILES with their labels.

    Cached rather than recomputed mainly so the leakage-filtered corpus is a durable, inspectable
    artifact, the same discipline as data/pool/zinc250k_filtered.txt. Labels are stored fp16: PCBA
    values are only 0.0, 1.0 and NaN, all exactly representable.
    """
    ds = load_moleculenet("PCBA", root=root)
    n_raw = len(ds)
    if limit:
        print(f"NOTE: --limit {limit} active; this is a smoke-test corpus, not a real run")

    smiles, labels = [], []
    for i, d in enumerate(ds):
        if limit and i >= limit:
            break
        smiles.append(d.smiles)
        labels.append(d.y.view(-1))
    print(f"read {len(smiles)} PCBA molecules (dataset holds {n_raw})", flush=True)

    kept_idx, dropped = filter_pool_indices(smiles, manifests)
    print(f"kept {len(kept_idx)}, dropped {dropped} "
          f"({dropped / max(1, len(smiles)):.2%} held-out scaffolds or unparseable)", flush=True)

    # Per-manifest attribution, so the choice to filter against all four stays auditable.
    per_manifest = {}
    for path, m in zip([p for p in DOWNSTREAM_MANIFESTS if Path(p).exists()], manifests):
        banned = heldout_scaffolds(m)
        per_manifest[path] = len(banned)

    payload = {
        "smiles": [smiles[i] for i in kept_idx],
        "y": torch.stack([labels[i] for i in kept_idx]).to(torch.float16),
        "manifests": [p for p in DOWNSTREAM_MANIFESTS if Path(p).exists()],
        "n_banned_scaffolds_per_manifest": per_manifest,
        "n_input": len(smiles),
        "n_dropped": dropped,
        "limit": limit,
        "git": git_rev(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"cached filtered corpus -> {cache_path}", flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kw", default=None)
    ap.add_argument("--cache", default="data/pool/pcba_filtered.pt")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--init-from", default=None,
                    help="node-level checkpoint; presence selects the node+graph arm")
    ap.add_argument("--epochs", type=int, default=20,
                    help="matches the node-level run for comparability")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--n-tasks", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="truncate the corpus (smoke tests only; recorded in the metrics JSON)")
    ap.add_argument("--out", default="artifacts/encoder_graph.pt")
    args = ap.parse_args()

    kw = resolve_keyword(args.kw)
    run_tag = make_run_tag(kw)
    arm = "node_graph" if args.init_from else "graph"
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{run_tag}] arm={arm} device={device} epochs={args.epochs}", flush=True)

    manifests = load_manifests()
    if manifests is None:
        return 1

    cache_path = Path(args.cache)
    if args.rebuild_cache or not cache_path.exists():
        cache = build_cache(cache_path, args.root, manifests, limit=args.limit)
    else:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"loaded cached corpus: {len(cache['smiles'])} molecules from {cache_path}", flush=True)

    # Re-assert rule 2 on every run rather than trusting the cache file.
    _kept, dropped = filter_pool_indices(cache["smiles"], manifests)
    if dropped:
        print(f"REFUSING: corpus contained {dropped} molecules with held-out scaffolds. "
              f"Rebuild with --rebuild-cache before pretraining.")
        return 1
    print(f"leakage re-check passed: 0 held-out scaffolds "
          f"(checked vs {len(manifests)} manifests)", flush=True)

    y = cache["y"].to(torch.float32)
    graphs = []
    n_unparseable = 0
    total_smi = len(cache["smiles"])
    for i, (smi, row) in enumerate(zip(cache["smiles"], y)):
        g = mol_to_data(smi, y=row.tolist())
        if g is None:
            n_unparseable += 1
            continue
        graphs.append(g)
        if i and i % 50000 == 0:
            print(f"  featurizing {i}/{total_smi}", flush=True)
    print(f"featurized {len(graphs)}/{len(cache['smiles'])} molecules "
          f"({n_unparseable} unparseable)", flush=True)
    if not graphs:
        print("REFUSING: no molecules survived featurization")
        return 1

    from torch_geometric.loader import DataLoader

    from src.ssl.supervised_graph import build
    from src.train import load_pretrained_encoder

    encoder = GINEEncoder(hidden=args.hidden, layers=args.layers)
    if args.init_from:
        # Reuse the downstream loader so a mismatched node checkpoint cannot silently become the
        # node+graph arm -- it carries the architecture-match raise.
        encoder.load_state_dict(load_pretrained_encoder(
            args.init_from, {"hidden": args.hidden, "layers": args.layers}))
        print(f"initialized encoder from node-level checkpoint {args.init_from}", flush=True)
    encoder = encoder.to(device)

    model = build({"encoder": encoder, "n_tasks": args.n_tasks}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            opt.step()
            total += float(loss) * batch.num_graphs
            n += batch.num_graphs
        mean = total / max(1, n)
        history.append(mean)
        print(f"  epoch {epoch:3d} assay_bce {mean:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Only the encoder transfers; the assay head is discarded, as Hu et al. do. src/train.py loads
    # encoder_state and checks hidden/layers, so the extra keys here are pure provenance.
    torch.save({
        "encoder_state": encoder.state_dict(),
        "hidden": args.hidden,
        "layers": args.layers,
        "run_tag": run_tag,
        "pretrain_stage": "graph_supervised",
        "arm": arm,
        "init_from": args.init_from,
        "corpus": "PCBA",
        "n_tasks": args.n_tasks,
        "n_pool": len(graphs),
        "epochs": args.epochs,
        "final_loss": history[-1] if history else None,
        "git": git_rev(),
    }, args.out)

    metrics = {
        "run_tag": run_tag, "script": "run_pretrain_graph", "arm": arm,
        "corpus": "PCBA", "cache": str(cache_path),
        "n_corpus_input": cache["n_input"], "n_corpus_dropped_leakage": cache["n_dropped"],
        "n_banned_scaffolds_per_manifest": cache.get("n_banned_scaffolds_per_manifest"),
        "manifests": cache["manifests"],
        "n_featurized": len(graphs), "n_unparseable": n_unparseable,
        "n_tasks": args.n_tasks, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "seed": args.seed, "limit": args.limit,
        "init_from": args.init_from,
        "final_loss": history[-1] if history else None,
        "loss_history": history, "checkpoint": args.out, "git": git_rev(),
    }
    write_metrics(run_tag, metrics)
    append_ledger({
        "run_tag": run_tag, "script": "run_pretrain_graph", "dataset": "PCBA",
        "ssl_method": f"pretrain_{arm}", "n_seeds": 1,
        "n_train": len(graphs), "n_pool_dropped": cache["n_dropped"],
        "final_mask_loss": metrics["final_loss"], "git": git_rev(),
    })
    print(f"\nsaved encoder -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
