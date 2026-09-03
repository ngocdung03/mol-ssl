#!/usr/bin/env python
"""The headline experiment: metric vs label budget, supervised against pretrained.

For each budget, train both arms on the SAME labeled subset (same seed -> same subsample), so the
only difference between them is whether the encoder was pretrained. Anything else would confound
the comparison with a data difference.

The claim this is designed to support is "here is what pretraining buys you per label, measured
properly" -- not "pretraining wins". If the gap sits inside the seed noise, that is the result.

Arms are named conditions on how the encoder is initialized; everything else (split, seeds, labeled
subsample, hyperparameters) is held identical. For the node-vs-graph 2x2 replicating Hu et al. 2019
Table 1:
  none       random init
  node       node-level attribute masking (Hu's AttrMasking)
  graph      graph-level supervised multi-task on PCBA (Hu's Supervised)
  node_graph node-level then graph-level, Hu's sequential order (Supervised + AttrMasking)
`pretrain` is kept as a legacy alias for `node` so the committed 0825_1756_tox21_sweep command
still runs verbatim.

Usage:
  python scripts/run_label_budget_sweep.py --config configs/labelbudget_tox21.yaml \
      --checkpoint artifacts/encoder_pretrained.pt --kw tox21_sweep
  python scripts/run_label_budget_sweep.py --config configs/labelbudget_tox21.yaml \
      --arms none,graph,node_graph \
      --checkpoint graph=artifacts/encoder_graph.pt \
      --checkpoint node_graph=artifacts/encoder_node_graph.pt --kw tox21_2x2
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import budget_list, load_config
from src.eval import aggregate_over_seeds
from src.runlog import append_ledger, git_rev, make_run_tag, resolve_keyword, write_metrics
from src.train import train_one_seed


def parse_checkpoint_map(entries: list[str]) -> dict[str, str]:
    """['graph=a.pt', 'b.pt'] -> {'graph': 'a.pt', 'pretrain': 'b.pt'}.

    A bare path is assigned to the 'pretrain' arm, so the original
    `--checkpoint artifacts/encoder_pretrained.pt` invocation that produced the committed
    label-budget result still works unchanged.
    """
    ckpts: dict[str, str] = {}
    for e in entries:
        arm, sep, path = e.partition("=")
        if not sep:
            arm, path = "pretrain", e
        if arm in ckpts:
            raise SystemExit(f"--checkpoint given twice for arm {arm!r}")
        if not path:
            raise SystemExit(f"--checkpoint {e!r} has an empty path")
        ckpts[arm] = path
    return ckpts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", action="append", default=[], metavar="[ARM=]PATH",
                    help="encoder checkpoint. Repeatable as ARM=PATH "
                         "(e.g. graph=artifacts/encoder_graph.pt). A bare PATH is assigned to the "
                         "'pretrain' arm for backward compatibility.")
    ap.add_argument("--kw", default=None)
    ap.add_argument("--arms", default="none,pretrain",
                    help="comma-separated arm names; 'none' means random init")
    args = ap.parse_args()

    kw = resolve_keyword(args.kw)
    run_tag = make_run_tag(kw)
    cfg = load_config(args.config)
    task = cfg["task"]
    key = "auroc" if task == "classification" else "rmse"
    budgets = budget_list(cfg)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ckpt_map = parse_checkpoint_map(args.checkpoint)

    # Fail before any GPU time rather than partway through a multi-hour sweep.
    for arm in arms:
        if arm == "none":
            continue
        if arm not in ckpt_map:
            raise SystemExit(f"arm {arm!r} has no checkpoint; pass --checkpoint {arm}=<path>")
        if not Path(ckpt_map[arm]).exists():
            raise SystemExit(f"checkpoint for arm {arm!r} not found: {ckpt_map[arm]}")

    print(f"[{run_tag}] {cfg['dataset']} sweep: budgets={budgets} arms={arms} seeds={cfg['seeds']}")
    for arm in arms:
        print(f"  arm {arm:11s} <- {ckpt_map.get(arm) or 'random init'}")

    curve = {}
    for arm in arms:
        curve[arm] = {}
        for budget in budgets:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["label_budget"] = budget
            run_cfg.setdefault("ssl", {})
            # Same seed -> same labeled subsample in every arm; only the encoder init differs.
            run_cfg["ssl"]["checkpoint"] = ckpt_map.get(arm)  # None for 'none' -> random init

            scores, n_train = [], None
            for seed in cfg["seeds"]:
                r = train_one_seed(run_cfg, seed=seed, verbose=False)
                scores.append(r["test"].get(key))
                n_train = r["data"]["n_train"]
            agg = aggregate_over_seeds(scores)
            curve[arm][str(budget)] = {**agg, "n_train": n_train}
            print(f"  {arm:9s} budget {budget:>5} n_train={n_train:5d} "
                  f"{key} {agg['mean']:.4f} +/- {agg['std']:.4f}", flush=True)

    # The comparison is only meaningful with the seed spread alongside it, so carry both.
    # Threshold rule is pre-registered in results/PRECLAIM_node_vs_graph.md: the unadjusted
    # |delta| < pooled_std verdict is kept for continuity with the earlier label-budget run, and a
    # 2x pooled_std verdict is reported alongside because 3-4 arms x 5 budgets is 15-20
    # comparisons and the unadjusted rule has no multiplicity control.
    deltas = {}
    if "none" in curve:
        for arm in arms:
            if arm == "none":
                continue
            deltas[arm] = {}
            for b in curve["none"]:
                base, other = curve["none"][b], curve[arm][b]
                gap = other["mean"] - base["mean"]
                pooled = (base["std"] ** 2 + other["std"] ** 2) ** 0.5
                deltas[arm][b] = {
                    "delta": gap,
                    "pooled_std": pooled,
                    "inside_seed_noise": abs(gap) < pooled,
                    "exceeds_widened_threshold": abs(gap) > 2 * pooled,
                }
                verdict = "inside seed noise" if abs(gap) < pooled else "exceeds seed noise"
                wide = "  [clears 2x pooled]" if abs(gap) > 2 * pooled else ""
                print(f"  delta {arm} vs none @ {b}: {gap:+.4f} "
                      f"(pooled std {pooled:.4f}) -- {verdict}{wide}")

    # The 2x2 interaction: does node-level pay off differently once graph-level is present?
    #   (node_graph - graph) - (node - none)  ==  node_graph - (graph + node) + none
    # ~0 means the two objectives are additive, >0 synergistic, <0 redundant.
    #
    # This contrast is NOT from Hu et al. -- the 2x2 layout is theirs, but they never compute an
    # interaction; their argument is comparative. It is a stricter test than they assert. Their own
    # Table 1 averages give 73.5 - (68.9 + 71.4) + 67.0 = +0.2, essentially additive, so a
    # near-zero interaction here replicates Hu rather than contradicting him.
    interaction = {}
    node_arm = "node" if "node" in curve else ("pretrain" if "pretrain" in curve else None)
    if node_arm and all(a in curve for a in ("none", "graph", "node_graph")):
        for b in curve["none"]:
            cells = {a: curve[a][b] for a in ("none", node_arm, "graph", "node_graph")}
            value = ((cells["node_graph"]["mean"] - cells["graph"]["mean"])
                     - (cells[node_arm]["mean"] - cells["none"]["mean"]))
            pooled = sum(c["std"] ** 2 for c in cells.values()) ** 0.5
            interaction[b] = {
                "interaction": value,
                "pooled_std": pooled,
                "inside_seed_noise": abs(value) < pooled,
                "exceeds_widened_threshold": abs(value) > 2 * pooled,
                "node_arm_used": node_arm,
            }
            verdict = "inside seed noise" if abs(value) < pooled else "exceeds seed noise"
            print(f"  interaction @ {b}: {value:+.4f} (pooled std {pooled:.4f}) -- {verdict}")

    metrics = {
        "run_tag": run_tag, "script": "run_label_budget_sweep",
        "dataset": cfg["dataset"], "task": task, "metric": key,
        "config": cfg["config_path"], "checkpoints": ckpt_map,
        "budgets": budgets, "arms": arms, "seeds": cfg["seeds"],
        "curve": curve, "deltas": deltas, "interaction": interaction, "git": git_rev(),
    }
    write_metrics(run_tag, metrics)
    for arm in arms:
        for b, agg in curve[arm].items():
            append_ledger({
                "run_tag": f"{run_tag}_{arm}_{b}", "script": "run_label_budget_sweep",
                "dataset": cfg["dataset"], "task": task, "split": "scaffold",
                "label_budget": b, "ssl_method": arm, "n_seeds": len(cfg["seeds"]),
                f"test_{key}_mean": agg["mean"], f"test_{key}_std": agg["std"],
                "n_train": agg["n_train"], "git": git_rev(),
            })
    print(f"\nwrote results/metrics_{run_tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
