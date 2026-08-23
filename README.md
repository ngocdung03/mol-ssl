# mol-ssl — semi-supervised molecular property prediction

Low-label molecular property prediction on scaffold-split MoleculeNet: how much does
semi-supervised learning actually buy you *per label*, measured with splits and error bars that
survive scrutiny.

**Status: scaffold + verified environment. No experimental results yet.** Every number in this
README will trace to a `results/metrics_{run_tag}.json`. There are none, so there are no numbers.
See [PLAN.md](PLAN.md) for phases and gates.

## The question

Assays are small; unlabeled molecules are effectively free. So the useful question is not "what is
the best AUROC on Tox21" but **"how does the accuracy-per-label curve bend when you add
self-supervision?"** — and whether the answer holds up under a Bemis–Murcko scaffold split, which
is the honest stand-in for "a chemotype the model has never seen".

Headline deliverable: a **label-budget curve** (metric vs. fraction of labels kept, one line per
method, ±1 std over ≥5 seeds), plus calibration and selective-prediction results.

## Non-negotiables

1. Scaffold splits only. `require_scaffold_split()` errors on a missing or random split; there is
   no random-split function in [src/splits.py](src/splits.py).
2. The unlabeled pool is filtered against every downstream val/test scaffold before it touches
   disk — and [tests/test_leakage.py](tests/test_leakage.py) **fails when the filter is disabled**
   (verified by mutation, not by assertion theatre).
3. No fabricated metrics. Real run, or say it didn't run.
4. Fixed seeds, mean ± std over ≥5 seeds, one row per run in `artifacts/experiments.csv`.
5. Claims written before the sweep. A null result — SSL gains inside seed noise — gets reported as
   a null result.

## Setup

```bash
conda env create -f environment.yml     # env `molssl`; torch 2.6.0+cu124, PyG 2.6.1, RDKit 2024.9.6
conda activate molssl
python -m pytest                        # 14 tests
python scripts/smoke.py                 # SMILES -> graph -> GINE forward pass on GPU
python scripts/overfit100.py --kw overfit_check   # Phase-0 gate: overfit 100 molecules
```

Verified on 2× Tesla T4 (sm_75 — fp16/fp32 only, no bf16), CUDA 12.4.

## Layout

| Path | What |
|---|---|
| `src/featurize.py` | RDKit SMILES → PyG graph tensors (explicit atom/bond feature scheme) |
| `src/splits.py` | Bemis–Murcko scaffolds, deterministic scaffold split, committed split manifests |
| `src/datamodule.py` | Split enforcement, label-budget subsampling, unlabeled-pool scaffold filter |
| `src/runlog.py` | Mandatory `--kw` run tags, per-run metrics JSON, append-only ledger |
| `src/models/gnn.py` | GINE encoder + prediction head |
| `src/ssl/` | Pretrain / contrastive / consistency / pseudo-label — one interface (**M3, stubs**) |
| `configs/` | One YAML per experiment; ablations are config flips |
| `data/splits/` | Cached scaffold splits, committed — single source of truth |
| `tests/` | Featurization, split integrity, leakage regression |

## Baselines

ECFP4-2048 + RandomForest, ECFP4 + XGBoost, and an own PyG GINE, all on scaffold splits.
Chemprop D-MPNN numbers are **cited from publication, not reproduced here** — labeled as such in
every table.

## Method sources

Hu et al. 2020 (pretraining strategies for GNNs) · MolCLR (contrastive molecular representations) ·
Gao et al. 2022 (sample efficiency, for the optional generation loop) · MoleculeNet / Chemprop
(Yang et al. 2019) for the benchmark protocol.

## Limitations (kept current, deliberately)

- SSL gains on MoleculeNet are frequently within seed noise once splits and seeds are handled
  honestly. This repo is built to measure that faithfully rather than to win a leaderboard.
- Unlabeled pool is capped at ~250k molecules by the T4 budget, well below the ChEMBL-scale
  pretraining in the literature.
- Small datasets (BBBP 2039, BACE 1513) cannot support a 1% label budget — 20 molecules is noise,
  so their sweep starts at 10%.
- `src/ssl/` is unimplemented (M3). Nothing in this repo yet demonstrates an SSL result.
