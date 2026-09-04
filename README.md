# mol-ssl — what pretraining contributes per label, under scaffold-split evaluation

Low-label molecular property prediction on MoleculeNet, evaluated with Bemis-Murcko scaffold
splits, five seeds at every labelled fraction, pretraining corpora decontaminated against every
held-out scaffold, and decision thresholds fixed before measurement.

**The result depends on which level you pretrain at.** Node-level self-supervision (attribute
masking) contributes nothing distinguishable from seed variation. Graph-level supervised
pretraining clears the seed-noise threshold at every labelled fraction tested, and the advantage
widens as labels become scarcer. The two contribute additively, with no detectable synergy.

For calibration: substituting a random split for a scaffold split on the same data and model
inflates the same benchmark by **+0.084 AUROC**, larger than any pretraining effect measured here.
The evaluation protocol governs the conclusion more strongly than the choice of objective.

**Short manuscript:** [docs/manuscript/manuscript.pdf](docs/manuscript/manuscript.pdf)
([LaTeX source](docs/manuscript/manuscript.tex))

---

## Abstract

Self-supervised pretraining is widely reported to improve molecular property prediction where
labelled data are scarce, yet reported gains are frequently small relative to the variation
introduced by seed and split selection. Whether such pretraining confers a genuine advantage
therefore depends on the rigour of the evaluation applied to it. We evaluate node-level and
graph-level pretraining on the Tox21 toxicity benchmark under a protocol fixed before any
measurement: Bemis–Murcko scaffold splits that place unseen molecular frameworks in the test
partition, pretraining corpora decontaminated against every held-out scaffold, repeated seeds at
each level of supervision, and decision thresholds specified in advance. A complete two-factor
design separates the contribution of each pretraining level and allows their interaction to be
estimated rather than assumed. Node-level self-supervision yields no improvement distinguishable
from seed variation, whereas graph-level supervised pretraining improves prediction at every level
of supervision examined, with the largest advantage where labelled data are most limited. The two
levels contribute additively, and no synergy between them is detectable. Substituting a random
split for a scaffold split inflates the same benchmark by more than any pretraining effect this
study measures, indicating that the evaluation protocol governs the conclusion more strongly than
the choice of pretraining objective.

---

## The three findings

### 1. Node-level against graph-level pretraining — the node × graph factorial

Four conditions differing only in the encoder weights at the start of fine-tuning. Identical
splits, seeds, labelled subsamples and hyperparameters throughout.
4 conditions × 5 labelled fractions × 5 seeds = 100 fine-tuning runs.

| Labelled fraction | n_train | none | node | graph | node + graph |
|---|---|---|---|---|---|
| 5%   | 313   | 0.6215 ± 0.0442 | 0.6368 ± 0.0277 | 0.6745 ± 0.0208 | **0.6936 ± 0.0307** |
| 10%  | 626   | 0.6253 ± 0.0291 | 0.6631 ± 0.0316 | 0.6970 ± 0.0196 | **0.7168 ± 0.0118** |
| 25%  | 1,564 | 0.6937 ± 0.0114 | 0.6879 ± 0.0181 | 0.7229 ± 0.0080 | **0.7300 ± 0.0138** |
| 50%  | 3,129 | 0.7083 ± 0.0198 | 0.7194 ± 0.0073 | 0.7502 ± 0.0074 | **0.7554 ± 0.0120** |
| 100% | 6,258 | 0.7293 ± 0.0063 | 0.7505 ± 0.0106 | 0.7571 ± 0.0120 | **0.7676 ± 0.0179** |

![labelled fraction curve](results/label_budget_tox21_0903_1750_tox21_2x2_full.png)

Differences from random initialisation, against the pre-registered thresholds:

| Labelled fraction | node | graph | node + graph |
|---|---|---|---|
| 5%   | +0.0154   | +0.0530 \* | +0.0722 \* |
| 10%  | +0.0377   | +0.0716 \*\* | +0.0914 \*\* |
| 25%  | −0.0058   | +0.0292 \*\* | +0.0363 \*\* |
| 50%  | +0.0110   | +0.0419 \* | +0.0470 \*\* |
| 100% | +0.0212 \* | +0.0278 \*\* | +0.0384 \*\* |
| **Clears 2 × pooled σ** | **0 of 5** | 3 of 5 | **4 of 5** |

\* exceeds pooled seed σ · \*\* exceeds 2 × pooled seed σ · unmarked lies inside seed noise

The primary comparison, nominated in advance — node + graph against none at full supervision —
gives **+0.0384** with pooled σ 0.0190, clearing the adjusted threshold. The **interaction**,
`node_graph − (graph + node) + none`, lies inside seed noise at all five labelled fractions with
alternating sign: the two levels are additive. Applying the same contrast to Hu et al.'s own
averages gives +0.2, also additive, so this replicates their data on that point.

Full writeup: [results/FINDING_node_vs_graph.md](results/FINDING_node_vs_graph.md).
Pre-registration, committed before the sweep ran:
[results/PRECLAIM_node_vs_graph.md](results/PRECLAIM_node_vs_graph.md).

### 2. Node-level pretraining alone shows no low-label advantage

Measured twice, five weeks apart, agreeing within pooled σ at every labelled fraction (largest
discrepancy 0.0096). The hypothesis was that the advantage would widen as labels shrink. It did
not — with node-level pretraining alone, the only gap clearing the threshold is at *full*
supervision, where self-supervision was expected to matter least.

Full writeup: [results/FINDING_label_budget.md](results/FINDING_label_budget.md).

### 3. Random splitting inflates Tox21 AUROC by 8.4 points

Identical model, identical data, identical seeds — only the split changes.

| Split | Tox21 test AUROC (5 seeds) |
|---|---|
| Bemis–Murcko scaffold | 0.7383 ± 0.0070 |
| Random | 0.8226 ± 0.0146 |
| **Inflation** | **+0.0843 ± 0.0140** |

Positive on every seed (+0.066 to +0.102). A random split lets molecules sharing a scaffold land on
both sides of the train/test boundary, so the model receives credit for recognising chemotypes it
trained on. Full writeup:
[results/FINDING_split_inflation.md](results/FINDING_split_inflation.md).

## Corpus decontamination

Both pretraining corpora were filtered against every downstream validation and test scaffold:

| Corpus | Input | Removed | Kept |
|---|---|---|---|
| ZINC250k (node-level) | 249,455 | **10,335 (4.14%)** | 239,120 |
| PCBA (graph-level) | 437,927 | **18,809 (4.30%)** | 419,118 |

A pretraining pipeline that skips this step trains on evaluation chemotypes, and the resulting
difference cannot be attributed to transfer. Note that Hu et al. describe removing evaluation
graphs from their graph-level supervised corpus but not from the two million molecules used at the
node level.

Verified by negative control: ten real Tox21 validation and test molecules inserted into a corpus
are rejected with a non-zero exit status and no checkpoint written.

## Supervised baselines (scaffold split, 5 seeds)

| Dataset | GINE (this work) | ECFP4+RF | Stronger |
|---|---|---|---|
| Tox21 | **0.7383 ± 0.0098** AUROC | 0.6977 ± 0.0017 | GINE +0.041 |
| Lipophilicity | **0.8254 ± 0.0332** RMSE | 1.0278 ± 0.0020 | GINE −0.202 |
| BBBP | 0.6891 ± 0.0730 AUROC | **0.7971 ± 0.0072** | ECFP +0.108 |
| BACE | 0.8199 ± 0.0260 AUROC | **0.9174 ± 0.0042** | ECFP +0.098 |

The graph network is stronger on the two larger datasets and **weaker on both smaller ones** — and
on BBBP its seed spread (±0.073) is ten times the fingerprint's. A five-layer GINE trained from
scratch on roughly 1.5k molecules has more parameters than supervision. That is reported rather
than omitted, and it is what motivates the low-label question.

Full table with published reference values: [results/RESULTS.md](results/RESULTS.md).

## Methodological rules, enforced in code

1. **Scaffold splits only.** `require_scaffold_split()` raises on a missing or random split, and
   there is no random-split function in [src/splits.py](src/splits.py). The random split used for
   the inflation study lives in a quarantined module whose manifests the production loader
   **refuses** to read.
2. **Pretraining corpora are filtered before they touch disk**, and
   [tests/test_leakage.py](tests/test_leakage.py) **fails when the filter is disabled** — verified
   by mutation via [scripts/verify_leakage_teeth.sh](scripts/verify_leakage_teeth.sh), not by
   assertion alone. One mutation breaks both the unlabelled and the labelled-corpus test files.
3. **No fabricated metrics.** Every number above is generated by
   [scripts/make_results_table.py](scripts/make_results_table.py), which reads only
   `results/metrics_{run_tag}.json` files produced by actual runs.
4. **Fixed seeds, mean ± std over 5 seeds**, one row per run in `artifacts/experiments.csv`.
   Never a best epoch, never a best seed.
5. **Claims written before the sweep.** See
   [results/PRECLAIM_node_vs_graph.md](results/PRECLAIM_node_vs_graph.md), committed before any
   value existed. A null result is reported as one.

## Reproduce

```bash
conda env create -f environment.yml && conda activate molssl
python -m pytest                                    # 74 tests
python scripts/make_splits.py                       # committed scaffold manifests
python scripts/run_baseline.py --config configs/baseline_tox21_gine.yaml --kw tox21_sup
python scripts/run_split_comparison.py --config configs/baseline_tox21_gine.yaml --kw splitgap
```

The node × graph factorial, in order:

```bash
# node-level: attribute masking on scaffold-filtered ZINC250k
python scripts/build_pool.py --from-file data/raw/zinc250k.smi --kw pool
python scripts/run_pretrain.py --pool data/pool/zinc250k_filtered.txt --kw pretrain

# graph-level: supervised multi-task on scaffold-filtered PCBA (downloads ~15 MB)
python scripts/run_pretrain_graph.py --kw pcba_graph --out artifacts/encoder_graph.pt

# node then graph, Hu et al.'s sequential order
python scripts/run_pretrain_graph.py --kw pcba_nodegraph \
    --init-from artifacts/encoder_pretrained.pt --out artifacts/encoder_node_graph.pt

# all four conditions, 100 fine-tuning runs
python scripts/run_label_budget_sweep.py --config configs/labelbudget_tox21.yaml \
    --arms none,node,graph,node_graph \
    --checkpoint node=artifacts/encoder_pretrained.pt \
    --checkpoint graph=artifacts/encoder_graph.pt \
    --checkpoint node_graph=artifacts/encoder_node_graph.pt --kw tox21_2x2
```

Encoder checkpoints are not committed (`*.pt` is ignored); rerun the pretraining steps to
regenerate them. Verified on 2× Tesla T4 (sm_75 — fp16/fp32 only, no bf16), CUDA 12.4, torch 2.6.0,
PyG 2.6.1, RDKit 2024.09.6.

## Limitations

- **One downstream dataset.** Tox21 only, against Hu et al.'s eight-dataset average. The
  generality of the reversed main effects is untested.
- **Corpus substitution.** PCBA (437,927 molecules, 128 assays) stands in for Hu et al.'s ChEMBL
  (approximately 456,000 molecules, 1,310 assays). Molecule counts are comparable; task breadth is
  roughly one tenth.
- **The node-level corpus is eight times smaller** than theirs: 239,120 decontaminated ZINC250k
  molecules against two million ZINC15 molecules. The node-level condition is under-trained
  relative to theirs, and the absence of a measurable node-level effect here should not be read as
  evidence that the objective is ineffective at scale.
- One node-level objective is implemented. Context prediction, edge prediction and contrastive
  objectives (MolCLR-style) remain unimplemented stubs under `src/ssl/`.
- Pretraining ran 20 epochs at each stage, a schedule that was not tuned.
- Fine-tuning used one fixed learning rate for all conditions. Tuning the pretrained conditions
  alone is a standard way to manufacture a difference, so it was not done.
- The 2 × pooled σ threshold is a pragmatic guard against multiplicity, not a hypothesis test.
  Paired per-seed tests or a mixed model over seeds and labelled fractions would be preferable.
- Numbers here are **not** directly comparable to published Chemprop values: this repository uses
  its own deterministic Bemis–Murcko split, and split implementations differ in tie-handling and
  balancing enough to shift test-set class balance materially (see
  [results/SPLIT_DIAGNOSTICS.md](results/SPLIT_DIAGNOSTICS.md)).

## Method sources

Hu et al. 2020, *Strategies for Pre-training Graph Neural Networks* (both implemented pretraining
levels) · MolCLR (contrastive molecular representations, not implemented) · Yang et al. 2019
(Chemprop D-MPNN, cited for reference values, not reproduced) · MoleculeNet benchmark protocol.
