# mol-ssl — Semi-Supervised GNN for Molecular Property Prediction

**Revision of** `molecular_ssl_project_plan.md` (source doc), amended 2026-08-23 after checking it
against this machine and against `../../PLAN.md` (the AIGEN campaign plan).

---

## Context — why this project, and what changed from the source plan

The source doc is the entry-slice of "Direction #2" (generative molecular AI): a
semi-supervised / low-label molecular property predictor, i.e. the Test/Analyze oracle of a DMTA
loop. Target: AIGEN Sciences **role ① Drug Development AI Engineer**.

Five amendments were needed. Each is a correction of fact, not taste.

**1. Positioning conflict — resolved by user decision, recorded honestly.**
`../../PLAN.md` (approved 2026-08-20) targets **role ② LLM & Clinical AI Engineer** and lists
under *Explicitly out of scope*: "No RDKit / DeepChem / molecular work. That serves role ① and
dilutes the story." `../../research/role-fit.md` rates role ① **"Fit: weak"** — no RDKit, no PyG,
no MD/docking — and notes it is the referrer's own territory (PepTri, EquiCPI).
`../../outreach/ngoc-quang.md` is still **DRAFT — not sent**, so nothing has reversed that
analysis. On 2026-08-23 the user chose to make this the **primary** side project anyway. That
decision stands; the risk is stated once here and not re-litigated.
*Mitigation that follows from it:* do not compete on generative chemistry, where the referrer and
DMIS chemistry PhDs are ahead. Compete on **evaluation integrity in the low-label regime** —
scaffold-split honesty, leakage tests with teeth, seeded error bars, calibration. That is the
candidate's actual edge (see #3) and it is un-crowded.

**2. The source doc mislabels itself.** Its heading reads
"PART 1 — Foundation Knowledge Roadmap (Direction #1 — this project)". This project is
**Direction #2**. Direction #1 (KG / causal target discovery) does not exist on disk.

**3. The "transferred from Direction #1" effort column is false — but the real transfer is
stronger.** Direction #1 was never built, and the existing stack (`../../../Care/`) is nnU-Net
CNNs, transformer survival models, and radiomics/GBM ensembles. **No PyG, no message passing.**
So Phase 0 is genuine new learning, not review. What *does* transfer, wholesale, from `Care/`:

| `Care/` asset | Reused here as |
|---|---|
| Unseen-Center-C generalization framing | Scaffold split = "unseen chemotype", same problem restated |
| Leak-free seeded splits as single source of truth (`artifacts/splits_final.json`) | `data/splits/*.json`, committed |
| Leakage regression test (`NNUNet/tests/test_lifs_leakage.py`) | `tests/test_leakage.py`, must fail on a contaminated pool |
| Mandatory `--kw` run tagging (`runlog.resolve_keyword()`) | `src/runlog.py`, same contract |
| 231-run append-only ledger (`artifacts/experiments.csv`) | `artifacts/experiments.csv` |
| Pseudo-mask quality gate (Dice ≥ 0.85) | Pseudo-label confidence gate |
| Seed-ensemble + calibration findings | Deep ensemble, ECE, selective prediction |
| Documented LB collapse 0.9206 → 0.2632 and its root causes | The reason every gate below exists |

**4. Chemprop reproduction demoted.** Reproducing D-MPNN scaffold-split numbers means chemprop
v1↔v2 API churn plus its own torch pin against this box's torch 2.6.0+cu124 — a multi-day
dependency fight for a table row. **Decision:** run ECFP+RF/XGBoost and an own PyG GIN/GINE
baseline; compare against Chemprop's *published* numbers in the results table, cited, and label
them as published rather than reproduced. Reproduce for real only if M1 finishes early.

**5. Compute reality.** 2× **Tesla T4** (15 GB, sm_75 → **no bf16**, no flash-attention), CUDA 12.4,
torch 2.6.0. Small molecular GNNs are a comfortable fit; full-ChEMBL pretraining is not.
Unlabeled pool is capped (§Phase 3). fp16 or fp32 only.

Two further scope corrections that follow from the compute and statistics, not from the docs:

- **1% label budgets are dropped for the small datasets.** 1% of BBBP (2039) is 20 molecules —
  the measurement is seed noise, not a result. Budgets are dataset-dependent (§Phase 3).
- **W&B dropped as a hard dependency.** `Care/` already proves the run-tag + per-run metrics JSON +
  append-only CSV ledger pattern, offline and greppable. Use that. W&B optional, off by default.

**Intended outcome:** a public repo whose headline figure is a **label-budget advantage curve with
honest error bars**, plus a calibration / selective-prediction result, plus a leakage test that
demonstrably fails when fed a contaminated pool — and a README that states plainly if the SSL
advantage did not materialize.

---

## Hard rules (the defensibility guardrails — also in CLAUDE.md)

1. **Always scaffold-split (Bemis–Murcko). Never random-split.** If a config omits the split, the
   code errors out; it does not default.
2. **No test-set leakage into pretraining or pseudo-labeling.** The unlabeled pool must contain no
   scaffold present in any downstream val or test split. Enforced by a test that fails on a
   deliberately contaminated pool.
3. **Never fabricate or estimate a metric.** Every number in README/REPORT traces to a
   `results/metrics_{run_tag}.json` produced by an actual run. If a run did not happen, say so.
4. **Every result traces to a config + seed.** No magic numbers, no best-epoch cherry-picking.
   Report mean ± std over ≥5 seeds; report the seed list.
5. **Write the claims before the sweep, and do not soften them afterwards.** A null result is a
   deliverable.

---

## Phases

### Phase 0 — PyG foundation (3–5 days, NEW learning, not review)

Message passing from scratch → `MessagePassing`, `GINConv`/`GINEConv`, mini-batch collation of
variable-size graphs (`Batch`, `follow_batch`), pooling. Deliverable: a GINE that overfits 100
molecules to ~zero loss. That single check catches most featurization and batching bugs.

### Phase 1 — Cheminformatics primer (3–5 days, the new-skill tax)

SMILES canonicalization and the many-valid-SMILES fact (needed for enumeration augmentation);
atom/bond featurization; ECFP/Morgan fingerprints; Bemis–Murcko scaffold extraction.
Deliverable: `src/featurize.py` + `src/splits.py`, both tested.

### Phase 2 — Supervised baseline done right (1–1.5 wk)

- **Datasets (4, not 6).** Primary: **Tox21** (7831 mols, 12 tasks, AUROC) and **Lipophilicity**
  (4200, RMSE) — big enough to sweep label budgets. Small-data check: **BBBP** (2039) and
  **BACE** (1513), both ADMET-type. ESOL/FreeSolv optional, they are small and noisy.
  Loader: `torch_geometric.datasets.MoleculeNet` (avoids a `deepchem` dependency).
- **Baselines:** ECFP4-2048 + RandomForest, ECFP4 + XGBoost, own GINE (supervised, from scratch).
- **Validation gate (M1):** scaffold-split numbers within a reasonable margin of published
  MoleculeNet/Chemprop values, or the gap explained in writing. **A wrong baseline invalidates
  every SSL claim downstream — debug before proceeding.**

### Phase 3 — The semi-supervised layer (the differentiator) (2–3 wk)

**Unlabeled pool.** ZINC250k, or a ≤1M-molecule ChEMBL subset — capped for T4 budget, and
scaffold-filtered against the union of all downstream val+test scaffolds before it is ever written
to disk. The filter is a pipeline step with a test, not a comment.

**Methods, one swappable interface under `src/ssl/` so ablations are config flips, not code forks:**
- *Pretrain* — Hu et al. 2020 attribute masking + context prediction.
- *Contrastive* — MolCLR-style NT-Xent over atom masking / bond deletion / subgraph removal.
- *Consistency* — prediction stability under SMILES enumeration and graph augmentation.
- *Pseudo-label* — self-training with a confidence gate (the `Care/` Dice-gate pattern).

**Core experiment — label-budget sweep.** Budgets × methods × ≥5 seeds:
- Tox21, Lipophilicity: {5, 10, 25, 50, 100} %
- BBBP, BACE: {10, 25, 50, 100} % (1% and 5% are ~20–100 molecules → noise)

Headline figure: metric vs label budget, one line per method, shaded ±1 std over seeds.

**Validation gate (M4):** the SSL advantage should widen as labels shrink. If it does not, that is
the finding — report it. **Known risk, stated up front:** published SSL-on-MoleculeNet gains are
frequently within seed noise once splits and seeds are handled honestly. The seeded error bars are
what make either outcome publishable-quality; the claim is *"here is what SSL buys you per label,
measured properly"*, never *"SSL wins"*.

**Calibration / uncertainty.** 5-seed deep ensemble + MC-dropout; report ECE, reliability diagram,
and a selective-prediction (coverage vs accuracy) curve. Cheap, and it is the drug-discovery-real
part most benchmark repos skip.

### Phase 4 — OPTIONAL generation loop (2–4 wk, boxed)

Only after Phase 3 is written up. generate → predict (Phase-3 model) → select, benchmarked for
**sample efficiency under a fixed oracle budget** on TDC PMO (Gao et al. 2022). Docking oracle
optional. **Framing rule: claim sample efficiency of the loop, never "my agent found good
molecules"** — the latter invites the oracle-hacking critique and cannot be defended.

---

## Milestones and gates

| M | Content | Gate |
|---|---|---|
| M1 | Featurization, scaffold splits, supervised baselines | Baselines match published within margin, on scaffold splits |
| M2 | Unlabeled pool, label-budget subsampling | `tests/test_leakage.py` **fails** on a contaminated pool (prove teeth) |
| M3 | 4 SSL methods behind one interface | Each runs end-to-end from a config; ablation = config flip |
| M4 | Label-budget sweep + calibration | Curve produced with ≥5 seeds; result reported as-is |
| M5 | (Optional) generation loop | Sample efficiency vs random/greedy baseline |
| M6 | Portfolio packaging | Every README number traces to a `metrics_{run_tag}.json` |

## Effort

| Phase | Effort |
|---|---|
| 0 PyG foundation | 3–5 d |
| 1 Cheminformatics primer | 3–5 d |
| 2 Supervised baseline | 1–1.5 wk |
| 3 SSL + calibration | 2–3 wk |
| **Core (0–3)** | **~5–7 focused weeks** |
| 4 Generation (optional) | +2–4 wk |

Part-time calendar ≈ 2×. Note against `../../PLAN.md`'s ~6-week window: the core alone consumes it.
Phase 4 is not in the budget.

## Agentic-coding failure modes to guard (from the source doc, kept verbatim in intent)

- **Default is random split.** Agents reach for `train_test_split`. CLAUDE.md rule + a hard assert
  in `src/datamodule.py`.
- **Silent leakage in SSL.** Pretraining on a pool containing test molecules is the classic
  invisible bug. Encode exclusion as a failing test.
- **Optimistic reporting.** Agents print best epoch/seed. Fixed seeds, mean ± std, ledger.
- **Notebook drift.** `notebooks/` is exploratory; source of truth is `src/` + `configs/`.
- **Scope creep toward a full generator.** M5 stays boxed.

## Verification

1. `pytest` passes from repo root.
2. `tests/test_leakage.py` fails when fed a deliberately contaminated pool (teeth, not assertion theatre).
3. `python scripts/smoke.py` runs one molecule → graph → GINE forward pass on GPU.
4. A GINE overfits 100 molecules to near-zero loss (Phase 0 deliverable).
5. `artifacts/experiments.csv` has one row per run, no missing run tags.
6. Every number in `README.md` traces to a `results/metrics_{run_tag}.json`.

## Open items

- Which AIGEN role has live headcount — `../../outreach/ngoc-quang.md` still unsent.
- Unlabeled pool source: ZINC250k vs ChEMBL subset (decide at M2, on download size).
- Whether Chemprop gets reproduced for real (only if M1 lands early).
- Public repo name.
