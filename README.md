# Prior-Guided RL for Antibody CDRH3 Design

*Code base for an ongoing personal research project on **moving the design decision earlier**.*

A CPU-only, reproducible pipeline that fine-tunes an antibody language model (IgLM) with
reinforcement learning (REINVENT-style) to design CDRH3 loops that are **developable**,
**human**, and **HER2-binding**, evaluated by a multi-oracle judge spanning sequence,
structure, and binding.

## The larger goal — shifting the decision line left

Antibody discovery spends its money late: candidates are carried deep into wet-lab
campaigns before liabilities — poor developability, immunogenicity, weak binding — surface
and kill them. Every such late failure is months and reagents already spent.

This project is the code base of a larger personal effort with one thesis: **move the
go / no-go decision to the left in time.** If we can read out the axes that usually fail a
candidate — chemical liabilities, humanness, physicochemistry, structural developability,
target binding — *computationally, on sequence, before committing*, then the decision to
advance or drop a design happens earlier, on cheaper evidence. The multi-oracle judge here
is a first concrete instrument for that: each oracle is one liability axis made legible at
the design stage, and the RL loop shows how a generator can be steered by that combined
readout instead of by a single objective.

The point is not that these in-silico scores replace the assay. It is that the more of the
failure surface you can see up front, the further left the decision line moves — and the
less you spend learning things you could have known earlier.

## What this is

A prior-guided RL system for antibody loop design:

- **Generator (prior):** IgLM, frozen. Samples realistic CDRH3 loops on a fixed VH framework.
- **Agent:** a trainable copy of IgLM, updated by RL toward a reward.
- **Reward (the judge):** a late-fusion of separately-scored oracles —
  - *sequence:* humanness (AbLang2), chemical liabilities (motif regex), physicochemistry (ProtParam)
  - *structure:* TAP-style developability (ABodyBuilder2 fold -> ANARCI -> SASA patches)
  - *binding:* HER2 P(binder) from the Mason trastuzumab DMS (ESM2-8M + logistic regression)
- **Update:** REINVENT augmented likelihood  log P*(x) = log P_prior(x) + sigma * S(x).

It is a **generative model guided by RL** — a unimodal generator with a **multimodal evaluator**.
The scientific claims are about **optimization dynamics** (the loop optimizes an objective when it
has headroom), not about experimentally validated binders.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_generator.ipynb` | IgLM generator; scaffold, CDRH3 extraction, temperature sweep, EDA |
| 02 | `02_oracles.ipynb` | Three sequence oracles (humanness / liability / physchem); orthogonality |
| 03 | `03_oracle_structure.ipynb` | TAP-style structural developability oracle; cascade filter |
| 04 | `04_reward.ipynb` | Weighted-sum reward S(x); weight sensitivity; augmented likelihood |
| 05 | `05_rl_loop.ipynb` | The REINVENT loop; baseline, sigma / batch / length-guard ablations, positive control |
| 06 | `06_binding_oracle_her2.ipynb` | HER2 binding oracle (Mason DMS); oracle ablation; RL headline 0.30->0.80; XAI |
| 07 | `07_multiobjective_pareto.ipynb` | Multi-objective Pareto front over all four objectives (the trade-off) |
| 08 | `08_multimodal_reward.ipynb` | Learned multimodal reward (sequence + structure fusion); honest null result |

## Repository layout

```
notebooks/   the 8 notebooks (01-08), with embedded outputs
src/         oracles.py, reward.py, binding_oracle.py (importable modules)
models/      trained classifier checkpoints (.joblib)
data/        cached scores, RL histories, Pareto data, embeddings (.json / .npy)
figures/     all figures used across notebooks and slides
slides/      presentation decks (Palatino, editable) + drawio schematic
docs/        this project README, results notes, arXiv manuscript skeleton, schematics PDF
html/        static HTML exports of notebooks 01-06
structures/  an example folded Fv (.pdb)
```

## Key results

- **The loop optimizes when there is headroom.** Developability (already saturated at ~0.76) stays
  flat; HER2 binding (14% binders at start) rises **0.30 -> ~0.80** across 3 seeds, KL proxy -> -10.
- **Positive control** (tyrosine-fraction proxy reward) rises 0.12 -> 0.52 across 3 seeds — the RL
  machinery provably optimizes.
- **Sequence and structure are orthogonal** (corr ~ 0). TAP reorders otherwise-equal candidates but
  adds no binding signal over sequence (fusion AUC 0.75 vs sequence-only 0.77) — an honest negative.
- **Multi-objective Pareto:** binding and developability trade off (corr 0.07); 6/120 loops are
  non-dominated in 2D, 17/120 in 4D.
- **XAI (saturation mutagenesis):** on the WT trastuzumab loop, position 8 dominates (A->K drops
  P(binder) 0.97 -> 0.22); across the 120 generated loops the signal is distributed with a mild
  C-terminal bias.

## Environment & setup

CPU-only; no GPU required. Python 3.11.

```bash
pip install -r requirements.txt
```

### AbLang2 weights (NOT in this repo)

The AbLang2 humanness model weights (~166 MB) exceed GitHub's file-size limit and are not
included. The `ablang2` package downloads them automatically on first use via `ablang2.pretrained(...)`
(see notebook 02, which sets `oracles.ABLANG_MODEL`). If offline, use the package's own
weight-download utility rather than a hard-coded URL, since the hosting location may change.

IgLM, ESM2-8M, and ImmuneBuilder (ABodyBuilder2 / NanoBodyBuilder2) weights are likewise
downloaded by their packages on first use.

## Data provenance

- **Mason DMS (HER2):** trastuzumab CDRH3 deep mutational scan, ~36k unique 10-mers
  (github.com/dahjan/DMS_opt). Used to train the binding oracle.
- **IgLM:** pretrained antibody language model (the PyPI checkpoint is 12.9M parameters).
- The 120 CDRH3 loops scored throughout are **generated by IgLM**, not from Mason.

## Honest scope

Everything downstream of the binding oracle is **predicted** P(binder) under a **transferred proxy**
(the oracle is trained on 10-mers of one framework; the generator makes 8-18mers). No wet-lab
validation. The contribution is a reproducible **method** and a set of controlled optimization
experiments, not a designed therapeutic.
