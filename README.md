# Reinforcement Learning for Antibody CDRH3 Design


This project moves the design decision earlier in antibody discovery by making the axes that usually kill a candidate readable on the sequence itself: a CPU only pipeline that fine tunes IgLM with reinforcement learning  toward CDRH3 loops that are developable, human, and HER2 binding, scored by a panel of oracles spanning sequence, structure, and binding.

## The larger goal: moving the decision point earlier

Antibody discovery spends its money late. Candidates are carried deep into wet lab campaigns before their liabilities (poor developability, immunogenicity, weak binding) surface and kill them. Every late failure is months and reagents already spent.

The idea behind this project is a single thesis: move the go / no-go decision earlier in time. If we can read the axes that usually fail a candidate (chemical liabilities, humanness, physicochemistry, structural developability, target binding) computationally, on the sequence, before committing, then the choice to advance or drop a design happens sooner and on cheaper evidence. The panel of oracles here is a first concrete instrument for that idea: each oracle makes one liability axis legible at the design stage, and the RL loop shows how a generator can be steered by that combined readout instead of by a single objective.

The point is not that these in silico scores replace the assay. It is that the more of the failure surface you see up front, the earlier the decision point moves, and the less you spend learning what you could have known sooner.

## Built to extend

The reward is a panel of independent oracles combined at the end, so the framework grows by adding scorers rather than by rebuilding it. A new liability axis (a second target, an immunogenicity predictor, an expression or aggregation model) enters as one more oracle in the panel, and the same RL loop optimizes against the updated readout. Whether a given axis carries useful signal is then an empirical question the pipeline is set up to answer, as the multimodal test in notebook 08 shows with an honest null result for structure.

## What this is

A generative antibody model steered by RL: a single generator with an evaluator built from multiple modalities.

- **Generator (prior):** IgLM, frozen. Samples realistic CDRH3 loops on a fixed VH framework.
- **Agent:** a trainable copy of IgLM, updated by RL toward a reward.
- **Reward (the panel):** separately scored oracles combined at the end.
  - *sequence:* humanness (AbLang2), chemical liabilities (motif regex), physicochemistry (ProtParam)
  - *structure:* TAP style developability (ABodyBuilder2 fold, ANARCI, SASA patches)
  - *binding:* HER2 P(binder) from the Mason trastuzumab DMS (ESM2 8M plus logistic regression)
- **Update:** REINVENT augmented likelihood, log P*(x) = log P_prior(x) + sigma * S(x).

The scientific claims are about optimization dynamics (the loop optimizes an objective when it has headroom), not about experimentally validated binders.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_generator.ipynb` | IgLM generator; scaffold, CDRH3 extraction, temperature sweep, EDA |
| 02 | `02_oracles.ipynb` | Three sequence oracles (humanness / liability / physchem); orthogonality |
| 03 | `03_oracle_structure.ipynb` | TAP style structural developability oracle; cascade filter |
| 04 | `04_reward.ipynb` | Weighted sum reward S(x); weight sensitivity; augmented likelihood |
| 05 | `05_rl_loop.ipynb` | The REINVENT loop; baseline, sigma / batch / length guard ablations, positive control |
| 06 | `06_binding_oracle_her2.ipynb` | HER2 binding oracle (Mason DMS); oracle ablation; RL headline 0.30 to 0.80; XAI |
| 07 | `07_multiobjective_pareto.ipynb` | Multi objective Pareto front over all four objectives (the trade off) |
| 08 | `08_multimodal_reward.ipynb` | Learned multimodal reward (sequence plus structure fusion); honest null result |

## Repository layout

```
notebooks/   the 8 notebooks (01 to 08), with embedded outputs
src/         oracles.py, reward.py, binding_oracle.py (importable modules)
models/      trained classifier checkpoints (.joblib)
data/        cached scores, RL histories, Pareto data, embeddings (.json / .npy)
structures/  an example folded Fv (.pdb)
```

## Key results

- **The loop optimizes when there is headroom.** Developability (already saturated at ~0.76) stays flat; HER2 binding (14% binders at start) rises **0.30 to ~0.80** across 3 seeds, KL proxy to about -10.
- **Positive control** (tyrosine fraction proxy reward) rises 0.12 to 0.52 across 3 seeds, so the RL machinery provably optimizes.
- **Sequence and structure are orthogonal** (corr ~ 0). TAP reorders otherwise equal candidates but adds no binding signal over sequence (fusion AUC 0.75 vs sequence only 0.77), an honest negative.
- **Multi objective Pareto:** binding and developability trade off (corr 0.07); 6/120 loops are non dominated in 2D, 17/120 in 4D.
- **XAI (saturation mutagenesis):** on the WT trastuzumab loop, position 8 dominates (A to K drops P(binder) 0.97 to 0.22); across the 120 generated loops the signal is distributed with a mild C terminal bias.

## Environment and setup

CPU only; no GPU required. Python 3.11.

```bash
pip install -r requirements.txt
```

### AbLang2 weights (NOT in this repo)

The AbLang2 humanness model weights (~166 MB) exceed GitHub's file size limit and are not included. The `ablang2` package downloads them automatically on first use via `ablang2.pretrained(...)` (see notebook 02, which sets `oracles.ABLANG_MODEL`). If offline, use the package's own weight download utility rather than a hard coded URL, since the hosting location may change.

IgLM, ESM2 8M, and ImmuneBuilder (ABodyBuilder2 / NanoBodyBuilder2) weights are likewise downloaded by their packages on first use.

## Data provenance

- **Mason DMS (HER2):** trastuzumab CDRH3 deep mutational scan, ~36k unique 10-mers (github.com/dahjan/DMS_opt). Used to train the binding oracle.
- **IgLM:** pretrained antibody language model (the PyPI checkpoint is 12.9M parameters).
- The 120 CDRH3 loops scored throughout are **generated by IgLM**, not from Mason.

## Honest scope

Everything downstream of the binding oracle is **predicted** P(binder) under a **transferred proxy** (the oracle is trained on 10-mers of one framework; the generator makes 8 to 18 mers). No wet lab validation. The contribution is a reproducible **method** and a set of controlled optimization experiments, not a designed therapeutic.

## How to cite

If you use this code or build on it, please cite:

Gonzalez, Daniela A. (2026). *Reinforcement Learning for Antibody CDRH3 Design*.
Part of the prior-guided-antibody-design project. GitHub repository.
https://github.com/DaniAGonzalez/prior-guided-antibody-design
