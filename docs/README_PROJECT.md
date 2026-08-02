# Antibody CDRH3 Design with Prior-Guided RL and a Multimodal Evaluator

A research-grade, reproducible pipeline for optimizing antibody CDRH3 loops with reinforcement
learning, following AB-Gen (Xu et al. 2023) and REINVENT (Olivecrona et al. 2017). Built entirely
on CPU. Every notebook is self-contained, executed end-to-end with embedded outputs, and written
to teach: background from zero, design decisions with the rejected alternative, ablation studies,
diagnostic curves, and honest limitations.

## Reading order

| # | Notebook | What it builds | Headline result |
|---|----------|----------------|-----------------|
| 01 | `01_generator.ipynb` | IgLM CDRH3 generator + prior | IgLM loops more human than random (0.66 vs 0.47); temperature sweep |
| 02 | `02_oracles.ipynb` | 3 sequence oracles (liability, physchem, humanness) | oracles are orthogonal (max corr 0.14) |
| 03 | `03_oracle_structure.ipynb` | TAP-style structural oracle (both Fv and VH-only) | sequence vs structure orthogonal (corr -0.04); TAP as final filter |
| 04 | `04_reward.ipynb` | weighted reward S(x) + REINVENT augmented likelihood | weighting reorders top-20 (overlap 0.75) |
| 05 | `05_rl_loop.ipynb` | the RL loop + 4 ablations x 3 seeds | positive control 0.12->0.52 proves the loop optimizes |
| 06 | `06_binding_oracle_her2.ipynb` | HER2 binding oracle (Mason DMS) + RL + XAI | **binding 0.30->0.80, 3 seeds** (the main result) |
| 07 | `07_multimodal_reward.ipynb` | learned sequence+structure fusion head | honest ablation: structure adds no binding signal over sequence |

## The scientific story

1. **The RL loop works.** Notebook 05's positive control (tyrosine proxy) rises 0.12->0.52 across
   3 seeds - the policy-gradient machinery optimizes correctly.
2. **Developability is saturated.** The IgLM prior already produces developable loops (S(x)~0.78),
   so the developability reward is flat - not a bug, an oracle-headroom result.
3. **Binding has headroom, and the loop exploits it.** Notebook 06: the SAME loop raises predicted
   HER2 binding from 0.30 to 0.80 (3 seeds, KL -10). This contrast is the core finding.
4. **The evaluator is multimodal, honestly stated.** Sequence (oracles + ESM), structure (TAP), and
   binding (Mason) are scored separately and fused at the reward level. Notebook 07 tests learned
   early fusion and finds - honestly - that structure adds no binding signal over sequence, because
   the two modalities measure different biology.

## Key numbers to quote

- Generator: 100 loops in ~10s on CPU (IgLM ~13M params, not the 558M paper model)
- Oracles orthogonal: max pairwise correlation 0.14
- HER2 binding classifier: AUC 0.911 (reproduces Mason et al.)
- RL headline: binding 0.30 -> peak 0.84 -> last-5 0.80 (3 seeds), KL -10
- XAI: CDRH3 position 8 dominates binding (sensitivity 0.16); WT residues near-optimal
- Multimodal ablation: sequence AUC 0.77, structure 0.53 (near chance), fusion 0.75 (no gain)

## Honest scope

Not publishable as-is without more work. The binding oracle is a transferred proxy (Mason length-10
trastuzumab vs our 11-13-mers on a different framework). TAP is a TAP-style reproduction (official
code not public). The multimodal result is a rigorous negative on a single task. The path to a
stronger preprint: a joint developability-and-binding target where fusion can genuinely help,
multiple frameworks (ANARCI/IMGT), and the controlled comparison against these baselines.

## Files

- `notebooks/` - the 7 executed notebooks (.ipynb)
- `html/` - standalone HTML exports (open in any browser, no Jupyter needed)
- `src/` - oracles.py, reward.py, binding_oracle.py, config.py (single source of truth)
- `figures/` - all publication figures (.png)
- `data/` - Mason HER2 CSVs, generated pools, ablation results (.json/.csv)
- `models/` - trained binding classifiers (.joblib)
- `structures/` - example predicted Fv structure (.pdb)
- `documents/` - manuscript skeleton, presentation, architecture schematics

## Environment

Python 3.11 + PyTorch (CPU), IgLM (transformers 4.36.2), AbLang2, ImmuneBuilder (ABodyBuilder2),
ANARCI, freesasa, ESM2, scikit-learn, Biopython. All runs are CPU-only.
