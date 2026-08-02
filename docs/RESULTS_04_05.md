# Stages 04 and 05: reward and RL loop

Completes the antibody CDRH3 design project (notebooks 00-02 done previously).
Built following AB-Gen (Xu et al. 2023) with a REINVENT augmented-likelihood loop
(Olivecrona et al. 2017). Everything was run and validated on real IgLM + AbLang2,
on CPU.

## What was added

### 04_reward.ipynb
- `property_score(oracle_scores)` -> S(x) in [0, 1]: weighted sum of the three
  oracles (humanness 0.50, liability 0.30, protparam 0.20). Weights are the
  multi-objective design and the first knob for ablation.
- `score_sequence(full_sequence)`: live path from a raw heavy chain to S(x).
- `augmented_log_likelihood(prior_logp, S, sigma)`: the REINVENT target
  log P_prior(x) + sigma * S(x). The prior term is the leash against reward hacking.

### 05_rl_loop.ipynb
- `span_logp(model, seq)`: differentiable CDRH3 log-likelihood, validated to match
  IgLM's native `log_likelihood` to 1e-3.
- `reinvent_step(seqs)`: one augmented-likelihood update, with a degenerate-length
  guard and gradient clipping.
- A multi-objective training run, a proxy-reward positive control, and the four
  AB-Gen benchmark metrics.

### src/
- `oracles.py`, `reward.py`: the oracle and reward definitions as importable
  modules so both notebooks share one source of truth.

## Key results (real runs, CPU, 12.9M IgLM checkpoint)

1. The loop is correct and stable. The differentiable likelihood matches IgLM
   exactly, and the agent measurably departs the prior (KL proxy goes negative)
   in every run.

2. Positive control: with a headroom-rich proxy reward (CDRH3 tyrosine fraction),
   the agent moved the metric from 0.11 to a peak of 0.54 at step 30, then settled
   near 0.43 over the final steps (last-5 mean 0.43). The clean rise off the prior
   baseline proves the policy-gradient machinery works.

3. The developability oracles saturate. Multi-objective S(x) starts at 0.78 and
   stays near 0.76; humanness-only stays near 0.64. IgLM output already scores
   high on these oracles (as notebook 02 noted), so there is little to optimize.
   This is an oracle-headroom limit, not a loop failure.

4. Reward hacking observed and controlled. An early high-sigma run collapsed CDRH3
   toward very short loops (a per-token-sum reward favors shorter sequences). The
   prior leash plus the length guard stop it.

5. Benchmark (proxy-trained agent vs prior, 80 sequences each):
   uniqueness 1.00 vs 1.00, novelty 1.00 (all agent sequences new vs prior pool),
   diversity 0.81 vs 0.91 (expected drop from concentrating on tyrosine),
   success rate 0.53 vs 0.55.

## What this implies for the arXiv path

The mechanism is done and demonstrated. The single most valuable next step is a
reward with real headroom: a binding/target oracle (HER2 to mirror AB-Gen, or
AlphaSeq SARS-CoV-2 data). The developability oracles alone leave nothing for RL
to improve. After that, multiple frameworks (ANARCI/IMGT) and an oracle ablation
turn this into a controlled study.

## Runtime note

The `iglm` PyPI package loads a 12.9M-parameter checkpoint, not the 558M model in
the spec. This is why the loop runs on CPU at all. On the Colab T4, raise BATCH
and STEPS; the code is unchanged.
