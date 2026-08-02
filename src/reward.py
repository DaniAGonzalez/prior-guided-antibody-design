"""Reward: combine oracle scores into one number and form the REINVENT augmented
likelihood. Extracted from notebook 04.
"""

# TUNABLE: multi-objective weights, must sum to 1.0.
ORACLE_WEIGHTS = {"humanness": 0.50, "liability": 0.30, "protparam": 0.20}
assert abs(sum(ORACLE_WEIGHTS.values()) - 1.0) < 1e-9

# TUNABLE: property-vs-prior trade-off in the augmented likelihood.
SIGMA = 15.0

def property_score(oracle_scores: dict) -> float:
    """Weighted sum of per-oracle scores -> S(x) in [0, 1]."""
    return float(sum(ORACLE_WEIGHTS[k] * oracle_scores[k] for k in ORACLE_WEIGHTS))

def augmented_log_likelihood(prior_logp: float, s_score: float, sigma: float = SIGMA) -> float:
    """REINVENT target: log P_prior(x) + sigma * S(x)."""
    return prior_logp + sigma * s_score
