"""Oracles: sequence in, score in [0, 1]. Extracted from notebook 02 so both
the reward (04) and the RL loop (05) import one definition. All score the CDRH3.
"""
import re
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

INFILL_START = 98
CLOSING_MOTIF = "WGQGTLVTVS"

def extract_cdrh3(sequence: str) -> str:
    end = sequence.index(CLOSING_MOTIF)
    return sequence[INFILL_START:end]

# ---- Oracle 1: chemical liabilities ----
LIABILITY_MOTIFS = {
    "deamidation_NG": "NG", "deamidation_NS": "NS",
    "isomerization_DG": "DG", "isomerization_DS": "DS",
    "oxidation_M": "M", "nglyc_sequon": "N[^P][ST]", "fragmentation_DP": "DP",
}
PENALTY_PER_MOTIF = 0.15  # TUNABLE

def liability_oracle(cdrh3: str) -> float:
    hits = sum(len(re.findall(p, cdrh3)) for p in LIABILITY_MOTIFS.values())
    return max(0.0, 1.0 - PENALTY_PER_MOTIF * hits)

# ---- Oracle 2: physicochemical (ProtParam) ----
CHARGE_RANGE = (-2.0, 2.0)  # TUNABLE
GRAVY_RANGE = (-1.0, 0.0)   # TUNABLE

def _range_score(v, lo, hi):
    if lo <= v <= hi: return 1.0
    w = hi - lo; d = (lo - v) if v < lo else (v - hi)
    return max(0.0, 1.0 - d / w)

def protparam_oracle(cdrh3: str) -> float:
    a = ProteinAnalysis(cdrh3)
    return (_range_score(a.charge_at_pH(7.0), *CHARGE_RANGE)
            + _range_score(a.gravy(), *GRAVY_RANGE)) / 2.0

# ---- Oracle 3: humanness (AbLang2) ----
# The AbLang2 model is loaded by the caller and assigned to ABLANG_MODEL, so this
# module stays import-cheap and testable. See notebook 02 for the loader.
ABLANG_MODEL = None
START_TOKEN_OFFSET = 1

def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def humanness_oracle(full_sequence: str) -> float:
    if ABLANG_MODEL is None:
        raise RuntimeError("Set oracles.ABLANG_MODEL = ablang2.pretrained() first.")
    per_pos = ABLANG_MODEL([[full_sequence, ""]], mode="likelihood")
    probs = _softmax(per_pos[0])
    s = INFILL_START; e = full_sequence.index(CLOSING_MOTIF)
    vals = [probs[pos + START_TOKEN_OFFSET, ABLANG_MODEL.tokenizer.aa_to_token[full_sequence[pos]]]
            for pos in range(s, e)]
    return float(np.mean(vals))
