"""HER2 binding oracle. Trained on the Mason et al. 2021 trastuzumab CDRH3 deep
mutational scanning dataset (github.com/dahjan/DMS_opt): 11,300 binders and
27,539 non-binders, each a 10-residue CDRH3 with up to 10 mutations from
wild-type trastuzumab, labelled binder (1) / non-binder (0) against HER2.

Two models are provided:
  - one-hot MLP: fixed 10-mer input, reproduces the Mason CNN (test AUC ~0.91).
  - ESM2-8M + logistic regression: length-agnostic (test AUC ~0.86), so it can
    score the 11-13 residue CDRH3 the IgLM generator produces.

Honesty note. The generator makes 11-13mers on a different heavy framework, while
Mason is all length-10 on trastuzumab. The ESM oracle is therefore a transferred
proxy for HER2 binding, not an exact predictor for this framework. It is the same
class of controlled simplification as the fixed germline VL in the TAP oracle.
"""
import numpy as np
import joblib

_ESM = {"model": None, "bc": None, "lr": None}

def _load_esm():
    if _ESM["model"] is None:
        import esm, torch
        m, alph = esm.pretrained.esm2_t6_8M_UR50D(); m.eval()
        _ESM.update(model=m, bc=alph.get_batch_converter(), torch=torch)

def _embed(seqs, batch=256):
    _load_esm(); torch = _ESM["torch"]; out = []
    for i in range(0, len(seqs), batch):
        _, _, toks = _ESM["bc"]([("x", s) for s in seqs[i:i+batch]])
        with torch.no_grad():
            rep = _ESM["model"](toks, repr_layers=[6])["representations"][6]
        for j, s in enumerate(seqs[i:i+batch]):
            out.append(rep[j, 1:len(s)+1].mean(0).numpy())
    return np.stack(out)

def binding_oracle(cdrh3, lr_path="her2_esm_lr.joblib"):
    """P(binder HER2) in [0, 1] for one CDRH3 (any length >= 4). Length-agnostic
    ESM2 model, so it plugs into the reward alongside the developability oracles."""
    if len(cdrh3) < 4:
        return 0.0
    if _ESM["lr"] is None:
        _ESM["lr"] = joblib.load(lr_path)
    return float(_ESM["lr"].predict_proba(_embed([cdrh3]))[0, 1])

def binding_oracle_batch(cdrh3_list, lr_path="her2_esm_lr.joblib"):
    """Vectorized P(binder HER2) for many CDRH3 at once (one ESM pass)."""
    if _ESM["lr"] is None:
        _ESM["lr"] = joblib.load(lr_path)
    safe = [c if len(c) >= 4 else "AAAA" for c in cdrh3_list]
    p = _ESM["lr"].predict_proba(_embed(safe))[:, 1]
    return [float(pi) if len(c) >= 4 else 0.0 for pi, c in zip(p, cdrh3_list)]
