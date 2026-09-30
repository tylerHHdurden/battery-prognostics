"""
Phase 3: nearest-source trust report - query-time module.

Loads `models/_source_profiles.pkl` (built by `build_source_trust_
profiles.py`) and answers, for any battery's own 25-dim feature vector
(same construction `live_inference.py` uses): which known source is
this battery MOST LIKE, how confident should that be treated (trust
level), and what error rate has actually been MEASURED for that source
(not invented).

NEAREST-SOURCE METHOD: Gaussian negative log-likelihood (Mahalanobis^2
+ log|covariance|) across all 16 source profiles - NOT raw Mahalanobis
distance alone, which `build_source_trust_profiles.py`'s own validation
found to be a badly biased comparator across profiles with very
different covariance "volume" (see that module's docstring for the
full finding: 11%->44%->88-92% nearest-source accuracy across the
fixes this went through, all measured, not assumed).

TRUST LEVEL: per the spec - the query point's RAW Mahalanobis distance
to its ASSIGNED (nearest-by-log-likelihood) source's own profile,
compared against THAT source's own 95th/99th percentile of its own
held-out battery distances (a same-profile self-comparison, unaffected
by the cross-profile bias the log-likelihood fix addresses):
  - "familiar"          if distance <= that source's own 95th pctile
  - "somewhat_familiar"  if <= that source's own 99th pctile
  - "unfamiliar"         otherwise

MEASURED ERROR: the gate-table MAE (`toolkit_phase2_gate_table.csv`'s
own `candidate_mae` - the candidate model's own battery-level-split
evaluation, the SAME split the deployed routing decision was made on)
when "familiar"; the STRICTER LODO family-holdout MAE
(`finalpass3_check1_lodo_family_holdout.csv`'s `family_lodo_mae` - a
genuinely held-out-SOURCE test, not just held-out-battery) otherwise -
matching the spec's own "gate-table MAE if familiar, LODO family-
holdout MAE otherwise" rule. Tongji predates that LODO file (Phase
1(b) integration came after it) - falls back to Phase 2B's own
`centralized_mae` for tongji specifically, which is itself a genuine
LODO-family-holdout MAE (same methodology, computed later, across all
16 sources) - disclosed, not silently substituted.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

_CACHE = {}


def _load():
    if "profiles" in _CACHE:
        return _CACHE["profiles"]
    import encoder_provenance as ep
    ep.assert_store(MODELS_DIR / "_source_profiles.pkl", ep.CAND, "trust_report: profiles must be built on candidate-encoder embeddings")
    with open(MODELS_DIR / "_source_profiles.pkl", "rb") as f:
        data = pickle.load(f)
    # TEST HOOK (step 3 end-to-end validation only): TRUST_EXCLUDE_SOURCES="mich,mich_exp" removes those profiles from the set so a
    # real battery of a source can be scored as if its source were unknown (leave-source-out). Unset in normal use.
    import os
    _excl = {x.strip() for x in os.environ.get("TRUST_EXCLUDE_SOURCES", "").split(",") if x.strip()}
    if _excl:
        data["profiles"] = {k: v for k, v in data["profiles"].items() if k not in _excl}

    gate_table = pd.read_csv(OUT_DIR / "toolkit_phase2_gate_table.csv").set_index("source")["candidate_mae"].to_dict()
    lodo_table = pd.read_csv(OUT_DIR / "finalpass3_check1_lodo_family_holdout.csv") \
        .set_index("held_out_source")["family_lodo_mae"].to_dict()
    p2b_table = pd.read_csv(OUT_DIR / "toolkit_phase2b_federated_results.csv") \
        .set_index("held_out_source")["centralized_mae"].to_dict()
    # tongji has no LODO family-holdout entry (predates that file) - falls back to Phase 2B's own
    # centralized_mae, itself a genuine LODO-family-holdout MAE, computed later, across all 16 sources.
    lodo_table.setdefault("tongji", p2b_table.get("tongji"))

    data["gate_table_mae"] = gate_table
    data["lodo_mae"] = lodo_table
    _CACHE["profiles"] = data
    return data


def _mahalanobis(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> float:
    d = x - mean
    quad = float(d @ precision @ d)
    assert np.isfinite(quad), "non-finite Mahalanobis quadratic form - a NaN/Inf reached this " \
        "point uncaught (should be impossible after _prepare_vector's own imputation+assert)"
    return float(np.sqrt(max(0.0, quad)))  # only clamps genuine tiny FP noise now, never a real NaN


def _neg_log_likelihood(x: np.ndarray, mean: np.ndarray, precision: np.ndarray, logdet_cov: float) -> float:
    d = x - mean
    quad = float(d @ precision @ d)
    assert np.isfinite(quad), "non-finite log-likelihood quadratic form - see _mahalanobis's own note"
    return quad + logdet_cov


def _prepare_vector(feature_vector: np.ndarray, col_medians: np.ndarray, scale_mean: np.ndarray,
                     scale_std: np.ndarray) -> np.ndarray:
    """Impute NaN/Inf with the SAME pooled column medians the profiles
    were built from (found necessary, not defensive: a real caller's
    vector CAN have genuine NaN HI columns - e.g. a no-temperature-
    channel source - and without this step a NaN silently propagated
    all the way to `max(0.0, nan)` clipping to an exactly-0.0 distance,
    reporting a confidently-wrong "familiar" verdict instead of a
    correctly-imputed one - caught directly via a real HUST battery
    that happened to have a NaN MET value, not found by inspection)."""
    x = np.asarray(feature_vector, dtype=float).copy()
    x = np.where(np.isinf(x), np.nan, x)
    nan_mask = np.isnan(x)
    x[nan_mask] = col_medians[nan_mask]
    x_std = (x - scale_mean) / scale_std
    assert np.all(np.isfinite(x_std)), "non-finite value survived imputation - real bug, do not proceed"
    return x_std


def nearest_source_trust_report(feature_vector: np.ndarray) -> dict:
    """feature_vector: the SAME 25-dim vector live_inference.py builds
    for one cycle (8 reformulated HI + cycle_idx + 16 candidate fusion
    dims), OR a battery's own mean/median across cycles - whichever the
    caller has on hand; this function does not average across cycles
    itself (the caller decides what "this battery's own vector" means
    for its context, e.g. live_inference.py's per-cycle call site vs. a
    whole-battery upload summary)."""
    data = _load()
    profiles = data["profiles"]
    x = _prepare_vector(feature_vector, data["col_medians"], data["scale_mean"], data["scale_std"])

    nll = {name: _neg_log_likelihood(x, p["mean"], p["precision"], p["logdet_cov"]) for name, p in profiles.items()}
    nearest = min(nll, key=nll.get)
    prof = profiles[nearest]
    dist = _mahalanobis(x, prof["mean"], prof["precision"])

    if dist <= prof["threshold_familiar"]:
        trust = "familiar"
    elif dist <= prof["threshold_somewhat_familiar"]:
        trust = "somewhat_familiar"
    else:
        trust = "unfamiliar"

    if trust == "familiar":
        measured_mae = data["gate_table_mae"].get(nearest)
        error_source = "gate-table (battery-level split)"
    else:
        measured_mae = data["lodo_mae"].get(nearest)
        error_source = "LODO family-holdout (held-out SOURCE, stricter)"

    return {
        "nearest_source": nearest,
        "nll_min": float(nll[nearest]),  # the out-of-domain score (see live_inference.OOD_* constants)
        "gate_table_mae": data["gate_table_mae"].get(nearest),
        "lodo_family_mae": data["lodo_mae"].get(nearest),
        "trust_level": trust,
        "distance_to_nearest": dist,
        "distance_threshold_familiar": prof["threshold_familiar"],
        "distance_threshold_somewhat_familiar": prof["threshold_somewhat_familiar"],
        "measured_mae": measured_mae,
        "measured_mae_source": error_source,
        "n_train_batteries_for_nearest": prof["n_train_batteries"],
    }


if __name__ == "__main__":
    # smoke test: query a real battery's own vector against itself
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_toolkit_phase2b_federated import load_pooled_data
    from build_source_trust_profiles import battery_mean_vectors

    sources, all_cols, _ = load_pooled_data()
    for test_source in ["HUST", "tongji", "NASA"]:
        df = sources[test_source]
        bvecs = battery_mean_vectors(df, all_cols)
        any_bid = next(iter(bvecs))
        report = nearest_source_trust_report(bvecs[any_bid])
        print(f"query battery {any_bid} (true source={test_source}): {report}")
