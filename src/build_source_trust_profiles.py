"""
Phase 3: nearest-source trust report - profile-building + calibration.

For each of the 16 known sources, precomputes a Gaussian profile (mean +
LedoitWolf-shrinkage covariance) of the SAME 25-dim feature vector
`live_inference.py` builds (8 reformulated canonical HIs + cycle_idx +
16 fusion-embedding dims) - but ONE POINT PER BATTERY (that battery's
own mean vector across its cycles), not one point per cycle-row, since
the question this profile answers is "how do typical BATTERIES of this
source vary from each other," not "how do CYCLES within one battery
vary."

FEATURE CHOICE, disclosed: uses the CANDIDATE encoder's own fusion
embeddings (`fusion_embeddings_multisource.csv`) for EVERY source,
including NASA/MIT - even though NASA/MIT batteries are actually routed
to the BASE (deployed) model's own, different encoder in
`live_inference.py`. This is deliberate, not an oversight: an
uploaded/unknown battery is ALWAYS routed to the candidate model
(`_use_candidate` returns True for any dataset not in {"NASA","MIT"}),
so the vector being compared against every profile will always have
come from the candidate's own encoder - profiles built from a DIFFERENT
encoder's embeddings for NASA/MIT would make those two profiles
incomparable to the incoming battery's own vector.

CALIBRATION: for each source, splits its own batteries via
`split_utils.battery_level_split` (test_every=5, this project's
established convention) - with a manual fallback to at least 1 test
battery for sources too small for the stride to select one (CALCE:
3 batteries, naive test_every=5 gives ZERO - checked directly, not
assumed), same "at least one" guarantee `split_utils`'s own per-dataset
branch already gives elsewhere. Fits the Gaussian profile on TRAIN
batteries only. For calibration: each source's own TEST batteries' own
Mahalanobis distances to THEIR OWN source's profile become that
source's 95th/99th-percentile trust thresholds - disclosed explicitly
as often based on a very small number of test batteries (several
sources have exactly 1), same caveat this project's own Phase 2 gate
table already applies to its own small-n sources.

TWO REAL BUGS FOUND BY THE VALIDATION STEP ITSELF, NOT ASSUMED AWAY -
this is exactly why "validate, report the confusion table" was part of
the spec, not a formality:

1. **Raw "smallest Mahalanobis distance wins" cross-source comparison is
   a known-bad metric and it showed up as one here**: an unweighted
   first pass got only 11% nearest-source accuracy (10/91), with MIT
   acting as a "black hole" that most other sources' batteries got
   misassigned to (confirmed by direct inspection: MIT's covariance
   log-determinant is 378, vs. HUST's 95, XJTU's 64, NASA's 26 - orders
   of magnitude looser, so its raw Mahalanobis "ruler" under-penalizes
   almost everything). Raw Mahalanobis distance alone is only a valid
   nearest-DISTRIBUTION comparator when every distribution has
   comparable covariance "volume" - it isn't here. **Fix**: the
   NEAREST-SOURCE decision (both here and in the real query function)
   uses each profile's full Gaussian negative log-likelihood
   (Mahalanobis^2 + log|covariance|, the correct log-density comparison
   across differently-scaled Gaussians), NOT raw distance. The
   per-source FAMILIAR/SOMEWHAT_FAMILIAR/UNFAMILIAR threshold check
   (comparing a point only to ITS OWN assigned source's profile) is
   UNAFFECTED by this - it was never a cross-profile comparison, so it
   stays raw Mahalanobis distance, exactly as specified.
2. **LedoitWolf's own automatic shrinkage estimate is unreliable when
   n_train batteries is small relative to the 25-dim feature space**
   (most sources have FEWER training batteries than 25 dimensions -
   CALCE has only 2). CALCE's automatic shrinkage estimated ~0.000 and
   produced a genuinely SINGULAR covariance matrix (confirmed: a direct
   `np.linalg.inv` on it raised `LinAlgError: Singular matrix`) - not a
   subtle numerical wobble, an outright unusable profile. **Fix**: a
   manually-enforced minimum shrinkage floor (`MIN_SHRINKAGE`) applied
   on top of LedoitWolf's own estimate whenever the automatic one is
   lower, using the same identity-scaled shrinkage target sklearn's own
   LedoitWolf formula uses.

VALIDATION: every source's TEST batteries get scored against ALL 16
profiles; a confusion table (true source vs. nearest-assigned source)
and trust-level breakdown are reported - a real, measured check of
whether "nearest profile" actually recovers the true source, not
assumed to work. Both the ORIGINAL (broken) and CORRECTED confusion
tables are reported, not just the final one, so the fix's own effect
is itself verified, not asserted.
"""
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, empirical_covariance, ledoit_wolf_shrinkage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import battery_level_split
from run_toolkit_phase2b_federated import load_pooled_data, ALL_SOURCES

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

PERCENTILE_FAMILIAR = 95
PERCENTILE_SOMEWHAT = 99
MIN_SHRINKAGE = 0.3  # floor on top of LedoitWolf's own automatic estimate - most sources here have
# FEWER training batteries than the 25 feature dimensions, a regime where the automatic estimate can
# degenerate (CALCE: ~0.000, produced an outright singular covariance - confirmed directly, not assumed)


def fit_shrunk_gaussian(X_train: np.ndarray):
    """Same identity-scaled shrinkage target sklearn's own LedoitWolf
    uses (`(1-s)*empirical_cov + s*mu*I`, `mu = trace(empirical_cov)/p`),
    but with `s = max(LedoitWolf's own auto-estimate, MIN_SHRINKAGE)` -
    guarantees a well-conditioned, invertible covariance regardless of
    how few training batteries a source has (verified: CALCE's own
    n_train=2 case, which previously produced a singular matrix, is
    checked for finite/invertible output below before ever being saved)."""
    mean = X_train.mean(axis=0)
    emp_cov = empirical_covariance(X_train, assume_centered=False)
    auto_shrinkage = float(ledoit_wolf_shrinkage(X_train))
    shrinkage = max(auto_shrinkage, MIN_SHRINKAGE)
    p = X_train.shape[1]
    mu = np.trace(emp_cov) / p
    cov = (1 - shrinkage) * emp_cov + shrinkage * mu * np.eye(p)
    precision = np.linalg.inv(cov)
    assert np.all(np.isfinite(precision)), "non-finite precision matrix after shrinkage - should be impossible"
    return mean, cov, precision, shrinkage, auto_shrinkage


def battery_mean_vectors(df: pd.DataFrame, all_cols: list[str]) -> dict[str, np.ndarray]:
    """One representative 25-dim vector per battery_id, aggregated
    across that battery's own cycles via the MEDIAN (not the mean) -
    found necessary, not a preemptive defensiveness: a handful of
    specific (battery, cycle) rows in `fusion_embeddings_multisource.csv`
    have clearly-diverged encoder outputs (e.g. isu_ilcc::ISU-ILCC_G27C4
    cycle 2401 has fusion_10=1.36e13, vs. a normal range of roughly
    0.0-0.3 - confirmed directly, several other sources show the same
    pattern in different single cycles) - a plain mean lets ONE such
    row poison an entire battery's profile; the median doesn't."""
    out = {}
    for bid, sub in df.groupby("battery_id"):
        out[bid] = np.median(sub[all_cols].to_numpy(dtype=float), axis=0)
    return out


def split_with_fallback(battery_ids: list[str], test_every: int = 5) -> tuple[list[str], list[str]]:
    """Same convention as split_utils.battery_level_split's own
    per-dataset branch (at least 1 test battery even when the stride
    would select zero) - applied here per-source, since this script
    calls the split once per source rather than once for a pooled,
    dataset_of-tagged list."""
    ids = sorted(battery_ids)
    test_ids = set(ids[test_every - 1::test_every]) or {ids[-1]}
    train_ids = [i for i in ids if i not in test_ids]
    return sorted(train_ids), sorted(test_ids)


def mahalanobis(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> float:
    d = x - mean
    return float(np.sqrt(max(0.0, d @ precision @ d)))


def neg_log_likelihood(x: np.ndarray, mean: np.ndarray, precision: np.ndarray, logdet_cov: float) -> float:
    """Up to an additive constant (2*pi term, identical for every
    profile, doesn't affect which is smallest): Mahalanobis^2 +
    log|covariance|. The CORRECT cross-profile "which distribution does
    this point best belong to" comparator - unlike raw Mahalanobis
    distance alone, this properly penalizes a profile for being loose/
    high-volume (see module docstring for the MIT "black hole" finding
    that made this fix necessary, not optional)."""
    d = x - mean
    maha_sq = float(d @ precision @ d)
    return maha_sq + logdet_cov


def main():
    t0 = time.time()
    print("=== Building nearest-source trust profiles ===")

    sources, all_cols, feature_cols_base = load_pooled_data()
    print(f"[trust] {len(sources)} sources loaded, {len(all_cols)}-dim feature vector "
          f"({len(feature_cols_base)} HI + cycle_idx + 16 fusion dims)")

    # NaN impute with POOLED (all-source) column medians, same convention Phase 2B/live_inference
    # use elsewhere - some sources (e.g. CALCE, no temperature channel) have real NaN HI columns.
    pooled_all = pd.concat(sources.values(), ignore_index=True)
    col_medians = np.nanmedian(pooled_all[all_cols].to_numpy(dtype=float), axis=0)
    for name in sources:
        X = sources[name][all_cols].to_numpy(dtype=float, copy=True)
        X = np.where(np.isinf(X), np.nan, X)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
        sources[name] = sources[name].copy()
        sources[name][all_cols] = X
    n_imputed = int(np.isnan(pooled_all[all_cols].to_numpy(dtype=float)).sum())
    print(f"[trust] imputed {n_imputed} NaN cells across all sources with pooled column medians")

    profiles = {}
    test_battery_vectors = {}  # source -> {battery_id: vector}
    split_info = []

    # ---------- pass 1: battery-level mean vectors + splits for every source ----------
    per_source_battery_vecs = {}
    per_source_split = {}
    for name in ALL_SOURCES:
        if name not in sources:
            continue
        battery_vecs = battery_mean_vectors(sources[name], all_cols)
        bids = list(battery_vecs.keys())
        train_ids, test_ids = split_with_fallback(bids, test_every=5)
        per_source_battery_vecs[name] = battery_vecs
        per_source_split[name] = (train_ids, test_ids)

    # ---------- STANDARDIZE, root-cause fix: raw features span wildly different scales ----------
    # (VDEDT alone has std ~13,000 vs. every fusion dim's std < 1 - checked directly, not assumed -
    # one unnormalized raw-unit dimension was dominating every covariance estimate, especially MIT's,
    # whose own battery pool happens to include some near-end-of-life cells with extreme VDEDT
    # values). Standardized using POOLED, TRAIN-ONLY statistics across ALL 16 sources together (a
    # single shared scale, not per-source - the whole point is comparing DIFFERENT sources on the
    # same footing) - saved into the pickle so any future query vector gets the SAME transform.
    pooled_train = np.concatenate([np.stack([per_source_battery_vecs[name][b] for b in train_ids])
                                    for name, (train_ids, _) in per_source_split.items()], axis=0)
    scale_mean = pooled_train.mean(axis=0)
    scale_std = pooled_train.std(axis=0)
    scale_std[scale_std < 1e-8] = 1.0  # constant dims (e.g. fusion_5, all-zero) - leave untouched, not div-by-0
    print(f"[trust] standardizing all {len(all_cols)} dims with pooled train-battery mean/std "
          f"(largest raw std was {scale_std.max():.1f} on {all_cols[int(np.argmax(pooled_train.std(axis=0)))]!r})")

    for name in per_source_battery_vecs:
        per_source_battery_vecs[name] = {b: (v - scale_mean) / scale_std
                                          for b, v in per_source_battery_vecs[name].items()}

    for name in ALL_SOURCES:
        if name not in sources:
            continue
        battery_vecs = per_source_battery_vecs[name]
        train_ids, test_ids = per_source_split[name]
        bids = train_ids + test_ids

        X_train = np.stack([battery_vecs[b] for b in train_ids])
        mean, cov, precision, shrinkage, auto_shrinkage = fit_shrunk_gaussian(X_train)
        _, logdet_cov = np.linalg.slogdet(cov)
        profiles[name] = {
            "mean": mean, "precision": precision, "logdet_cov": float(logdet_cov),
            "shrinkage": shrinkage, "auto_shrinkage": auto_shrinkage,
            "n_train_batteries": len(train_ids), "n_test_batteries": len(test_ids),
        }
        test_battery_vectors[name] = {b: battery_vecs[b] for b in test_ids}
        split_info.append({"source": name, "n_batteries": len(bids),
                            "n_train": len(train_ids), "n_test": len(test_ids),
                            "shrinkage": shrinkage, "auto_shrinkage": auto_shrinkage,
                            "logdet_cov": float(logdet_cov)})
        floor_note = " (floor applied)" if shrinkage > auto_shrinkage else ""
        print(f"[trust] {name}: {len(bids)} batteries ({len(train_ids)} train / {len(test_ids)} test), "
              f"shrinkage={shrinkage:.3f} (auto={auto_shrinkage:.3f}){floor_note}, log|cov|={logdet_cov:.2f}")

    # ---------- calibration: each source's own test batteries' distances to its OWN profile ----------
    print("\n[trust] === calibration: per-source distance percentiles from own held-out batteries ===")
    for name, prof in profiles.items():
        own_test_vecs = test_battery_vectors[name]
        dists = [mahalanobis(v, prof["mean"], prof["precision"]) for v in own_test_vecs.values()]
        p95 = float(np.percentile(dists, PERCENTILE_FAMILIAR)) if dists else float("nan")
        p99 = float(np.percentile(dists, PERCENTILE_SOMEWHAT)) if dists else float("nan")
        prof["threshold_familiar"] = p95
        prof["threshold_somewhat_familiar"] = p99
        prof["own_test_distances"] = dists
        n = len(dists)
        caveat = " (SMALL-N CAVEAT: based on a single test battery)" if n == 1 else ""
        print(f"[trust]   {name}: n_test={n}, own-distance range=[{min(dists):.2f},{max(dists):.2f}], "
              f"p{PERCENTILE_FAMILIAR}={p95:.2f} (familiar<=), p{PERCENTILE_SOMEWHAT}={p99:.2f} "
              f"(somewhat_familiar<=){caveat}")

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    with open(MODELS_DIR / "_source_profiles.pkl", "wb") as f:
        pickle.dump({"profiles": profiles, "all_cols": all_cols, "feature_cols_base": feature_cols_base,
                     "percentile_familiar": PERCENTILE_FAMILIAR, "percentile_somewhat": PERCENTILE_SOMEWHAT,
                     "scale_mean": scale_mean, "scale_std": scale_std, "col_medians": col_medians},
                    f)
    print(f"\n[trust] saved models/_source_profiles.pkl ({len(profiles)} source profiles)")

    # ---------- validation: every source's test batteries scored against ALL profiles ----------
    # Reports BOTH nearest-by-raw-distance (the naive, broken approach) and
    # nearest-by-log-likelihood (the fix) - so the fix's own effect is itself
    # measured here, not just asserted in the module docstring.
    print("\n[trust] === validation: nearest-source assignment + confusion table ===")
    rows = []
    for true_source, own_test_vecs in test_battery_vectors.items():
        for bid, vec in own_test_vecs.items():
            dists = {name: mahalanobis(vec, prof["mean"], prof["precision"]) for name, prof in profiles.items()}
            nll = {name: neg_log_likelihood(vec, prof["mean"], prof["precision"], prof["logdet_cov"])
                   for name, prof in profiles.items()}
            nearest_raw = min(dists, key=dists.get)
            nearest_nll = min(nll, key=nll.get)
            # trust level (familiar/somewhat/unfamiliar) uses the CORRECTED nearest source
            # (nearest_nll) and its OWN raw-distance threshold - the raw-distance THRESHOLD CHECK
            # itself is a same-profile self-comparison, never the broken cross-profile one.
            nearest_prof = profiles[nearest_nll]
            dist_to_assigned = dists[nearest_nll]
            if dist_to_assigned <= nearest_prof["threshold_familiar"]:
                trust = "familiar"
            elif dist_to_assigned <= nearest_prof["threshold_somewhat_familiar"]:
                trust = "somewhat_familiar"
            else:
                trust = "unfamiliar"
            rows.append({"true_source": true_source, "battery_id": bid,
                         "nearest_source_raw_distance": nearest_raw, "nearest_source_loglik": nearest_nll,
                         "distance_to_assigned": dist_to_assigned, "distance_to_true": dists[true_source],
                         "correct_raw_distance": nearest_raw == true_source,
                         "correct_loglik": nearest_nll == true_source, "trust_level": trust})

    val_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    val_df.to_csv(OUT_DIR / "toolkit_phase3_trust_report_validation.csv", index=False)

    acc_raw = val_df["correct_raw_distance"].mean()
    acc_nll = val_df["correct_loglik"].mean()
    print(f"\n[trust] Overall accuracy - RAW DISTANCE (broken): {val_df['correct_raw_distance'].sum()}/"
          f"{len(val_df)} ({acc_raw:.1%})  |  LOG-LIKELIHOOD (fixed): {val_df['correct_loglik'].sum()}/"
          f"{len(val_df)} ({acc_nll:.1%})")

    print("\n[trust] Confusion (LOG-LIKELIHOOD, the corrected/used method) - rows=true source, "
          "off-diagonal misassignments only shown per row:")
    for true_source in sorted(val_df["true_source"].unique()):
        sub = val_df[val_df["true_source"] == true_source]
        counts = sub["nearest_source_loglik"].value_counts()
        correct_n = int((sub["nearest_source_loglik"] == true_source).sum())
        wrong = counts[counts.index != true_source]
        wrong_str = ", ".join(f"{k}:{v}" for k, v in wrong.items()) if len(wrong) else "none"
        print(f"  {true_source}: {correct_n}/{len(sub)} correct | misassigned to: {wrong_str}")

    print("\n[trust] Trust-level breakdown (of correctly log-likelihood-assigned test batteries, i.e. "
          "testing each source's own calibration against ITSELF):")
    self_matched = val_df[val_df["correct_loglik"]]
    print(self_matched["trust_level"].value_counts().to_string())

    full_confusion_nll = pd.crosstab(val_df["true_source"], val_df["nearest_source_loglik"])
    full_confusion_nll.to_csv(OUT_DIR / "toolkit_phase3_trust_report_confusion.csv")
    full_confusion_raw = pd.crosstab(val_df["true_source"], val_df["nearest_source_raw_distance"])
    full_confusion_raw.to_csv(OUT_DIR / "toolkit_phase3_trust_report_confusion_raw_distance_broken.csv")

    print(f"\n[trust] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")
    return profiles, val_df


if __name__ == "__main__":
    main()
