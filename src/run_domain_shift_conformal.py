"""
Session 19: domain-shift-aware conformal prediction, addressing session
13's finding directly - fixing point-prediction accuracy via MMD
alignment did NOT fix conformal coverage on CALCE (6.1% original,
4.4% after MMD-recalibration), because this project's split-conformal
implementation uses ONE global fixed-width interval with no per-input
adaptivity: it cannot tell a given point is out-of-domain.

Implements WEIGHTED split-conformal prediction (Tibshirani, Barber,
Candes, Ramdas 2019, "Conformal Prediction Under Covariate Shift"):
calibration residuals are reweighted by an estimated covariate-shift
density ratio w(x) = P_target(x) / P_calib(x), estimated via a
lightweight logistic-regression domain classifier (calibration-domain
vs. target-domain features) - the exact "logistic-regression domain
classifier" approach named in the task. No base model is retrained:
this is a purely post-hoc reweighting of the EXISTING MMD-aligned
ensemble's (session 13) calibration residuals, applied identically to
whichever domain (NASA+MIT eval-half or CALCE) is being scored.

Zero additional label leakage beyond what session 13 already used: the
domain classifier is trained on FEATURES only (7 BFA-selected HIs + the
16-dim MMD-aligned fusion embedding - the exact feature space the
XGBoost-fusion/Ridge-meta model actually predicts from), never on
CALCE's SOH/RUL labels.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters import iterate_calce_cycles
from sequence_features import build_dataset_tensors, apply_channel_norm
from models.ica_encoder import ICAEncoder
from run_conformal import calib_eval_battery_split

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]
ICA_CHANNEL_SLICE = slice(3, 6)
ALPHA = 0.1  # 90% target coverage, same as every other conformal script in this project
PROB_CLIP = (0.01, 0.99)  # clip domain-classifier probabilities before taking odds - avoids infinite weights from a single near-certain prediction dominating everything
_CALCE_FUSION_CACHE = PROC_DIR / "_calce_fusion_mmd_cache.csv"


def build_calce_fusion_embeddings():
    """Rebuilds CALCE's 16-dim MMD-aligned fusion embedding per cycle -
    session 13's eval script computed this internally but never saved it
    to disk (only the final SOH predictions), so it must be rebuilt here
    from the raw CALCE cycles via the already-trained ica_encoder_mmd.pt
    (forward pass only, no training). Cached to disk since raw CALCE
    tensor parsing (~minutes) is the expensive step and identical on
    every re-run of this script; the cache is deleted at the end of this
    session, same convention as session 13's temporary CALCE tensor
    cache."""
    if _CALCE_FUSION_CACHE.exists():
        print(f"[domain-conformal] reusing cached CALCE fusion embeddings ({_CALCE_FUSION_CACHE.name})")
        return pd.read_csv(_CALCE_FUSION_CACHE)

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    all_X, all_bid, all_cyc = [], [], []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        all_X.append(X.astype(np.float32))
        all_bid += [cid] * len(idxs)
        all_cyc += list(idxs)
        print(f"[domain-conformal] CALCE/{cid}: {X.shape[0]} cycles")
    X_all = np.concatenate(all_X)
    X_all = apply_channel_norm(X_all, norm_stats)

    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder_mmd.pt"))
    encoder.eval()
    with torch.no_grad():
        emb = encoder.encode(torch.tensor(X_all[:, :, ICA_CHANNEL_SLICE])).numpy()

    out = pd.DataFrame({"battery_id": all_bid, "cycle_idx": all_cyc})
    for i in range(16):
        out[f"fusion_{i}"] = emb[:, i]
    out.to_csv(_CALCE_FUSION_CACHE, index=False)
    print(f"[domain-conformal] rebuilt & cached {len(out)} CALCE fusion embeddings")
    return out


def logistic_domain_weights(X_calib, X_target, clip=PROB_CLIP):
    """
    Lightweight density-ratio estimation via a logistic-regression domain
    classifier: label calibration points 0, target-domain points 1, fit
    LogisticRegression on the pooled+standardized features, then
    w(x) = p(target|x) / p(calib|x) (odds) - the standard probabilistic-
    classifier trick for estimating a covariate-shift density ratio
    (Bickel et al. 2009; used exactly this way for conformal by
    Tibshirani et al. 2019). Clipped predicted probabilities avoid a
    single near-certain classification producing an infinite/absurd
    weight. Features are standardized on the POOLED calib+target set
    first, since HI values and fusion-embedding values sit on very
    different scales and logistic regression's boundary is scale-
    sensitive.

    Returns (w_calib, w_target, auc) - auc is reported as a direct,
    quantitative diagnostic of how separable calib vs. target actually
    are (AUC~0.5 = indistinguishable/well-overlapped, AUC~1.0 = near-
    total separation, i.e. near-zero feature overlap - the exact
    degenerate regime flagged in the task).
    """
    X_pool = np.concatenate([X_calib, X_target])
    mean, std = X_pool.mean(axis=0), X_pool.std(axis=0) + 1e-8
    X_calib_z = (X_calib - mean) / std
    X_target_z = (X_target - mean) / std
    y_pool = np.concatenate([np.zeros(len(X_calib)), np.ones(len(X_target))])

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(np.concatenate([X_calib_z, X_target_z]), y_pool)
    p_pool = clf.predict_proba(np.concatenate([X_calib_z, X_target_z]))[:, 1]
    auc = roc_auc_score(y_pool, p_pool)

    p_calib = np.clip(clf.predict_proba(X_calib_z)[:, 1], *clip)
    p_target = np.clip(clf.predict_proba(X_target_z)[:, 1], *clip)
    w_calib = p_calib / (1 - p_calib)
    w_target = p_target / (1 - p_target)
    return w_calib, w_target, auc


def effective_sample_size(w):
    """Kish's effective sample size: (sum w)^2 / sum(w^2). Diagnoses how
    much the reweighting concentrates mass on a handful of calibration
    points - a small ESS relative to n means the weighted quantile below
    is effectively driven by very few calibration residuals, exactly the
    known instability mode of covariate-shift conformal methods under
    near-zero support overlap."""
    return float((w.sum() ** 2) / (w ** 2).sum())


def weighted_quantile_halfwidths(calib_resid, w_calib, w_test, alpha=ALPHA):
    """
    Tibshirani et al. 2019 weighted split-conformal quantile, vectorized
    over many test points sharing the same calibration set/weights but
    each with their OWN w_test (so each test point gets its own,
    genuinely per-input interval half-width, not one shared constant).

    For a single test point with weight w_t, the weighted empirical
    distribution over {r_1,...,r_n, +inf} has masses
    p_i = w_calib[i] / (sum(w_calib) + w_t), p_inf = w_t / (sum(w_calib) + w_t).
    The (1-alpha) quantile Q is the smallest r_i such that the
    cumulative mass over calibration residuals <= r_i reaches 1-alpha;
    if even ALL calibration mass combined can't reach 1-alpha (i.e. w_t
    is large enough that p_inf alone exceeds alpha), Q = +inf - a
    genuine, meaningful output meaning "this test point is too unlike
    calibration for this method to bound its error," not a bug.
    """
    order = np.argsort(calib_resid)
    sorted_resid = calib_resid[order]
    sorted_w = w_calib[order]
    cum_w = np.cumsum(sorted_w)  # unnormalized cumulative weight
    W = w_calib.sum()

    threshold = (1 - alpha) * (W + w_test)  # vector, one per test point
    idx = np.searchsorted(cum_w, threshold, side="left")
    halfwidths = np.where(idx < len(sorted_resid),
                           sorted_resid[np.clip(idx, 0, len(sorted_resid) - 1)],
                           np.inf)
    return halfwidths


def evaluate_group(name, calib_resid, X_calib, X_target, y_true_target, pred_target):
    w_calib, w_target, auc = logistic_domain_weights(X_calib, X_target)
    ess = effective_sample_size(w_calib)
    halfwidths = weighted_quantile_halfwidths(calib_resid, w_calib, w_target)

    finite = np.isfinite(halfwidths)
    frac_degenerate = float((~finite).mean())
    lo = pred_target - halfwidths
    hi = pred_target + halfwidths
    covered = (y_true_target >= lo) & (y_true_target <= hi)  # an infinite-width interval trivially covers - reported explicitly below, not hidden
    coverage = float(covered.mean())
    mean_width_finite = float(np.mean(2 * halfwidths[finite])) if finite.any() else float("nan")

    print(f"\n[domain-conformal] === {name} ===")
    print(f"[domain-conformal] domain-classifier AUC (calib vs. {name}): {auc:.4f} "
          f"({'near-chance, well-overlapped' if auc < 0.65 else 'high separability' if auc < 0.9 else 'NEAR-TOTAL separation - near-zero feature overlap'})")
    print(f"[domain-conformal] calibration effective sample size: {ess:.1f} of {len(calib_resid)} "
          f"({ess/len(calib_resid)*100:.1f}%)")
    print(f"[domain-conformal] degenerate (infinite-width) intervals: {frac_degenerate*100:.1f}% of {len(halfwidths)} points")
    print(f"[domain-conformal] mean interval width (finite only): {mean_width_finite:.3f}")
    print(f"[domain-conformal] empirical coverage (infinite intervals counted as covered, per their literal definition): {coverage:.3f}")

    return {
        "domain": name, "n_points": len(halfwidths), "auc": auc,
        "effective_sample_size": ess, "ess_pct": ess / len(calib_resid) * 100,
        "frac_degenerate_interval": frac_degenerate,
        "mean_width_finite_only": mean_width_finite,
        "min_width": float(halfwidths[finite].min()) if finite.any() else float("nan"),
        "max_width_finite": float(halfwidths[finite].max()) if finite.any() else float("nan"),
        "empirical_coverage": coverage, "target_coverage": 1 - ALPHA,
    }, halfwidths


def main():
    # feature_set diagnostic switch (not part of the original ask, added
    # after the first run's results demanded investigation - see
    # DEVELOPMENT_LOG.md session 19): "full" = 7 BFA HIs + 16-dim MMD
    # fusion embedding (the model's actual input space, the default);
    # "fusion_only" = just the 16-dim MMD-aligned embedding, to isolate
    # whether the raw (never MMD-aligned) HI features are what's driving
    # the unexpectedly high in-domain calib-vs-eval AUC.
    feature_set = sys.argv[1] if len(sys.argv) > 1 else "full"
    print(f"[domain-conformal] feature_set={feature_set}")

    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        BFA_SELECTED = [l.strip() for l in f if l.strip()]

    # --- NASA+MIT calib/eval halves: same split as sessions 5/13 ---
    ens = pd.read_csv(PRED_DIR / "ensemble_fusion_mmd_test_preds.csv")
    calib_ids, eval_ids = calib_eval_battery_split(ens["battery_id"].unique().tolist())
    print(f"[domain-conformal] calib batteries: {calib_ids}, eval batteries: {eval_ids}")

    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_nasa_mit = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])]
    train_medians = hi_nasa_mit[BFA_SELECTED].median(numeric_only=True)

    ens = pd.merge(ens, hi_nasa_mit[["dataset", "battery_id", "cycle_idx"] + BFA_SELECTED],
                    on=["dataset", "battery_id", "cycle_idx"], how="left")
    for col in BFA_SELECTED:
        ens[col] = ens[col].fillna(train_medians[col])

    fusion_cols = [f"fusion_{i}" for i in range(16)]
    feature_cols = fusion_cols if feature_set == "fusion_only" else BFA_SELECTED + fusion_cols

    calib_df = ens[ens["battery_id"].isin(calib_ids)].reset_index(drop=True)
    eval_df = ens[ens["battery_id"].isin(eval_ids)].reset_index(drop=True)
    print(f"[domain-conformal] NASA+MIT calib rows={len(calib_df)}, eval rows={len(eval_df)}")

    X_calib = calib_df[feature_cols].to_numpy(dtype=float)
    calib_resid = np.abs(calib_df["SOH"].to_numpy() - calib_df["pred_Stacking_Ridge_fusion"].to_numpy())

    # --- CALCE: HI columns cheaply from hi_table.parquet (already
    # computed), fusion embedding rebuilt (see build_calce_fusion_embeddings) ---
    calce_hi = hi_df[hi_df["dataset"] == "CALCE"][["battery_id", "cycle_idx"] + BFA_SELECTED].copy()
    for col in BFA_SELECTED:
        calce_hi[col] = calce_hi[col].fillna(train_medians[col])  # MATC/MATD 100% missing - same NASA+MIT-training-median imputation as sessions 5/13, never from CALCE itself
    calce_fusion = build_calce_fusion_embeddings()
    calce_preds = pd.read_csv(PRED_DIR / "calce_zero_retrain_mmd_preds.csv")

    calce_df = pd.merge(calce_hi, calce_fusion, on=["battery_id", "cycle_idx"], how="inner")
    calce_df = pd.merge(calce_df, calce_preds[["battery_id", "cycle_idx", "SOH", "pred_Stacking_Ridge_fusion_mmd"]],
                         on=["battery_id", "cycle_idx"], how="inner")
    print(f"[domain-conformal] CALCE rows: hi={len(calce_hi)}, fusion={len(calce_fusion)}, "
          f"preds={len(calce_preds)}, merged={len(calce_df)}")

    X_calce = calce_df[feature_cols].to_numpy(dtype=float)

    # --- (1) in-domain sanity check: calib vs. NASA+MIT eval half ---
    X_eval = eval_df[feature_cols].to_numpy(dtype=float)
    r_eval, hw_eval = evaluate_group(
        "NASA+MIT eval half (in-domain)", calib_resid, X_calib, X_eval,
        eval_df["SOH"].to_numpy(), eval_df["pred_Stacking_Ridge_fusion"].to_numpy(),
    )

    # --- (2) the actual test: calib vs. CALCE (out-of-domain) ---
    r_calce, hw_calce = evaluate_group(
        "CALCE (out-of-domain, zero-retrain, MMD-aligned)", calib_resid, X_calib, X_calce,
        calce_df["SOH"].to_numpy(), calce_df["pred_Stacking_Ridge_fusion_mmd"].to_numpy(),
    )

    print("\n[domain-conformal] === COMPARISON: original fixed-width vs. domain-shift-aware ===")
    print(f"[domain-conformal] ORIGINAL (session 13, MMD-recalibrated, fixed-width): "
          f"in-domain half-width=2.217 coverage=94.6%  |  CALCE half-width=2.217 (IDENTICAL) coverage=4.4%")
    print(f"[domain-conformal] NEW (domain-shift-aware, this session): "
          f"in-domain mean half-width={r_eval['mean_width_finite_only']/2:.3f} coverage={r_eval['empirical_coverage']*100:.1f}%  |  "
          f"CALCE mean half-width={r_calce['mean_width_finite_only']/2:.3f} coverage={r_calce['empirical_coverage']*100:.1f}%")
    print(f"[domain-conformal] widths now differ between domains: "
          f"{'YES' if abs(r_calce['mean_width_finite_only'] - r_eval['mean_width_finite_only']) > 0.01 else 'NO'} "
          f"(CALCE is {r_calce['mean_width_finite_only']/r_eval['mean_width_finite_only']:.1f}x wider on average, finite intervals only)")

    suffix = "" if feature_set == "full" else f"_{feature_set}"
    summary_df = pd.DataFrame([r_eval, r_calce])
    summary_df.to_csv(OUT_DIR / f"domain_shift_conformal_summary{suffix}.csv", index=False)

    per_point = pd.DataFrame({
        "domain": ["NASA+MIT_eval"] * len(hw_eval) + ["CALCE"] * len(hw_calce),
        "battery_id": list(eval_df["battery_id"]) + list(calce_df["battery_id"]),
        "cycle_idx": list(eval_df["cycle_idx"]) + list(calce_df["cycle_idx"]),
        "halfwidth": np.concatenate([hw_eval, hw_calce]),
    })
    per_point.to_csv(PRED_DIR / f"domain_shift_conformal_per_point{suffix}.csv", index=False)

    print(f"\n[domain-conformal] saved outputs/domain_shift_conformal_summary{suffix}.csv and "
          f"predictions/domain_shift_conformal_per_point{suffix}.csv")
    print("[domain-conformal] DONE")


if __name__ == "__main__":
    main()
