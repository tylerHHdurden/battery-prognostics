"""
Stage 3, Item 3.1: KMM-CP (attempt five at CALCE conformal coverage).

CONFIGURATION: Stage 1's 1.1+1.5 canonical XGBoost-fusion model
(reformulated duration features + monotone_constraints on cycle_idx),
original 32-battery pool - the same model/pool used throughout Stage 1
and its follow-ups, so this is directly comparable to every number in
the comparison table below.

Calibration set: the SAME calib-half of the 6 NASA+MIT test batteries
used throughout this project's conformal work (calib_eval_battery_split,
session 6 onward). Target: CALCE zero-retrain (all 3 cells, never
trained on).

Kernel-matching feature space: the model's own 8 canonical (1.1-
reformulated) HI features + cycle_idx + 16-dim fusion embedding (25-dim
total, standardized) - the exact feature space the point predictor
itself consumes, matching this project's own precedent (session 19's
domain classifier used the same "7 BFA HIs + 16-dim fusion" space).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, build_calce_merged, eval_calce, fusion_cols, OUT_DIR,
)
from run_conformal import calib_eval_battery_split
from kmm_utils import fit_kmm_weights, selective_bounds, weighted_conformal_interval, median_heuristic_gamma

ALPHA = 0.1


def build_feature_matrix(df, cols, medians):
    X = df[cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X))
    if len(inds[0]):
        X[inds] = np.take(medians, inds[1])
    return X


def main():
    print("[kmm-cp] === Building the Stage 1.1+1.5 canonical model ===")
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    base_features = canonical_feature_cols(reformulated=True)
    feature_cols = base_features + ["cycle_idx"]
    fcols = fusion_cols()
    n_fusion = len(fcols)
    monotone = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)
    model, medians, cols = fit_xgb(merged, train_mask, feature_cols, xgb_extra_kwargs={"monotone_constraints": monotone})
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    calce_merged = build_calce_merged(hi_full)
    calce = eval_calce(model, medians, feature_cols, calce_merged)
    print(f"[kmm-cp] in-domain R2={indomain['r2']:.4f} | CALCE R2={calce['r2']:.4f} RMSE={calce['rmse']:.4f}")

    # --- calibration/target feature matrices, in the model's own input space ---
    calib_ids, eval_ids = calib_eval_battery_split(sorted(set(indomain["battery_id"].tolist())))
    calib_mask_full = merged["battery_id"].isin(calib_ids).to_numpy() & test_mask
    print(f"[kmm-cp] calibration batteries: {calib_ids}")

    X_calib_raw = build_feature_matrix(merged.loc[calib_mask_full], cols, medians)
    X_calce_raw = build_feature_matrix(calce_merged, cols, medians)

    feat_mean, feat_std = X_calib_raw.mean(axis=0), X_calib_raw.std(axis=0) + 1e-8
    X_calib = (X_calib_raw - feat_mean) / feat_std
    X_calce = (X_calce_raw - feat_mean) / feat_std

    calib_pred = indomain["pred"][np.isin(indomain["battery_id"], calib_ids)]
    calib_y = indomain["y_true"][np.isin(indomain["battery_id"], calib_ids)]
    calib_resid = np.abs(calib_y - calib_pred)
    calce_pred = calce["pred"]
    calce_y = calce["y_true"]

    print(f"[kmm-cp] calibration set: {len(X_calib)} rows | CALCE target set: {len(X_calce)} rows")

    # --- baseline: plain (unweighted) split-conformal, for reference ---
    from run_conformal import split_conformal
    _, lo_plain, hi_plain, method_plain = split_conformal(calib_pred, calib_y, calce_pred, ALPHA)
    cov_plain = float(((calce_y >= lo_plain) & (calce_y <= hi_plain)).mean())
    width_plain = float(np.mean(hi_plain - lo_plain))
    print(f"[kmm-cp] plain split-conformal (unweighted): coverage={cov_plain:.4f} width={width_plain:.4f}")

    # --- Step 1: standard KMM ---
    print("\n[kmm-cp] === Standard KMM ===")
    gamma = median_heuristic_gamma(np.vstack([X_calib, X_calce]))
    B = 4.0
    w_kmm, diag_kmm = fit_kmm_weights(X_calib, X_calce, B=B, gamma=gamma)
    print(f"[kmm-cp] KMM diagnostics: {diag_kmm}")

    lo_kmm, hi_kmm, q_kmm = weighted_conformal_interval(calib_resid, w_kmm, calce_pred, ALPHA)
    cov_kmm = float(((calce_y >= lo_kmm) & (calce_y <= hi_kmm)).mean())
    width_kmm = float(np.mean(hi_kmm - lo_kmm)) if np.isfinite(hi_kmm - lo_kmm).all() else float("inf")
    print(f"[kmm-cp] standard KMM-CP: coverage={cov_kmm:.4f} width={width_kmm:.4f} (q={q_kmm:.4f})")

    # --- Step 2: is standard KMM unstable? (per instruction: implement Selective KMM if so) ---
    unstable = diag_kmm["ess_frac"] < 0.20 or diag_kmm["frac_near_zero"] > 0.80
    print(f"\n[kmm-cp] standard KMM stability check: ESS={diag_kmm['effective_sample_size']:.1f} "
          f"({diag_kmm['ess_frac']*100:.1f}% of n={diag_kmm['n_calib']}), "
          f"{diag_kmm['frac_near_zero']*100:.1f}% of weights near-zero -> "
          f"{'UNSTABLE, implementing Selective KMM' if unstable else 'stable enough, Selective KMM not required'}")

    results = [
        {"method": "plain_split_conformal", "coverage": cov_plain, "avg_width": width_plain, "ess_frac": 1.0},
        {"method": "standard_KMM_CP", "coverage": cov_kmm, "avg_width": width_kmm, "ess_frac": diag_kmm["ess_frac"]},
    ]

    if unstable:
        print("\n[kmm-cp] === Selective KMM (per-point bounds from target-support relevance) ===")
        bounds = selective_bounds(X_calib, X_calce, gamma, B=B, percentile=50)
        w_sel, diag_sel = fit_kmm_weights(X_calib, X_calce, B=B, gamma=gamma, per_point_bound=bounds)
        print(f"[kmm-cp] Selective KMM diagnostics: {diag_sel}")
        lo_sel, hi_sel, q_sel = weighted_conformal_interval(calib_resid, w_sel, calce_pred, ALPHA)
        cov_sel = float(((calce_y >= lo_sel) & (calce_y <= hi_sel)).mean())
        width_sel = float(np.mean(hi_sel - lo_sel)) if np.isfinite(hi_sel - lo_sel).all() else float("inf")
        print(f"[kmm-cp] Selective KMM-CP: coverage={cov_sel:.4f} width={width_sel:.4f} (q={q_sel:.4f})")
        results.append({"method": "selective_KMM_CP", "coverage": cov_sel, "avg_width": width_sel, "ess_frac": diag_sel["ess_frac"]})

    # --- comparison table against all 4 prior attempts ---
    print("\n[kmm-cp] === COMPARISON: all 5 CALCE-coverage attempts ===")
    print("[kmm-cp] (1) MMD (session 13): fixed point-accuracy, coverage stayed collapsed (6.1%->4.4%, got WORSE)")
    print("[kmm-cp] (2) Weighted conformal via logistic domain classifier (session 19): CALCE coverage "
          "NEVER improved (fusion-only config: 4.4%, statistically identical to the unweighted baseline; "
          "full-feature config: nominally 100% but VACUOUS - every CALCE point got an infinite-width "
          "interval from AUC=1.0 total separation). A genuine SIDE EFFECT also broke the in-domain sanity "
          "check (94.6%->43.6% under the full-feature config) - the classifier-odds approach's instability "
          "under limited overlap, exactly the structural failure mode this item's own context describes.")
    print("[kmm-cp] (3) Dataset expansion + input-adaptive CP (session 35 Part 4): 7.4% -> 19.3% (Normalized CP) / 21.3% (CQR)")
    print("[kmm-cp] (4) Jackknife+ (Stage 1 follow-up, configuration-specific): 6.7% -> 37.1% on the 1.1+1.5 model specifically")
    print(f"[kmm-cp] (5) THIS ITEM - KMM-CP: plain={cov_plain*100:.1f}% -> "
          f"standard KMM={cov_kmm*100:.1f}%" + (f" -> Selective KMM={cov_sel*100:.1f}%" if unstable else ""))

    final_cov = results[-1]["coverage"]
    final_width = results[-1]["avg_width"]
    if final_cov >= 0.85:
        verdict = "(a) BREAKTHROUGH - coverage closes to near 90%"
    elif final_cov > 0.371:  # beats Jackknife+'s best prior number
        verdict = "(b) real, meaningful improvement beyond all 4 prior attempts, still short of 90%"
    else:
        verdict = "(c) no meaningful improvement beyond prior attempts, or a new KMM-specific failure mode"
    print(f"\n[kmm-cp] VERDICT: {verdict}")
    print(f"[kmm-cp] final coverage={final_cov*100:.1f}%, width={final_width:.3f} "
          f"(vs. plain width={width_plain:.3f} - {'WIDER' if final_width>width_plain else 'narrower or comparable'})")

    pd.DataFrame(results).to_csv(OUT_DIR / "stage3_1_kmm_cp_results.csv", index=False)
    print("\n[kmm-cp] saved outputs/stage3_1_kmm_cp_results.csv")
    print("[kmm-cp] DONE")


if __name__ == "__main__":
    main()
