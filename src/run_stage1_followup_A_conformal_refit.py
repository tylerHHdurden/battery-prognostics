"""
Stage 1 follow-up, Part A: refit conformal calibration against the FINAL
adopted Stage 1 configuration (1.1's reformulated duration features +
1.5's monotone_constraints on cycle_idx, canonical 8-feature set, NOT
1.2's sample weighting - 1.2 was a real trade-off and was never
adopted).

STEP 1 ANSWER (confirmed by reading the actual code, not inference):
1.1's own reported CALCE coverage (9.5%) was computed against 1.1
ALONE - stage1_common.canonical_feature_cols(reformulated=True), no
cycle_idx, no monotone_constraints (see run_stage1_1_duration_features.py).
1.5's own script ALREADY happens to train and evaluate the true final
1.1+1.5 combined configuration (base_features = 1.1's reformulated set
+ cycle_idx, monotone_constraints active) and already reports a coverage
number for it (6.7%) - but as an incidental byproduct of item 1.5's own
ablation, not framed as "the final state's calibration." This script
retrains that EXACT same final configuration explicitly and
independently, as this follow-up's own dedicated, clearly-labeled
verification, and checks whether the result is reproducible (it must
be bit-for-bit identical, given fixed seeds/deterministic split-
conformal - confirming or refuting that 1.5's number was really this
final state's number, not a coincidence).

Every calce_coverage() call in this project's Stage 1 code already
fits split-conformal FRESH against whichever model was just trained
(confirmed by reading stage1_common.py directly) - there is no
"stale calibration" bug in the code itself. This script exists to
verify that directly, and to give the final-config coverage a clean,
dedicated, unambiguous report rather than leaving it as an incidental
number inside item 1.5's writeup.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge  # unused here, kept for parity w/ shared imports elsewhere
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, build_calce_merged, eval_calce, calce_coverage,
    fusion_cols, OUT_DIR, ALPHA,
)
from run_conformal import calib_eval_battery_split, split_conformal
from mapie.regression import CrossConformalRegressor


def main():
    print("[stage1-followA] === Building the FINAL adopted Stage 1 configuration (1.1 + 1.5) ===")
    base_features = canonical_feature_cols(reformulated=True)  # 1.1's reformulated set
    feature_cols = base_features + ["cycle_idx"]                # 1.5's added feature
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)

    corrs = merged[train_mask].groupby("battery_id").apply(
        lambda g: g["cycle_idx"].corr(g["SOH"]), include_groups=False)
    assert (corrs < 0).all(), "sign check failed - see 1.5's own script for why this must hold"
    n_fusion = len(fusion_cols())
    monotone = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)

    model, medians, cols = fit_xgb(merged, train_mask, feature_cols,
                                    xgb_extra_kwargs={"monotone_constraints": monotone})
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    calce_merged = build_calce_merged(hi_full)
    calce = eval_calce(model, medians, feature_cols, calce_merged)
    print(f"[stage1-followA] final-config in-domain: RMSE={indomain['rmse']:.4f} R2={indomain['r2']:.4f}")
    print(f"[stage1-followA] final-config CALCE:      RMSE={calce['rmse']:.4f} R2={calce['r2']:.4f}")

    # --- clean, explicit split-conformal refit against THIS model's own residuals ---
    cov = calce_coverage(indomain, calce)
    print(f"\n[stage1-followA] === Split-conformal, freshly refit against the FINAL (1.1+1.5) model ===")
    print(f"[stage1-followA] method={cov['method']} empirical_coverage={cov['empirical_coverage']:.4f} "
          f"avg_width={cov['avg_interval_width']:.4f} (n_calib={cov['n_calib']}, n_calce={cov['n_calce']})")

    print(f"\n[stage1-followA] === reproducibility check vs. item 1.5's own incidentally-reported number ===")
    print(f"[stage1-followA] item 1.5's own reported coverage: 0.0673 - this run: {cov['empirical_coverage']:.4f} "
          f"({'MATCHES (confirms 1.5 already reported the true final-config number)' if abs(cov['empirical_coverage']-0.0673) < 1e-6 else 'DIFFERS - investigate before trusting either number'})")

    # --- comparison table ---
    print(f"\n[stage1-followA] === COMPARISON ===")
    print(f"[stage1-followA] (a) session 5 (32-battery, ORIGINAL leaked 7-feature set, FULL 4-branch "
          f"ensemble - NOT XGBoost-fusion alone): CALCE coverage = 6.1%")
    print(f"[stage1-followA] (a) session 33 (204-battery expanded, FULL 4-branch ensemble): CALCE coverage = 7.4%")
    print(f"[stage1-followA]     CAVEAT stated plainly: (a) uses a DIFFERENT model architecture (the full "
          f"stacking ensemble, not XGBoost-fusion alone) AND a different/leaked feature set - not a clean "
          f"apples-to-apples number against anything in this stage. Included because explicitly requested.")
    print(f"[stage1-followA] (a') Stage-1's OWN canonical-raw XGBoost-fusion-only baseline "
          f"(no 1.1, no 1.5 - the true apples-to-apples 'before'): CALCE coverage = 16.6%")
    print(f"[stage1-followA] (b) 1.1 ALONE's own reported coverage (reformulated features, no "
          f"monotone_constraints): CALCE coverage = 9.5%")
    print(f"[stage1-followA] FINAL (1.1+1.5 combined, this run): CALCE coverage = {cov['empirical_coverage']*100:.1f}%")

    if cov['empirical_coverage'] >= 0.166 - 0.01:
        outcome = ("(a) RECOVERS - coverage is back in line with (or better than) the pre-Stage-1 "
                   "XGBoost-fusion baseline (16.6%) once cleanly refit against the true final model.")
    else:
        outcome = ("(b) STILL WORSE than baseline even after a clean, verified-fresh refit - this is a "
                   "REAL, separate calibration cost of the Stage 1 accuracy gains, NOT a staleness "
                   "artifact. Coverage gets progressively WORSE as more Stage 1 changes are layered on "
                   f"(16.6% baseline -> 9.5% after 1.1 alone -> {cov['empirical_coverage']*100:.1f}% after "
                   "1.1+1.5 combined) - the same direction as the accuracy improvement, not opposite it.")
    print(f"\n[stage1-followA] OUTCOME: {outcome}")

    # --- optional/secondary: Jackknife+/CV+ as a drop-in replacement, same final model ---
    print(f"\n[stage1-followA] === OPTIONAL: Jackknife+/CV+ on this same final (1.1+1.5) XGBoost-fusion model ===")
    calib_ids, eval_ids = calib_eval_battery_split(sorted(set(indomain["battery_id"].tolist())))
    print(f"[stage1-followA] calibration battery pool (from NASA+MIT test half): {calib_ids} ({len(calib_ids)} batteries)")

    calib_mask_full = merged["battery_id"].isin(calib_ids).to_numpy() & test_mask
    X_all = merged[cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X_all))
    X_all[inds] = np.take(medians, inds[1])
    X_calib_cv = X_all[calib_mask_full]
    y_calib_cv = merged.loc[calib_mask_full, "SOH"].to_numpy()
    groups_cv = merged.loc[calib_mask_full, "battery_id"].to_numpy()
    n_calib_batteries = len(set(groups_cv))
    print(f"[stage1-followA] CV+ calibration pool: {len(X_calib_cv)} rows, {n_calib_batteries} batteries "
          f"(XGBoost refits are cheap - genuine leave-one-battery-out Jackknife+ is feasible here, "
          f"unlike the deep-model case in item 1.6)")

    X_calce_cv = calce_merged[cols].to_numpy(dtype=float, copy=True)
    for j in range(len(cols)):
        nan_mask = np.isnan(X_calce_cv[:, j])
        if nan_mask.any():
            X_calce_cv[nan_mask, j] = medians[j]
    y_calce_true = calce_merged["SOH"].to_numpy()

    base_est = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             n_jobs=-1, reg_lambda=1.0, monotone_constraints=monotone)
    cv_loo = GroupKFold(n_splits=n_calib_batteries)
    mapie_jk = CrossConformalRegressor(estimator=base_est, confidence_level=1 - ALPHA, method="plus", cv=cv_loo)
    mapie_jk.fit_conformalize(X_calib_cv, y_calib_cv, groups=groups_cv)
    pred_jk, interval_jk = mapie_jk.predict_interval(X_calce_cv)
    lo_jk, hi_jk = interval_jk[:, 0, 0], interval_jk[:, 1, 0]
    covered_jk = (y_calce_true >= lo_jk) & (y_calce_true <= hi_jk)
    coverage_jk = float(covered_jk.mean())
    width_jk = float(np.mean(hi_jk - lo_jk))
    print(f"[stage1-followA] Jackknife+ (K={n_calib_batteries}, leave-one-battery-out) CALCE coverage: "
          f"{coverage_jk:.4f} (avg_width={width_jk:.3f})")
    print(f"[stage1-followA] vs. plain split-conformal on the same final model: {cov['empirical_coverage']:.4f}")
    jk_verdict = ("Jackknife+ is a meaningful drop-in improvement" if coverage_jk > cov['empirical_coverage'] + 0.01
                  else "Jackknife+ does NOT meaningfully improve CALCE coverage over plain split-conformal here"
                  if coverage_jk < cov['empirical_coverage'] - 0.01 else
                  "Jackknife+ and plain split-conformal give essentially the same CALCE coverage here")
    print(f"[stage1-followA] VERDICT: {jk_verdict}")

    pd.DataFrame([
        {"method": "session5_ensemble_32batt_ORIGINAL", "coverage": 0.061, "note": "different model architecture (full ensemble), reported for reference only"},
        {"method": "session33_ensemble_204batt", "coverage": 0.074, "note": "different model architecture (full ensemble), reported for reference only"},
        {"method": "stage1_canonical_raw_baseline_xgbfusion", "coverage": 0.166, "note": "apples-to-apples XGBoost-fusion-only pre-Stage-1 baseline"},
        {"method": "1.1_alone", "coverage": 0.095, "note": "reformulated features only, no monotone_constraints"},
        {"method": "1.1+1.5_final_split_conformal", "coverage": cov["empirical_coverage"], "note": "final adopted Stage 1 config, this run"},
        {"method": "1.1+1.5_final_jackknife_plus", "coverage": coverage_jk, "note": "optional CV+/Jackknife+ drop-in, same final model"},
    ]).to_csv(OUT_DIR / "stage1_followup_A_conformal_comparison.csv", index=False)
    print("\n[stage1-followA] saved outputs/stage1_followup_A_conformal_comparison.csv")
    print("[stage1-followA] DONE")


if __name__ == "__main__":
    main()
