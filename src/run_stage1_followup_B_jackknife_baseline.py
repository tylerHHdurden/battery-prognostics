"""
Stage 1 follow-up, Part B (second use of that letter - this is the
"is the Jackknife+ jump general or 1.1/1.5-specific" check, immediately
following the conformal-refit and B0044 follow-up): reuses
run_stage1_followup_A_conformal_refit.py's Jackknife+ code path
UNCHANGED in mechanism, pointed at the PRE-STAGE-1 BASELINE model
(canonical 8-feature set, RAW/unreformulated, no monotone_constraints,
no sample weighting) instead of the final 1.1+1.5 model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    CANONICAL_FEATURES, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, build_calce_merged, eval_calce, calce_coverage,
    OUT_DIR, ALPHA,
)
from run_conformal import calib_eval_battery_split
from mapie.regression import CrossConformalRegressor


def main():
    print("[stage1-followB] === Building the PRE-STAGE-1 BASELINE configuration (raw canonical features) ===")
    feature_cols = list(CANONICAL_FEATURES)  # raw, unreformulated - no 1.1, no 1.5
    merged, hi_full = load_nasa_mit_pool(reformulated=False)
    train_mask, test_mask, split = battery_split_masks(merged)

    model, medians, cols = fit_xgb(merged, train_mask, feature_cols)
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    calce_merged = build_calce_merged(hi_full)
    calce = eval_calce(model, medians, feature_cols, calce_merged)
    print(f"[stage1-followB] baseline in-domain: RMSE={indomain['rmse']:.4f} R2={indomain['r2']:.4f}")
    print(f"[stage1-followB] baseline CALCE:      RMSE={calce['rmse']:.4f} R2={calce['r2']:.4f}")

    # --- verification: must reproduce the existing 16.6% number before proceeding ---
    cov = calce_coverage(indomain, calce)
    print(f"\n[stage1-followB] === VERIFICATION: split-conformal on this baseline ===")
    print(f"[stage1-followB] this run: {cov['empirical_coverage']:.4f} | "
          f"previously reported (outputs/stage1_canonical_baseline.csv): 0.16626997619857192")
    matches = abs(cov['empirical_coverage'] - 0.16626997619857192) < 1e-9
    print(f"[stage1-followB] reproduces bit-for-bit: {matches}")
    if not matches:
        print("[stage1-followB] STOPPING - baseline does not match the previously reported number, "
              "the wrong model/config would be being compared. NOT proceeding to Jackknife+.")
        return

    # --- Jackknife+, identical mechanism/settings to the A follow-up, same calibration battery pool ---
    print(f"\n[stage1-followB] === Jackknife+ (K=3, leave-one-battery-out) on this SAME baseline model ===")
    calib_ids, eval_ids = calib_eval_battery_split(sorted(set(indomain["battery_id"].tolist())))
    print(f"[stage1-followB] calibration battery pool: {calib_ids} ({len(calib_ids)} batteries) - "
          f"identical pool used in the A follow-up's 1.1+1.5 Jackknife+ run")

    calib_mask_full = merged["battery_id"].isin(calib_ids).to_numpy() & test_mask
    X_all = merged[cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X_all))
    X_all[inds] = np.take(medians, inds[1])
    X_calib_cv = X_all[calib_mask_full]
    y_calib_cv = merged.loc[calib_mask_full, "SOH"].to_numpy()
    groups_cv = merged.loc[calib_mask_full, "battery_id"].to_numpy()
    n_calib_batteries = len(set(groups_cv))
    print(f"[stage1-followB] CV+ calibration pool: {len(X_calib_cv)} rows, {n_calib_batteries} batteries")

    X_calce_cv = calce_merged[cols].to_numpy(dtype=float, copy=True)
    for j in range(len(cols)):
        nan_mask = np.isnan(X_calce_cv[:, j])
        if nan_mask.any():
            X_calce_cv[nan_mask, j] = medians[j]
    y_calce_true = calce_merged["SOH"].to_numpy()

    base_est = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             n_jobs=-1, reg_lambda=1.0)  # NOTE: no monotone_constraints - this is the baseline
    cv_loo = GroupKFold(n_splits=n_calib_batteries)
    mapie_jk = CrossConformalRegressor(estimator=base_est, confidence_level=1 - ALPHA, method="plus", cv=cv_loo)
    mapie_jk.fit_conformalize(X_calib_cv, y_calib_cv, groups=groups_cv)
    pred_jk, interval_jk = mapie_jk.predict_interval(X_calce_cv)
    lo_jk, hi_jk = interval_jk[:, 0, 0], interval_jk[:, 1, 0]
    covered_jk = (y_calce_true >= lo_jk) & (y_calce_true <= hi_jk)
    coverage_jk = float(covered_jk.mean())
    width_jk = float(np.mean(hi_jk - lo_jk))
    print(f"[stage1-followB] Jackknife+ (baseline model) CALCE coverage: {coverage_jk:.4f} (avg_width={width_jk:.3f})")

    print(f"\n[stage1-followB] === SIDE-BY-SIDE ===")
    print(f"[stage1-followB] | model config | split-conformal | Jackknife+ |")
    print(f"[stage1-followB] | pre-Stage-1 baseline | {cov['empirical_coverage']*100:.1f}% (width={cov['avg_interval_width']:.2f}) | "
          f"{coverage_jk*100:.1f}% (width={width_jk:.2f}) |")
    print(f"[stage1-followB] | 1.1+1.5 final (from prior follow-up) | 6.7% (width=2.33) | 37.1% (width=11.89) |")

    baseline_jump = coverage_jk - cov['empirical_coverage']
    final_jump = 0.37062223733424005 - 0.06732403944236655
    print(f"\n[stage1-followB] baseline jump (split-conformal -> Jackknife+): {baseline_jump*100:+.1f}pp")
    print(f"[stage1-followB] 1.1+1.5 final jump (from prior follow-up): {final_jump*100:+.1f}pp")

    if baseline_jump > 0.15:  # a "comparably large" jump threshold, generous given final's +30.3pp
        outcome = ("(a) GENERAL property - the baseline ALSO shows a large Jackknife+ jump, broadly "
                   "comparable to the 1.1+1.5 case. This is a general calibration-mechanism improvement "
                   "on this pipeline/dataset, not specific to 1.1/1.5 - a high-priority, well-evidenced "
                   "lead for Stage 3, potentially worth prioritizing above KMM-CP.")
    else:
        outcome = ("(b) NOT general - the baseline does NOT show a comparably large jump. The 37.1% "
                   "result on the 1.1+1.5 model is specific to that configuration (interacting with "
                   "the reformulated features and/or the monotone constraint), not a general property "
                   "of switching calibration mechanisms alone. The Jackknife+ finding is narrower/more "
                   "conditional than it first appeared - still real, just not a blanket calibration fix.")
    print(f"\n[stage1-followB] OUTCOME: {outcome}")

    width_note = ("wider" if width_jk > cov['avg_interval_width'] else "narrower")
    print(f"\n[stage1-followB] WIDTH CHECK: Jackknife+ intervals are {width_note} than split-conformal's "
          f"on the baseline too ({cov['avg_interval_width']:.2f} -> {width_jk:.2f}) - "
          f"{'consistent with the 1.1+1.5 case (also wider, 2.33->11.89)' if width_jk > cov['avg_interval_width'] else 'DIFFERENT from the 1.1+1.5 case'}.")

    pd.DataFrame([
        {"config": "pre-Stage-1 baseline", "method": "split-conformal", "coverage": cov["empirical_coverage"], "avg_width": cov["avg_interval_width"]},
        {"config": "pre-Stage-1 baseline", "method": "jackknife_plus", "coverage": coverage_jk, "avg_width": width_jk},
        {"config": "1.1+1.5 final", "method": "split-conformal", "coverage": 0.06732403944236655, "avg_width": 2.3315267577021075},
        {"config": "1.1+1.5 final", "method": "jackknife_plus", "coverage": 0.37062223733424005, "avg_width": 11.890},
    ]).to_csv(OUT_DIR / "stage1_followup_B_jackknife_baseline_comparison.csv", index=False)
    print("\n[stage1-followB] saved outputs/stage1_followup_B_jackknife_baseline_comparison.csv")
    print("[stage1-followB] DONE")


if __name__ == "__main__":
    main()
