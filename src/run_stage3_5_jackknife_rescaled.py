"""
Stage 3, Item 3.5: Jackknife+ (CV+) with locally rescaled conformal
scores. Run AFTER 3.1 (this script reads/prints 3.1's KMM-CP result
directly for the final comparison), per the task's own explicit
ordering instruction, so 3.5's contribution is separable from KMM-CP's.

CONFIGURATION: Stage 1's 1.1+1.5 canonical XGBoost-fusion model
(reformulated duration features + monotone_constraints), original
32-battery pool - identical to every other Stage 3 item and to the
plain Jackknife+/CV+ result this compares against (Stage 1 follow-up:
37.1% coverage, width=11.89, on this exact model).

MECHANISM - manual CV+ implementation (Barber, Candes, Ramdas,
Tibshirani 2021), K=n_calib_batteries folds (same convention as the
existing plain Jackknife+/CV+ script - "Jackknife+" in this project's
own naming means CV+ with K=n_calibration_batteries groups, not true
per-point LOO). VERIFIED FIRST, before trusting the rescaled version:
an UNSCALED (sigma_hat=1 for every point) run of this manual
implementation is checked against the existing mapie-based result
(coverage=0.3706, width=11.890) before the rescaled version is trusted -
per this project's "verify before trusting" standard, since this is a
from-scratch reimplementation of CV+'s quantile-aggregation formula
(mapie's CrossConformalRegressor does not support per-point rescaling
natively, so it cannot be reused directly for this item).

RESCALING - local difficulty estimate sigma_hat(x), fit using ONLY
training-set information (per instruction: "not a separate held-out
sigma(x) model requiring its own calibration split, unlike Normalized
CP"): a SEPARATE 5-fold GroupKFold split of the 26 TRAINING batteries
(disjoint from the calibration/eval battery split used for CV+ itself,
and never touching CALCE) produces genuine out-of-fold |residual|
values across the training pool; a second XGBRegressor is fit on
(X_train, oof_abs_residual) to predict local difficulty sigma_hat(x)
anywhere. This never spends calibration-set data building sigma_hat -
it's a purely training-domain construction, applied afterward to both
the calibration and CALCE-target rows to rescale each nonconformity
score: R_i -> R_i / sigma_hat(x_i), and the interval is projected back
via sigma_hat(x_target) at prediction time - the standard normalized-
conformal generalization, applied here to CV+'s residuals rather than
split-conformal's (session 35's own convention, reused).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, build_calce_merged, eval_calce, fusion_cols, OUT_DIR,
)
from run_conformal import calib_eval_battery_split

ALPHA = 0.1
SIGMA_FLOOR = 0.3  # avoid division blowup for near-zero-residual training points


def cv_plus_interval(fold_preds_target, fold_id_of_calib, R_calib, sigma_calib, sigma_target, alpha):
    """
    fold_preds_target: dict fold_idx -> f_{-fold}(x_target) array (len=n_target)
    fold_id_of_calib:  array (len=n_calib) giving each calib point's fold id
    R_calib:           array (len=n_calib) raw |y_i - f_{-fold(i)}(x_i)| residuals
    sigma_calib:       array (len=n_calib) sigma_hat(x_i) for each calib point (1.0 = unscaled)
    sigma_target:      array (len=n_target) sigma_hat(x) for each target point (1.0 = unscaled)
    Returns lo, hi arrays (len=n_target), per Barber et al. 2021's CV+ formula.
    """
    n = len(R_calib)
    n_target = len(sigma_target)
    scaled_R = R_calib / np.maximum(sigma_calib, SIGMA_FLOOR)  # normalized nonconformity scores

    lo = np.empty(n_target)
    hi = np.empty(n_target)
    lo_idx = int(np.floor(alpha * (n + 1))) - 1  # 0-indexed
    hi_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
    hi_idx = min(hi_idx, n - 1)

    for t in range(n_target):
        preds_for_calib_folds = np.array([fold_preds_target[fold_id_of_calib[i]][t] for i in range(n)])
        lower_vals = preds_for_calib_folds - scaled_R * sigma_target[t]
        upper_vals = preds_for_calib_folds + scaled_R * sigma_target[t]
        lower_sorted = np.sort(lower_vals)
        upper_sorted = np.sort(upper_vals)
        lo[t] = lower_sorted[lo_idx] if lo_idx >= 0 else -np.inf
        hi[t] = upper_sorted[hi_idx] if hi_idx < n else np.inf
    return lo, hi


def main():
    print("[stage3.5] === Building the Stage 1.1+1.5 canonical model ===")
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    base_features = canonical_feature_cols(reformulated=True)
    feature_cols = base_features + ["cycle_idx"]
    fcols = fusion_cols()
    monotone = tuple([0] * len(base_features) + [-1] + [0] * len(fcols))
    model, medians, cols = fit_xgb(merged, train_mask, feature_cols, xgb_extra_kwargs={"monotone_constraints": monotone})
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    calce_merged = build_calce_merged(hi_full)
    calce = eval_calce(model, medians, feature_cols, calce_merged)
    print(f"[stage3.5] in-domain R2={indomain['r2']:.4f} | CALCE R2={calce['r2']:.4f}")

    calib_ids, eval_ids = calib_eval_battery_split(sorted(set(indomain["battery_id"].tolist())))
    print(f"[stage3.5] calibration batteries: {calib_ids}")
    calib_mask_full = merged["battery_id"].isin(calib_ids).to_numpy() & test_mask

    def build_X(df):
        X = df[cols].to_numpy(dtype=float, copy=True)
        inds = np.where(np.isnan(X))
        if len(inds[0]):
            X[inds] = np.take(medians, inds[1])
        return X

    X_calib = build_X(merged.loc[calib_mask_full])
    y_calib = merged.loc[calib_mask_full, "SOH"].to_numpy()
    groups_calib = merged.loc[calib_mask_full, "battery_id"].to_numpy()
    X_calce = build_X(calce_merged)
    y_calce = calce_merged["SOH"].to_numpy()
    n_calib_batteries = len(set(groups_calib))
    print(f"[stage3.5] CV+ calibration pool: {len(X_calib)} rows, {n_calib_batteries} battery-groups")

    # --- fold assignment + per-fold model, exactly mirroring the existing Jackknife+/CV+ convention ---
    gkf = GroupKFold(n_splits=n_calib_batteries)
    fold_id_of_calib = np.zeros(len(X_calib), dtype=int)
    R_calib = np.zeros(len(X_calib))
    fold_preds_calce = {}
    for fold_idx, (fit_idx, held_idx) in enumerate(gkf.split(X_calib, groups=groups_calib)):
        est = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            n_jobs=-1, reg_lambda=1.0, monotone_constraints=monotone)
        est.fit(X_calib[fit_idx], y_calib[fit_idx])
        fold_id_of_calib[held_idx] = fold_idx
        pred_held = est.predict(X_calib[held_idx])
        R_calib[held_idx] = np.abs(y_calib[held_idx] - pred_held)
        fold_preds_calce[fold_idx] = est.predict(X_calce)
        print(f"[stage3.5] fold {fold_idx}: held-out battery/ies "
              f"{sorted(set(groups_calib[held_idx]))}, {len(held_idx)} rows")

    # --- Step 1: VERIFY unscaled CV+ reproduces the known plain Jackknife+/CV+ result ---
    print("\n[stage3.5] === Step 1: verify unscaled CV+ (this manual implementation) ===")
    sigma_ones_calib = np.ones(len(X_calib))
    sigma_ones_target = np.ones(len(X_calce))
    lo_plain, hi_plain = cv_plus_interval(fold_preds_calce, fold_id_of_calib, R_calib,
                                           sigma_ones_calib, sigma_ones_target, ALPHA)
    cov_plain = float(((y_calce >= lo_plain) & (y_calce <= hi_plain)).mean())
    width_plain = float(np.mean(hi_plain - lo_plain))
    print(f"[stage3.5] manual unscaled CV+: coverage={cov_plain:.4f} width={width_plain:.4f}")
    print(f"[stage3.5] existing mapie-based Jackknife+/CV+ (Stage 1 follow-up): coverage=0.3706 width=11.890")
    close_enough = abs(cov_plain - 0.3706) < 0.05
    print(f"[stage3.5] reasonably reproduces the known result (within 5pp): {close_enough}")
    if not close_enough:
        print("[stage3.5] WARNING: manual CV+ implementation does not reproduce the known baseline "
              "closely enough - proceeding anyway but flagging this discrepancy explicitly rather "
              "than silently trusting the rescaled result below.")

    # --- Step 2: fit sigma_hat(x) using ONLY training-set information ---
    print("\n[stage3.5] === Step 2: fitting local difficulty sigma_hat(x) from TRAINING data only ===")
    X_train_all = build_X(merged.loc[train_mask])
    y_train_all = merged.loc[train_mask, "SOH"].to_numpy()
    groups_train_all = merged.loc[train_mask, "battery_id"].to_numpy()
    unique_train_batteries = sorted(set(groups_train_all))
    n_sigma_folds = min(5, len(unique_train_batteries))
    gkf_sigma = GroupKFold(n_splits=n_sigma_folds)
    oof_abs_resid = np.zeros(len(X_train_all))
    for fold_idx, (fit_idx, held_idx) in enumerate(gkf_sigma.split(X_train_all, groups=groups_train_all)):
        est = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        est.fit(X_train_all[fit_idx], y_train_all[fit_idx])
        oof_abs_resid[held_idx] = np.abs(y_train_all[held_idx] - est.predict(X_train_all[held_idx]))
    print(f"[stage3.5] out-of-fold training |residual|: mean={oof_abs_resid.mean():.4f} "
          f"std={oof_abs_resid.std():.4f} (from {n_sigma_folds}-fold GroupKFold over the TRAINING pool only)")

    sigma_model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    sigma_model.fit(X_train_all, np.log1p(oof_abs_resid))  # log-space to keep sigma_hat positive
    sigma_calib = np.expm1(np.clip(sigma_model.predict(X_calib), -5, 5))
    sigma_target = np.expm1(np.clip(sigma_model.predict(X_calce), -5, 5))
    print(f"[stage3.5] sigma_hat(calibration set): mean={sigma_calib.mean():.4f} "
          f"[{sigma_calib.min():.4f}, {sigma_calib.max():.4f}]")
    print(f"[stage3.5] sigma_hat(CALCE target set): mean={sigma_target.mean():.4f} "
          f"[{sigma_target.min():.4f}, {sigma_target.max():.4f}]")
    print(f"[stage3.5] CALCE/calibration sigma_hat ratio (mean): {sigma_target.mean()/sigma_calib.mean():.3f} "
          f"(>1 means CALCE is flagged as harder/more out-of-distribution, consistent with domain shift)")

    # --- Step 3: rescaled CV+ ---
    print("\n[stage3.5] === Step 3: rescaled Jackknife+/CV+ ===")
    lo_resc, hi_resc = cv_plus_interval(fold_preds_calce, fold_id_of_calib, R_calib,
                                         sigma_calib, sigma_target, ALPHA)
    cov_resc = float(((y_calce >= lo_resc) & (y_calce <= hi_resc)).mean())
    width_resc = float(np.mean(np.clip(hi_resc - lo_resc, -1e6, 1e6)))
    print(f"[stage3.5] rescaled CV+: coverage={cov_resc:.4f} width={width_resc:.4f}")

    print("\n[stage3.5] === COMPARISON ===")
    print(f"[stage3.5] (a) plain Jackknife+/CV+ (Stage 1 follow-up, mapie):     coverage=37.06%  width=11.890")
    print(f"[stage3.5] (b) plain Jackknife+/CV+ (this script, manual, unscaled): coverage={cov_plain*100:.2f}%  width={width_plain:.3f}")
    print(f"[stage3.5] (c) 3.1 KMM-CP (this Stage's item 3.1):                   coverage=34.04%  width=9.924")
    print(f"[stage3.5] (d) rescaled Jackknife+/CV+ (THIS ITEM):                  coverage={cov_resc*100:.2f}%  width={width_resc:.3f}")

    delta_vs_plain = cov_resc - cov_plain
    delta_vs_kmm = cov_resc - 0.3404
    print(f"\n[stage3.5] rescaling vs. plain CV+ (this script's own unscaled run): {delta_vs_plain*100:+.2f}pp")
    print(f"[stage3.5] rescaled CV+ vs. 3.1's KMM-CP:                             {delta_vs_kmm*100:+.2f}pp")
    if delta_vs_plain > 0.02:
        verdict = "rescaling adds a real, meaningful improvement beyond plain Jackknife+/CV+"
    elif delta_vs_plain < -0.02:
        verdict = "rescaling makes coverage WORSE than plain Jackknife+/CV+ - a genuine negative result"
    else:
        verdict = "rescaling adds no meaningful improvement beyond plain Jackknife+/CV+ (within noise)"
    print(f"[stage3.5] VERDICT: {verdict}")

    pd.DataFrame([
        {"method": "plain_jackknife_cv_plus_mapie", "coverage": 0.3706, "avg_width": 11.890},
        {"method": "plain_jackknife_cv_plus_manual_unscaled", "coverage": cov_plain, "avg_width": width_plain},
        {"method": "kmm_cp_stage3_1", "coverage": 0.3404, "avg_width": 9.924},
        {"method": "rescaled_jackknife_cv_plus", "coverage": cov_resc, "avg_width": width_resc},
    ]).to_csv(OUT_DIR / "stage3_5_jackknife_rescaled_results.csv", index=False)
    print("\n[stage3.5] saved outputs/stage3_5_jackknife_rescaled_results.csv")
    print("[stage3.5] DONE")


if __name__ == "__main__":
    main()
