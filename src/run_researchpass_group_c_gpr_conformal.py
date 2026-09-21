"""
Research pass Group C, item 9: composite-kernel Gaussian Process
Regression (RBF signal kernel + explicit WhiteKernel noise term) as a
locally-adaptive conformal scale model for CALCE - the same
"predict a local uncertainty, use it to rescale intervals" family as
item 8's isotonic approach and the project's own prior Normalized-CP
attempt, but via GPR's own native predictive standard deviation
instead of a separately-fit difficulty regressor.

SCALE NOTE, disclosed: GPR is O(n^3) in the number of fit points -
fitting on the full ~21k-row training pool is not tractable here in
reasonable time. Fit on a random SEED=42 subsample (same seed/
convention as every other subsampling decision in this project),
size chosen to keep fitting under a few minutes on this hardware -
a real, disclosed practical constraint, not a silently-shrunk dataset.

Reports coverage AND width, in the SAME consolidated comparison table
as item 8, against all prior CALCE attempts on record.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols,
    build_calce_merged, OUT_DIR, ROOT,
)
from xgboost import XGBRegressor

ALPHA = 0.1
GPR_FIT_SAMPLE = 800  # disclosed practical cap - GPR is O(n^3), full ~21k-row pool is not tractable here
SEED = 42

PRIOR_ATTEMPTS = [
    {"method": "Domain-alignment (MMD)", "coverage": 0.044, "width": 4.43},
    {"method": "Reweighted calibration (domain classifier)", "coverage": 0.044, "width": 4.38},
    {"method": "More training data alone (32->204 batteries)", "coverage": 0.074, "width": 2.309},
    {"method": "Normalized CP (GBR local-difficulty)", "coverage": 0.193, "width": 6.545},
    {"method": "CQR", "coverage": 0.213, "width": 18.267},
    {"method": "Jackknife+/CV+ resampling", "coverage": 0.371, "width": 11.89},
    {"method": "KMM-CP", "coverage": 0.340, "width": 9.924},
    {"method": "Domain-alignment (CORAL)", "coverage": 0.032, "width": 2.347},
    {"method": "Rescaled Jackknife+ (difficulty-aware)", "coverage": 0.820, "width": 58.905},
    {"method": "Plain split-conformal (do-nothing floor)", "coverage": 0.067, "width": 2.332},
]


def main():
    t0 = time.time()
    print("=== Research pass Group C, item 9: composite-kernel GPR conformal CALCE ===")
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    fcols = fusion_cols()
    cols = feature_cols + fcols

    model = XGBRegressor()
    model.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))

    X_all = merged[cols].to_numpy(dtype=float, copy=True)
    X_all = np.where(np.isinf(X_all), np.nan, X_all)
    train_medians = np.nanmedian(X_all[train_mask], axis=0)
    inds = np.where(np.isnan(X_all))
    X_all[inds] = np.take(train_medians, inds[1])
    y_all = merged["SOH"].to_numpy(dtype=float)
    pred_all = model.predict(X_all)
    resid_all = np.abs(y_all - pred_all)

    from run_conformal import calib_eval_battery_split
    test_ids = sorted(merged.loc[test_mask, "battery_id"].unique())
    calib_ids, eval_ids = calib_eval_battery_split(test_ids)
    calib_mask = merged["battery_id"].isin(calib_ids).to_numpy() & test_mask
    print(f"[gpr-conformal] calibration batteries: {calib_ids}, in-domain eval batteries: {eval_ids}")

    # GPR fits IN FEATURE SPACE (predicting local residual magnitude from
    # the same covariates as the point model), not just from the scalar
    # point prediction - a genuinely different, richer local-difficulty
    # signal than item 8's isotonic (scalar-input) approach.
    rng = np.random.default_rng(SEED)
    train_idx_all = np.where(train_mask)[0]
    gpr_fit_idx = rng.choice(train_idx_all, size=min(GPR_FIT_SAMPLE, len(train_idx_all)), replace=False)
    X_gpr_fit = X_all[gpr_fit_idx]
    y_gpr_fit = resid_all[gpr_fit_idx]
    print(f"[gpr-conformal] fitting GPR on {len(X_gpr_fit)} rows (subsampled from {len(train_idx_all)} "
          f"train rows - O(n^3) practical cap)")

    scaler = StandardScaler().fit(X_gpr_fit)
    X_gpr_fit_scaled = scaler.transform(X_gpr_fit)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X_gpr_fit_scaled.shape[1]),
                                                       length_scale_bounds=(1e-2, 1e2)) \
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 10.0))
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=SEED, normalize_y=True, alpha=1e-6)
    gpr.fit(X_gpr_fit_scaled, y_gpr_fit)
    print(f"[gpr-conformal] fitted kernel: {gpr.kernel_}")

    # calibration: predicted local-scale (GPR mean prediction of |residual|, floored positive)
    X_calib_scaled = scaler.transform(X_all[calib_mask])
    calib_scale_pred, _ = gpr.predict(X_calib_scaled, return_std=True)
    calib_scale_pred = np.clip(calib_scale_pred, 1e-3, None)
    calib_resid = resid_all[calib_mask]
    normalized_resid = calib_resid / calib_scale_pred
    n = len(normalized_resid)
    q_level = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
    q = np.quantile(normalized_resid, q_level, method="higher")
    print(f"[gpr-conformal] calibration: n={n}, normalized-residual quantile q={q:.4f}")

    calce_merged = build_calce_merged(hi_full)
    X_calce = calce_merged[cols].to_numpy(dtype=float, copy=True)
    X_calce = np.where(np.isinf(X_calce), np.nan, X_calce)
    inds_c = np.where(np.isnan(X_calce))
    X_calce[inds_c] = np.take(train_medians, inds_c[1])
    y_calce = calce_merged["SOH"].to_numpy(dtype=float)
    pred_calce = model.predict(X_calce)

    X_calce_scaled = scaler.transform(X_calce)
    calce_scale_pred, _ = gpr.predict(X_calce_scaled, return_std=True)
    calce_scale_pred = np.clip(calce_scale_pred, 1e-3, None)
    half_width = q * calce_scale_pred
    lo, hi = pred_calce - half_width, pred_calce + half_width
    covered = (y_calce >= lo) & (y_calce <= hi)
    coverage = float(covered.mean())
    avg_width = float(np.mean(hi - lo))
    print(f"[gpr-conformal] CALCE: coverage={coverage:.4f} ({coverage:.1%}), "
          f"avg_width={avg_width:.3f}, n={len(y_calce)}")

    results = PRIOR_ATTEMPTS + [{"method": "Composite-kernel GPR (RBF+WhiteKernel, THIS ITEM)",
                                  "coverage": coverage, "width": avg_width}]
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass_groupC9_gpr_conformal_calce.csv", index=False)
    print("\n=== CONSOLIDATED COMPARISON (all attempts on record) ===")
    print(results_df.to_string(index=False))

    best_prior_efficient = min(
        [r for r in PRIOR_ATTEMPTS if r["coverage"] >= 0.30],
        key=lambda r: r["width"], default=None,
    )
    if coverage >= 0.30 and best_prior_efficient and avg_width < best_prior_efficient["width"]:
        print(f"\n[gpr-conformal] VERDICT: WIN - reaches >=30% coverage at a narrower width than "
              f"any prior attempt in that coverage range")
    elif coverage < 0.10:
        print(f"\n[gpr-conformal] VERDICT: LOSS - coverage still near the do-nothing floor")
    else:
        print(f"\n[gpr-conformal] VERDICT: reported honestly against the table above")
    print(f"\n[gpr-conformal] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
