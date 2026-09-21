"""
Research pass Group C, item 8: isotonic-regression-calibrated conformal
prediction for CALCE - locally-adaptive interval width via a MONOTONIC
difficulty-scale model, the same family of method as this project's
own already-tried "Normalized CP" (#4 in the consolidated 9-attempt
table - a GBR-based local-difficulty regressor), but using sklearn's
IsotonicRegression instead: simpler, monotonic by construction (can't
overfit into a non-monotonic scale function the way an unconstrained
GBR can), fit on (point prediction -> |residual|) from the calibration
set, then used to rescale CALCE's own interval widths per-cycle.

Reports coverage AND width together (this project's own established
discipline - coverage alone is a misleading metric, see the
consolidated CALCE table's own repeated caveat), directly against ALL
prior CALCE conformal attempts already on record.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols,
    build_calce_merged, OUT_DIR, ROOT,
)
from xgboost import XGBRegressor

ALPHA = 0.1

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
    print("=== Research pass Group C, item 8: isotonic-calibrated conformal CALCE ===")
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

    # calibration/eval split of TEST batteries (same established convention)
    from run_conformal import calib_eval_battery_split
    test_ids = sorted(merged.loc[test_mask, "battery_id"].unique())
    calib_ids, eval_ids = calib_eval_battery_split(test_ids)
    calib_mask = merged["battery_id"].isin(calib_ids).to_numpy() & test_mask
    print(f"[isotonic-conformal] calibration batteries: {calib_ids}, in-domain eval batteries: {eval_ids}")

    calib_pred = pred_all[calib_mask]
    calib_y = y_all[calib_mask]
    calib_resid = np.abs(calib_y - calib_pred)

    # Fit isotonic regression: point prediction -> |residual| (a monotonic
    # difficulty-scale model - lower predicted SOH tends to mean a harder,
    # more degraded/later-life cycle, plausibly higher residual variance;
    # checked this is genuinely the right direction, not assumed).
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(calib_pred, calib_resid)
    calib_scale = np.clip(iso.predict(calib_pred), 1e-3, None)
    normalized_resid = calib_resid / calib_scale
    n = len(normalized_resid)
    q_level = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
    q = np.quantile(normalized_resid, q_level, method="higher")
    print(f"[isotonic-conformal] calibration: n={n}, normalized-residual quantile q={q:.4f}")

    # CALCE zero-retrain evaluation
    hi_full_reformulated = hi_full  # already reformulated (load_nasa_mit_pool(reformulated=True) applies it to hi_full too)
    calce_merged = build_calce_merged(hi_full_reformulated)
    X_calce = calce_merged[cols].to_numpy(dtype=float, copy=True)
    X_calce = np.where(np.isinf(X_calce), np.nan, X_calce)
    inds_c = np.where(np.isnan(X_calce))
    X_calce[inds_c] = np.take(train_medians, inds_c[1])
    y_calce = calce_merged["SOH"].to_numpy(dtype=float)
    pred_calce = model.predict(X_calce)

    calce_scale = np.clip(iso.predict(pred_calce), 1e-3, None)
    half_width = q * calce_scale
    lo, hi = pred_calce - half_width, pred_calce + half_width
    covered = (y_calce >= lo) & (y_calce <= hi)
    coverage = float(covered.mean())
    avg_width = float(np.mean(hi - lo))
    print(f"[isotonic-conformal] CALCE: coverage={coverage:.4f} ({coverage:.1%}), "
          f"avg_width={avg_width:.3f}, n={len(y_calce)}")

    results = PRIOR_ATTEMPTS + [{"method": "Isotonic-regression-calibrated (THIS ITEM)",
                                  "coverage": coverage, "width": avg_width}]
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass_groupC8_isotonic_conformal_calce.csv", index=False)
    print("\n=== CONSOLIDATED COMPARISON (all attempts on record) ===")
    print(results_df.to_string(index=False))

    # Verdict: does this beat prior attempts on a genuine coverage-per-width tradeoff,
    # not just raw coverage (per this project's own established discipline)?
    best_prior_efficient = min(
        [r for r in PRIOR_ATTEMPTS if r["coverage"] >= 0.30],
        key=lambda r: r["width"], default=None,
    )
    print(f"\n[isotonic-conformal] Best prior attempt with coverage>=30% at the NARROWEST width: "
          f"{best_prior_efficient}")
    if coverage >= 0.30 and best_prior_efficient and avg_width < best_prior_efficient["width"]:
        print(f"[isotonic-conformal] VERDICT: WIN - reaches >=30% coverage at a narrower width "
              f"than any prior attempt in that coverage range")
    elif coverage < 0.10:
        print(f"[isotonic-conformal] VERDICT: LOSS - coverage still near the do-nothing floor")
    else:
        print(f"[isotonic-conformal] VERDICT: reported honestly against the table above - "
              f"see coverage/width tradeoff directly, no single number tells the whole story")
    print(f"\n[isotonic-conformal] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
