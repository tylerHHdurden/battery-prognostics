"""
Stage 1, pre-1.1: canonical-feature-set baseline. Retrains XGBoost-fusion
on the CORRECTED (Check 0.3, NASA+MIT-only) 8-feature canonical set, raw
(unreformulated), unweighted, default objective, no monotone constraints
- i.e. every Stage 1 mechanism OFF. This is the single shared reference
point 1.1/1.2/1.5 each compare their ONE change against, so those three
comparisons isolate their own effect rather than conflating it with the
feature-set switch away from Stage 0's leaked 7-feature set.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    CANONICAL_FEATURES, load_nasa_mit_pool, battery_split_masks, fit_xgb,
    eval_indomain, build_calce_merged, eval_calce, calce_coverage, OUT_DIR,
)


def main():
    print(f"[stage1-baseline] canonical features: {CANONICAL_FEATURES}")
    merged, hi_full = load_nasa_mit_pool(reformulated=False)
    train_mask, test_mask, split = battery_split_masks(merged)
    print(f"[stage1-baseline] NASA+MIT pool: {len(merged)} rows, "
          f"{train_mask.sum()} train rows, {test_mask.sum()} test rows")

    model, medians, cols = fit_xgb(merged, train_mask, CANONICAL_FEATURES)
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    print(f"[stage1-baseline] IN-DOMAIN (NASA+MIT test): RMSE={indomain['rmse']:.4f} "
          f"MAE={indomain['mae']:.4f} R2={indomain['r2']:.4f}")

    calce_merged = build_calce_merged(hi_full)
    calce = eval_calce(model, medians, CANONICAL_FEATURES, calce_merged)
    print(f"[stage1-baseline] CALCE (zero-retrain): RMSE={calce['rmse']:.4f} "
          f"MAE={calce['mae']:.4f} R2={calce['r2']:.4f}")

    cov = calce_coverage(indomain, calce)
    print(f"[stage1-baseline] CALCE conformal coverage: method={cov['method']} "
          f"target={cov['target_coverage']:.0%} empirical={cov['empirical_coverage']:.3f} "
          f"avg_width={cov['avg_interval_width']:.3f} (n_calib={cov['n_calib']}, n_calce={cov['n_calce']})")

    pd.DataFrame([
        {"variant": "canonical_raw_baseline", "domain": "in-domain",
         "rmse": indomain["rmse"], "mae": indomain["mae"], "r2": indomain["r2"]},
        {"variant": "canonical_raw_baseline", "domain": "CALCE",
         "rmse": calce["rmse"], "mae": calce["mae"], "r2": calce["r2"],
         "coverage": cov["empirical_coverage"], "coverage_width": cov["avg_interval_width"]},
    ]).to_csv(OUT_DIR / "stage1_canonical_baseline.csv", index=False)
    print("[stage1-baseline] saved outputs/stage1_canonical_baseline.csv")
    print("[stage1-baseline] DONE")


if __name__ == "__main__":
    main()
