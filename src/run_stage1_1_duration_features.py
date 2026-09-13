"""
Stage 1, Item 1.1: protocol-invariant duration features.

Session 27 root-caused ICHV/TEVI's B0018-vs-MIT-train z-scores of
855/420 to NASA's slow-cycling protocol producing raw wall-clock
durations that are simply on a different absolute scale than MIT's
fast-charging protocol - encoding PROTOCOL, not battery health. This
reformulates every raw-duration feature in the canonical set (ICHV,
TEVD, TEVI - see stage1_common.DURATION_FEATURES; TEVD is a duration
too, added here as "any other raw time/duration HI" per the task,
though it wasn't part of the original leaked 7-feature set session 27
evaluated) as a ratio against that SAME BATTERY's own cycle-10 value.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    CANONICAL_FEATURES, DURATION_FEATURES, canonical_feature_cols,
    load_nasa_mit_pool, battery_split_masks, fit_xgb, eval_indomain,
    build_calce_merged, eval_calce, calce_coverage, OUT_DIR,
)


def zscore_b0018_vs_mit_train(hi_full: pd.DataFrame, split, features: list[str]) -> pd.DataFrame:
    """Exactly session 27's formula (run_b0018_root_cause_analysis.py
    section3_feature_outliers): z = (B0018_mean - MIT_train_mean) /
    MIT_train_std, computed per-feature on hi_full directly (not the
    fusion-merged frame) so it matches session 27's methodology exactly."""
    mit_train = hi_full[(hi_full["dataset"] == "MIT") & (hi_full["battery_id"].isin(split["train_ids"]))]
    b0018 = hi_full[hi_full["battery_id"] == "B0018"]
    rows = []
    for feat in features:
        tr = mit_train[feat].dropna()
        tgt = b0018[feat].dropna()
        mean, std = tr.mean(), tr.std() + 1e-8
        z = (tgt.mean() - mean) / std
        rows.append({"feature": feat, "mit_train_mean": mean, "mit_train_std": std,
                      "b0018_mean": tgt.mean(), "zscore": z})
    return pd.DataFrame(rows)


def main():
    print(f"[stage1.1] duration features being reformulated: {DURATION_FEATURES}")
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    split_json = None
    train_mask, test_mask, split_json = battery_split_masks(merged)

    # --- Step 2: z-score comparison, raw vs. reformulated ---
    raw_z = zscore_b0018_vs_mit_train(hi_full, split_json, DURATION_FEATURES)
    rel_features = [f"{f}_rel" for f in DURATION_FEATURES]
    rel_z = zscore_b0018_vs_mit_train(hi_full, split_json, rel_features)
    print("\n[stage1.1] === Z-SCORE COMPARISON: raw duration vs. reformulated (B0018 vs. MIT-train) ===")
    for raw_feat, (_, raw_row), (_, rel_row) in zip(DURATION_FEATURES, raw_z.iterrows(), rel_z.iterrows()):
        print(f"[stage1.1] {raw_feat}: raw z={raw_row['zscore']:.1f}  ->  "
              f"{raw_feat}_rel z={rel_row['zscore']:.1f}")
    z_compare = pd.concat([raw_z.assign(kind="raw"), rel_z.assign(kind="reformulated")], ignore_index=True)
    z_compare.to_csv(OUT_DIR / "stage1_1_zscore_comparison.csv", index=False)

    # --- Step 3: retrain XGBoost-fusion with reformulated duration features ---
    feature_cols = canonical_feature_cols(reformulated=True)
    print(f"\n[stage1.1] retraining with reformulated canonical feature set: {feature_cols}")
    model, medians, cols = fit_xgb(merged, train_mask, feature_cols)
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    print(f"[stage1.1] IN-DOMAIN (NASA+MIT test): RMSE={indomain['rmse']:.4f} "
          f"MAE={indomain['mae']:.4f} R2={indomain['r2']:.4f}")

    calce_merged = build_calce_merged(hi_full)
    calce = eval_calce(model, medians, feature_cols, calce_merged)
    print(f"[stage1.1] CALCE (zero-retrain): RMSE={calce['rmse']:.4f} "
          f"MAE={calce['mae']:.4f} R2={calce['r2']:.4f}")

    cov = calce_coverage(indomain, calce)
    print(f"[stage1.1] CALCE conformal coverage: empirical={cov['empirical_coverage']:.3f} "
          f"avg_width={cov['avg_interval_width']:.3f}")

    baseline = pd.read_csv(OUT_DIR / "stage1_canonical_baseline.csv")
    base_indomain = baseline[baseline.domain == "in-domain"].iloc[0]
    base_calce = baseline[baseline.domain == "CALCE"].iloc[0]
    print("\n[stage1.1] === COMPARISON vs. canonical-raw baseline (Stage 1 reference) ===")
    print(f"[stage1.1] In-domain R2:  baseline={base_indomain['r2']:.4f}  reformulated={indomain['r2']:.4f}  "
          f"(delta={indomain['r2']-base_indomain['r2']:+.4f})")
    print(f"[stage1.1] CALCE R2:      baseline={base_calce['r2']:.4f}  reformulated={calce['r2']:.4f}  "
          f"(delta={calce['r2']-base_calce['r2']:+.4f})")
    print(f"[stage1.1] CALCE coverage: baseline={base_calce['coverage']:.3f}  reformulated={cov['empirical_coverage']:.3f}  "
          f"(delta={cov['empirical_coverage']-base_calce['coverage']:+.3f})")

    calce_improved = calce['r2'] > base_calce['r2'] + 0.01
    calce_worse = calce['r2'] < base_calce['r2'] - 0.01
    if calce_improved:
        verdict = "CALCE IMPROVES - the protocol-invariant reformulation measurably narrows the CALCE gap."
    elif calce_worse:
        verdict = "CALCE WORSENS - the reformulation does NOT help, and actively hurts CALCE generalization."
    else:
        verdict = "CALCE is ESSENTIALLY UNCHANGED - the well-motivated reformulation does not measurably close the gap, despite the sound z-score-based reasoning."
    print(f"\n[stage1.1] VERDICT: {verdict}")

    pd.DataFrame([
        {"variant": "1.1_reformulated_duration", "domain": "in-domain",
         "rmse": indomain["rmse"], "mae": indomain["mae"], "r2": indomain["r2"]},
        {"variant": "1.1_reformulated_duration", "domain": "CALCE",
         "rmse": calce["rmse"], "mae": calce["mae"], "r2": calce["r2"],
         "coverage": cov["empirical_coverage"], "coverage_width": cov["avg_interval_width"]},
    ]).to_csv(OUT_DIR / "stage1_1_results.csv", index=False)
    print("[stage1.1] saved outputs/stage1_1_{zscore_comparison,results}.csv")
    print("[stage1.1] DONE")


if __name__ == "__main__":
    main()
