"""
Stage 1, Item 1.2: per-battery sample weighting.

Session 27: NASA is 2.7% of training cycles despite being 11.5% of
training batteries - every cycle currently contributes equally to the
loss, so short-lived (NASA) batteries are structurally underweighted.
Weight w_i = 1/n_cycles(battery(i)), renormalized so weights sum to the
original row count (keeps XGBoost's effective learning-rate/regularization
balance the same as the unweighted baseline - only the RELATIVE emphasis
across batteries changes, not the total mass).

Uses Stage 1.1's reformulated duration features as the base feature set
(1.1 completed and is a clear, verified improvement - see its own
report), stated explicitly per this item's own instruction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, build_calce_merged, eval_calce, calce_coverage, OUT_DIR,
)


def per_battery_sample_weight(merged: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    n = len(merged)
    w = np.ones(n, dtype=float)
    train_df = merged[train_mask]
    n_cycles_per_battery = train_df.groupby("battery_id").size()
    raw_w = train_df["battery_id"].map(1.0 / n_cycles_per_battery).to_numpy()
    # renormalize so weighted total mass == original number of train rows
    raw_w *= train_mask.sum() / raw_w.sum()
    w[train_mask] = raw_w
    return w


def main():
    print("[stage1.2] base feature set: Stage 1.1's reformulated duration features (1.1 already complete)")
    feature_cols = canonical_feature_cols(reformulated=True)
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)

    weights = per_battery_sample_weight(merged, train_mask)
    train_df = merged[train_mask]
    n_nasa_cyc = (train_df["dataset"] == "NASA").sum()
    n_mit_cyc = (train_df["dataset"] == "MIT").sum()
    nasa_weight_share = weights[train_mask][train_df["dataset"].to_numpy() == "NASA"].sum() / weights[train_mask].sum()
    print(f"[stage1.2] train cycles: NASA={n_nasa_cyc} ({n_nasa_cyc/len(train_df)*100:.1f}%), "
          f"MIT={n_mit_cyc} ({n_mit_cyc/len(train_df)*100:.1f}%)")
    print(f"[stage1.2] after 1/n_cycles(battery) weighting: NASA's share of total WEIGHT mass = "
          f"{nasa_weight_share*100:.1f}% (was {n_nasa_cyc/len(train_df)*100:.1f}% of raw cycle count)")
    print(f"[stage1.2] weight sum={weights[train_mask].sum():.1f} (train rows={train_mask.sum()}, "
          f"confirms renormalization preserved total mass)")

    model, medians, cols = fit_xgb(merged, train_mask, feature_cols, sample_weight=weights)
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    print(f"[stage1.2] IN-DOMAIN (NASA+MIT test): RMSE={indomain['rmse']:.4f} "
          f"MAE={indomain['mae']:.4f} R2={indomain['r2']:.4f}")

    calce_merged = build_calce_merged(hi_full)
    calce = eval_calce(model, medians, feature_cols, calce_merged)
    print(f"[stage1.2] CALCE (zero-retrain): RMSE={calce['rmse']:.4f} MAE={calce['mae']:.4f} R2={calce['r2']:.4f}")

    cov = calce_coverage(indomain, calce)
    print(f"[stage1.2] CALCE conformal coverage: empirical={cov['empirical_coverage']:.3f} "
          f"avg_width={cov['avg_interval_width']:.3f}")

    # NASA/B0018's own individual RMSE, in-domain
    b0018_mask = indomain["battery_id"] == "B0018"
    b0018_rmse = float(np.sqrt(np.mean((indomain["pred"][b0018_mask] - indomain["y_true"][b0018_mask]) ** 2)))
    print(f"[stage1.2] NASA/B0018 individual RMSE (weighted): {b0018_rmse:.4f}")

    unweighted = pd.read_csv(OUT_DIR / "stage1_1_results.csv")
    base_indomain = unweighted[unweighted.domain == "in-domain"].iloc[0]
    base_calce = unweighted[unweighted.domain == "CALCE"].iloc[0]

    # recompute B0018 RMSE for the unweighted (1.1) model for direct comparison
    model_u, medians_u, cols_u = fit_xgb(merged, train_mask, feature_cols)
    indomain_u = eval_indomain(model_u, medians_u, cols_u, merged, test_mask)
    b0018_mask_u = indomain_u["battery_id"] == "B0018"
    b0018_rmse_u = float(np.sqrt(np.mean((indomain_u["pred"][b0018_mask_u] - indomain_u["y_true"][b0018_mask_u]) ** 2)))

    print("\n[stage1.2] === COMPARISON vs. unweighted (1.1-reformulated-features) baseline ===")
    print(f"[stage1.2] In-domain R2:  unweighted={base_indomain['r2']:.4f}  weighted={indomain['r2']:.4f}  "
          f"(delta={indomain['r2']-base_indomain['r2']:+.4f})")
    print(f"[stage1.2] CALCE R2:      unweighted={base_calce['r2']:.4f}  weighted={calce['r2']:.4f}  "
          f"(delta={calce['r2']-base_calce['r2']:+.4f})")
    print(f"[stage1.2] CALCE coverage: unweighted={base_calce['coverage']:.3f}  weighted={cov['empirical_coverage']:.3f}  "
          f"(delta={cov['empirical_coverage']-base_calce['coverage']:+.3f})")
    print(f"[stage1.2] B0018 RMSE:    unweighted={b0018_rmse_u:.4f}  weighted={b0018_rmse:.4f}  "
          f"(delta={b0018_rmse-b0018_rmse_u:+.4f})")

    delta_r2_indomain = indomain['r2'] - base_indomain['r2']
    delta_r2_calce = calce['r2'] - base_calce['r2']
    delta_b0018 = b0018_rmse - b0018_rmse_u
    b0018_helped = delta_b0018 < -0.01
    indomain_neutral_or_better = delta_r2_indomain > -0.01
    calce_hurt = delta_r2_calce < -0.01
    calce_helped = delta_r2_calce > 0.01
    if b0018_helped and indomain_neutral_or_better and calce_hurt:
        verdict = ("MIXED, NOT a clean win: B0018's own error improves "
                   f"({delta_b0018:+.4f} RMSE) and pooled in-domain R2 is essentially unchanged "
                   f"({delta_r2_indomain:+.4f}), but CALCE R2 drops by a real, non-negligible amount "
                   f"({delta_r2_calce:+.4f}) - reweighting toward NASA's proportional battery share "
                   "helps the specific NASA battery it targets, but actively hurts generalization to "
                   "the third, unrelated (CALCE) domain. This is a genuine trade-off, not a free win.")
    elif b0018_helped and indomain_neutral_or_better and calce_helped:
        verdict = "CLEAN WIN: B0018 improves, in-domain is unchanged or better, and CALCE also improves."
    elif b0018_helped and indomain_neutral_or_better:
        verdict = "HELPS: B0018's own error improves with no meaningful pooled or CALCE cost."
    elif b0018_helped:
        verdict = "TRADE-OFF: B0018 improves but at a measurable pooled in-domain cost."
    elif abs(delta_b0018) <= 0.01 and abs(delta_r2_indomain) <= 0.01 and abs(delta_r2_calce) <= 0.01:
        verdict = "NEUTRAL: no meaningful change to B0018, pooled in-domain, or CALCE."
    else:
        verdict = "HURTS or is NEUTRAL on B0018 specifically, despite the theoretically-motivated rebalancing."
    print(f"\n[stage1.2] VERDICT: {verdict}")

    pd.DataFrame([
        {"variant": "1.2_sample_weighted", "domain": "in-domain",
         "rmse": indomain["rmse"], "mae": indomain["mae"], "r2": indomain["r2"], "b0018_rmse": b0018_rmse},
        {"variant": "1.2_sample_weighted", "domain": "CALCE",
         "rmse": calce["rmse"], "mae": calce["mae"], "r2": calce["r2"],
         "coverage": cov["empirical_coverage"], "coverage_width": cov["avg_interval_width"]},
    ]).to_csv(OUT_DIR / "stage1_2_results.csv", index=False)
    print("[stage1.2] saved outputs/stage1_2_results.csv")
    print("[stage1.2] DONE")


if __name__ == "__main__":
    main()
