"""
Stage 1, Item 1.5: XGBoost monotone_constraints.

Session 4's soft physics-informed monotonicity PENALTY (added to a deep
model's loss) made every deep model worse and was excluded. This is a
different, structural mechanism: XGBoost's native monotone_constraints
forces every individual tree split to respect the constraint exactly
(not just softly penalize violations), applied to XGBoost-fusion
specifically - the model that actually ships, which session 4 never
touched.

Base configuration decision (stated explicitly per this item's own
instruction): layered on top of Stage 1.1's reformulated duration
features ONLY, not 1.2's per-battery sample weighting - 1.2 was a
genuine, non-negligible trade-off (CALCE R2 -0.115) rather than a clean
win, so it is not being carried forward as an adopted default. 1.1's
reformulation was a clean, unambiguous improvement on every axis
checked and IS carried forward.

Sign-of-constraint verification (per the task's explicit warning that a
reversed constraint would silently produce garbage): checked directly
before picking a sign - corr(cycle_idx, SOH) is negative for ALL 32
NASA+MIT batteries individually (mean r=-0.872, range -0.75 to -0.99,
computed fresh from hi_table.parquet, not assumed). SOH is therefore
constrained NON-INCREASING as cycle_idx increases: monotone_constraints
value of -1 for cycle_idx, 0 (unconstrained) for every other feature.

cycle_idx is NOT part of the 8-feature canonical BFA set (BFA selects
only HI columns, never cycle_idx itself) and was never a model input
before this item - added here as an ADDITIONAL 9th feature purely to
give the constraint mechanism something with a guaranteed physical
monotonic relationship to attach to, per the task's own example ("e.g.
cycle number itself").
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, build_calce_merged, eval_calce, calce_coverage,
    fusion_cols, OUT_DIR,
)


def main():
    print("[stage1.5] base configuration: Stage 1.1's reformulated duration features "
          "(1.2's sample weighting NOT carried forward - see module docstring)")
    base_features = canonical_feature_cols(reformulated=True)
    feature_cols = base_features + ["cycle_idx"]
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)

    # --- sign verification, printed plainly before use ---
    corrs = merged[train_mask].groupby("battery_id").apply(
        lambda g: g["cycle_idx"].corr(g["SOH"]), include_groups=False)
    print(f"[stage1.5] corr(cycle_idx, SOH) per training battery: mean={corrs.mean():.3f}, "
          f"range=[{corrs.min():.3f}, {corrs.max():.3f}], negative for {(corrs < 0).sum()}/{len(corrs)} batteries")
    assert (corrs < 0).all(), "not all batteries show a negative cycle_idx/SOH relationship - STOP, do not assume the sign"
    print("[stage1.5] confirmed: ALL training batteries negative -> monotone_constraints=-1 for cycle_idx (non-increasing)")

    n_fusion = len(fusion_cols())
    monotone = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)
    print(f"[stage1.5] feature order: {feature_cols + fusion_cols()}")
    print(f"[stage1.5] monotone_constraints: {monotone}")

    model, medians, cols = fit_xgb(merged, train_mask, feature_cols,
                                    xgb_extra_kwargs={"monotone_constraints": monotone})
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    print(f"[stage1.5] IN-DOMAIN (NASA+MIT test): RMSE={indomain['rmse']:.4f} "
          f"MAE={indomain['mae']:.4f} R2={indomain['r2']:.4f}")

    calce_merged = build_calce_merged(hi_full)
    calce = eval_calce(model, medians, feature_cols, calce_merged)
    print(f"[stage1.5] CALCE (zero-retrain): RMSE={calce['rmse']:.4f} MAE={calce['mae']:.4f} R2={calce['r2']:.4f}")

    cov = calce_coverage(indomain, calce)
    print(f"[stage1.5] CALCE conformal coverage: empirical={cov['empirical_coverage']:.3f} "
          f"avg_width={cov['avg_interval_width']:.3f}")

    # --- spot-check individual battery curves for non-physical (SOH increasing) steps ---
    def count_increases(pred, cyc):
        order = np.argsort(cyc)
        p = pred[order]
        return int((np.diff(p) > 1e-9).sum()), len(p) - 1

    print("\n[stage1.5] === spot-check: non-physical (SOH-increasing) steps per battery, unconstrained vs. constrained ===")
    base_1_1 = fit_xgb(merged, train_mask, base_features)
    model_u, medians_u, cols_u = base_1_1
    indomain_u = eval_indomain(model_u, medians_u, cols_u, merged, test_mask)

    spot_rows = []
    for bid in sorted(set(indomain["battery_id"].tolist())):
        mask_c = indomain["battery_id"] == bid
        mask_u = indomain_u["battery_id"] == bid
        cyc = merged.loc[test_mask, "cycle_idx"].to_numpy()[mask_c]
        n_inc_c, n_steps_c = count_increases(indomain["pred"][mask_c], cyc)
        n_inc_u, n_steps_u = count_increases(indomain_u["pred"][mask_u], cyc)
        spot_rows.append({"battery_id": bid, "n_steps": n_steps_c,
                           "n_increasing_steps_unconstrained": n_inc_u,
                           "n_increasing_steps_constrained": n_inc_c})
        print(f"[stage1.5] {bid}: non-physical increasing steps - unconstrained={n_inc_u}/{n_steps_u}, "
              f"constrained={n_inc_c}/{n_steps_c}")
    spot_df = pd.DataFrame(spot_rows)
    spot_df.to_csv(OUT_DIR / "stage1_5_monotonicity_spotcheck.csv", index=False)

    baseline = pd.read_csv(OUT_DIR / "stage1_1_results.csv")
    base_indomain = baseline[baseline.domain == "in-domain"].iloc[0]
    base_calce = baseline[baseline.domain == "CALCE"].iloc[0]
    print("\n[stage1.5] === COMPARISON vs. Stage 1.1 (reformulated, unconstrained) baseline ===")
    print(f"[stage1.5] In-domain R2:  1.1={base_indomain['r2']:.4f}  1.5={indomain['r2']:.4f}  "
          f"(delta={indomain['r2']-base_indomain['r2']:+.4f})")
    print(f"[stage1.5] CALCE R2:      1.1={base_calce['r2']:.4f}  1.5={calce['r2']:.4f}  "
          f"(delta={calce['r2']-base_calce['r2']:+.4f})")
    print(f"[stage1.5] CALCE coverage: 1.1={base_calce['coverage']:.3f}  1.5={cov['empirical_coverage']:.3f}  "
          f"(delta={cov['empirical_coverage']-base_calce['coverage']:+.3f})")
    total_inc_u = spot_df["n_increasing_steps_unconstrained"].sum()
    total_inc_c = spot_df["n_increasing_steps_constrained"].sum()
    print(f"[stage1.5] TOTAL non-physical increasing steps across all test batteries: "
          f"unconstrained={total_inc_u}, constrained={total_inc_c}")

    delta_r2_indomain = indomain['r2'] - base_indomain['r2']
    delta_r2_calce = calce['r2'] - base_calce['r2']
    if total_inc_c < total_inc_u and delta_r2_indomain > -0.01 and delta_r2_calce > -0.01:
        verdict = ("HELPS: measurably fewer non-physical SOH-increasing steps "
                   f"({total_inc_u}->{total_inc_c}) with no meaningful accuracy cost.")
    elif total_inc_c < total_inc_u:
        verdict = (f"MIXED: fewer non-physical steps ({total_inc_u}->{total_inc_c}) but at a "
                   f"real accuracy cost (in-domain delta={delta_r2_indomain:+.4f}, CALCE delta={delta_r2_calce:+.4f}).")
    elif abs(delta_r2_indomain) <= 0.01 and abs(delta_r2_calce) <= 0.01:
        verdict = "NEUTRAL: constraint changes neither the non-physical-step count nor accuracy meaningfully."
    else:
        verdict = ("DOES NOT HELP (consistent with session 4's precedent for a different mechanism): "
                   f"no reduction in non-physical steps ({total_inc_u}->{total_inc_c}) "
                   f"and/or a real accuracy cost (in-domain delta={delta_r2_indomain:+.4f}, CALCE delta={delta_r2_calce:+.4f}).")
    print(f"\n[stage1.5] VERDICT: {verdict}")

    pd.DataFrame([
        {"variant": "1.5_monotone_constrained", "domain": "in-domain",
         "rmse": indomain["rmse"], "mae": indomain["mae"], "r2": indomain["r2"]},
        {"variant": "1.5_monotone_constrained", "domain": "CALCE",
         "rmse": calce["rmse"], "mae": calce["mae"], "r2": calce["r2"],
         "coverage": cov["empirical_coverage"], "coverage_width": cov["avg_interval_width"]},
    ]).to_csv(OUT_DIR / "stage1_5_results.csv", index=False)
    print("[stage1.5] saved outputs/stage1_5_{results,monotonicity_spotcheck}.csv")
    print("[stage1.5] DONE")


if __name__ == "__main__":
    main()
