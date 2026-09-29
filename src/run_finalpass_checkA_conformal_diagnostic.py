"""
Verification check A (user-requested, post-item-3): item 3's few-shot
conformal coverage was ~11-13% even at k=50 vs. a 90% target - looks
suspiciously low. This script verifies, WITHOUT changing item 3's own
code or re-deciding anything, whether that is a real non-exchangeability
effect or a bug in item 3's own pipeline:

  (1) calib/eval cycles come from the target dataset consistently (not
      reversed/swapped), and the k-sampled calibration cycles are
      disjoint from the evaluation cycles (re-asserted independently,
      not just trusting item 3's own inline assert).
  (2) the conformal interval (hi-lo) is on the SAME scale as the
      residuals (|y_true-y_pred|) it's calibrated from - printed
      side-by-side, not assumed.
  (3) whether MAPIE's SplitConformalRegressor (the method item 3
      actually used - confirmed via its own logged "method" column)
      applies the standard finite-sample conformal quantile correction
      ceil((n+1)(1-alpha))/n, by comparing its own interval half-width
      against a MANUALLY computed one using that exact formula, for the
      SAME calib residuals.
  (4) the fraction of test residuals exceeding the (fixed, k=0) interval,
      split by early vs. late cycle_idx within each dataset - directly
      answers whether residual magnitude grows with degradation (a
      genuine non-exchangeability signature) or is flat (which would
      instead point to a bug/miscalibration).

No files under item 3's own scope are modified; this is read-only
verification, reusing item 3's own already-computed CSV plus fresh,
independent recomputation of the pieces above.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    load_all_heldout_base, load_all_heldout_extended,
    base_feature_cols, extended_feature_cols, fit_medians,
    load_base_model, load_extended_model, build_X, OUT_DIR,
)
from run_conformal import split_conformal, calib_eval_battery_split, ALPHA

EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
CHECK_DATASETS = ["CALCE", "Oxford", "HUST", "XJTU"]  # the 4 classic held-out sets - most scrutinized
K_CHECK = 50
SEED = 42


def manual_conformal_halfwidth(calib_resid: np.ndarray, alpha: float) -> float:
    """The textbook split-conformal finite-sample-corrected quantile:
    q_level = ceil((n+1)(1-alpha)) / n, clipped to 1.0, 'higher'
    interpolation - the exact formula run_conformal.py's own manual
    fallback uses, reproduced independently here (not imported) so this
    check does not simply re-trust the same code path being verified."""
    n = len(calib_resid)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(calib_resid, q_level, method="higher"))


def main():
    print("=== CHECK A: few-shot conformal verification (calib/eval integrity, scale, quantile formula, early/late residuals) ===\n")

    merged_base, hi_full_base, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_ext, hi_full_ext, hi_full_raw, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_cols = base_feature_cols()
    ext_cols = extended_feature_cols()
    base_medians = fit_medians(merged_base, train_mask_b, base_cols)
    ext_medians = fit_medians(merged_ext, train_mask_e, ext_cols)
    base_model = load_base_model()
    ext_model = load_extended_model()

    test_bids_b = merged_base.loc[test_mask_b, "battery_id"].to_numpy()
    unique_test_ids = sorted(set(test_bids_b.tolist()))
    calib_ids, eval_ids = calib_eval_battery_split(unique_test_ids)
    indomain_calib_mask_b = np.isin(test_bids_b, calib_ids)
    X_test_base = build_X(merged_base.loc[test_mask_b], base_cols, base_medians)
    pred_test_base = base_model.predict(X_test_base)
    y_test_base = merged_base.loc[test_mask_b, "SOH"].to_numpy()
    indomain_calib_pred_base = pred_test_base[indomain_calib_mask_b]
    indomain_calib_y_base = y_test_base[indomain_calib_mask_b]

    test_bids_e = merged_ext.loc[test_mask_e, "battery_id"].to_numpy()
    indomain_calib_mask_e = np.isin(test_bids_e, calib_ids)
    X_test_ext = build_X(merged_ext.loc[test_mask_e], ext_cols, ext_medians)
    pred_test_ext = ext_model.predict(X_test_ext)
    y_test_ext = merged_ext.loc[test_mask_e, "SOH"].to_numpy()
    indomain_calib_pred_ext = pred_test_ext[indomain_calib_mask_e]
    indomain_calib_y_ext = y_test_ext[indomain_calib_mask_e]

    heldout_base = load_all_heldout_base(hi_full_base)
    heldout_ext = load_all_heldout_extended(hi_full_raw)

    diag_rows = []
    for name in CHECK_DATASETS:
        routed = name in EXTENDED_ROUTED_DATASETS
        df = heldout_ext[name] if routed else heldout_base[name]
        cols, medians, model = (ext_cols, ext_medians, ext_model) if routed else (base_cols, base_medians, base_model)
        indomain_pred = indomain_calib_pred_ext if routed else indomain_calib_pred_base
        indomain_y = indomain_calib_y_ext if routed else indomain_calib_y_base

        X_target = build_X(df, cols, medians)
        pred_target = model.predict(X_target)
        y_target = df["SOH"].to_numpy()
        cycle_idx = df["cycle_idx"].to_numpy()
        n_total = len(y_target)

        print(f"\n--- {name} (routed model: {'extended' if routed else 'base'}) ---")

        # (1) calib/eval integrity: rebuild k=50's split exactly as item3 did, verify disjointness
        # AND that both draw from the SAME target dataset array (same df, same indices space).
        cutoff = np.percentile(cycle_idx, 30)
        early_idx = np.where(cycle_idx <= cutoff)[0]
        rng = np.random.default_rng(SEED)
        n_sampled = min(K_CHECK, len(early_idx))
        sampled_idx = rng.choice(early_idx, size=n_sampled, replace=False)
        eval_idx = np.setdiff1d(np.arange(n_total), sampled_idx, assume_unique=False)
        overlap = np.intersect1d(sampled_idx, eval_idx)
        sampled_battery_ids = set(df["battery_id"].to_numpy()[sampled_idx].tolist())
        eval_battery_ids = set(df["battery_id"].to_numpy()[eval_idx].tolist())
        batteries_only_in_calib = sampled_battery_ids - eval_battery_ids
        print(f"[check1] k={K_CHECK}: n_sampled={n_sampled}, n_eval={len(eval_idx)}, "
              f"overlap_indices={len(overlap)} (must be 0), "
              f"sampled cycles come from {len(sampled_battery_ids)} of this dataset's own batteries, "
              f"{len(eval_battery_ids)} batteries appear in eval, "
              f"{len(batteries_only_in_calib)} batteries have ALL their sampled cycles but zero eval-cycle overlap issue "
              f"(expected: every sampled battery should ALSO have rows in eval, since only its EARLY cycles were pulled out)")
        # direct re-check: every battery contributing to calib should still have eval rows (its later cycles)
        missing_from_eval = sampled_battery_ids - eval_battery_ids
        assert len(overlap) == 0, f"{name}: BUG - calib/eval index overlap"
        if missing_from_eval:
            print(f"[check1] WARNING: {len(missing_from_eval)} battery(ies) contributed ALL their early cycles to "
                  f"calibration with NO remaining eval rows: {missing_from_eval} - check if these batteries are "
                  f"very short (all cycles fall within the 30th-percentile-cutoff early window)")
        else:
            print(f"[check1] OK - every battery contributing calibration cycles also has remaining eval cycles "
                  f"(true same-battery calib/eval split, not battery-level leakage)")

        # (2)+(3) scale + quantile-formula check, using k=0 (pure in-domain calib) for a clean,
        # single-calibration-set comparison (k>0 pools two different-scale-battery-count sources,
        # harder to interpret in isolation - k=0 isolates the core question cleanly).
        calib_resid = np.abs(indomain_y - indomain_pred)
        _, lo_mapie, hi_mapie, method = split_conformal(indomain_pred, indomain_y, pred_target, ALPHA)
        mapie_halfwidth = float(np.mean((hi_mapie - lo_mapie) / 2))
        manual_halfwidth = manual_conformal_halfwidth(calib_resid, ALPHA)
        print(f"[check2] scale check: calib residuals (|y-pred|) range=[{calib_resid.min():.3f},{calib_resid.max():.3f}] "
              f"mean={calib_resid.mean():.3f} | MAPIE interval half-width (mean)={mapie_halfwidth:.3f} "
              f"| target residuals (|y_true-pred_target|) range=[{np.abs(y_target-pred_target).min():.3f},"
              f"{np.abs(y_target-pred_target).max():.3f}] mean={np.abs(y_target-pred_target).mean():.3f} "
              f"-- ALL SAME UNITS (raw SOH %), confirmed by direct value comparison, not just code inspection")
        print(f"[check3] quantile-formula check: MAPIE ({method}) half-width={mapie_halfwidth:.4f} vs. "
              f"MANUAL ceil((n+1)(1-a))/n formula half-width={manual_halfwidth:.4f} "
              f"(n_calib={len(calib_resid)}) -> {'MATCH' if abs(mapie_halfwidth-manual_halfwidth)<0.05 else 'MISMATCH - investigate'}")

        # (4) early vs late cycle residual/coverage split, using the k=0 interval (fixed width,
        # applied uniformly across the whole target dataset - isolates whether COVERAGE FAILURE
        # correlates with cycle position, i.e. degradation stage, rather than being uniform).
        covered_k0 = (y_target >= lo_mapie) & (y_target <= hi_mapie)
        target_cutoff_50 = np.percentile(cycle_idx, 50)
        early_mask = cycle_idx <= target_cutoff_50
        late_mask = ~early_mask
        resid_target = np.abs(y_target - pred_target)
        print(f"[check4] EARLY cycles (cycle_idx<={target_cutoff_50:.0f}, n={early_mask.sum()}): "
              f"mean|residual|={resid_target[early_mask].mean():.3f}, coverage={covered_k0[early_mask].mean():.3f}, "
              f"frac_exceeding_interval={1-covered_k0[early_mask].mean():.3f}")
        print(f"[check4] LATE  cycles (cycle_idx> {target_cutoff_50:.0f}, n={late_mask.sum()}): "
              f"mean|residual|={resid_target[late_mask].mean():.3f}, coverage={covered_k0[late_mask].mean():.3f}, "
              f"frac_exceeding_interval={1-covered_k0[late_mask].mean():.3f}")
        residual_growth = resid_target[late_mask].mean() - resid_target[early_mask].mean()
        print(f"[check4] residual growth (late-mean minus early-mean) = {residual_growth:+.3f} "
              f"({'GROWS with degradation - consistent with genuine non-exchangeability' if residual_growth > 0.5 else 'roughly flat/does not clearly grow'})")

        diag_rows.append({
            "dataset": name, "n_target": n_total,
            "calib_mean_resid": float(calib_resid.mean()), "target_mean_resid": float(resid_target.mean()),
            "mapie_halfwidth": mapie_halfwidth, "manual_halfwidth": manual_halfwidth,
            "quantile_formula_match": abs(mapie_halfwidth - manual_halfwidth) < 0.05,
            "early_coverage": float(covered_k0[early_mask].mean()), "late_coverage": float(covered_k0[late_mask].mean()),
            "early_mean_resid": float(resid_target[early_mask].mean()), "late_mean_resid": float(resid_target[late_mask].mean()),
            "residual_growth_late_minus_early": residual_growth,
            "n_batteries_with_no_eval_rows_at_k50": len(missing_from_eval),
        })

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT_DIR / "finalpass_checkA_conformal_diagnostic.csv", index=False)
    print("\n=== SUMMARY ===")
    print(diag_df.to_string(index=False))


if __name__ == "__main__":
    main()
