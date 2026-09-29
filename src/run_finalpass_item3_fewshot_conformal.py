"""
Final research pass, item 3: few-shot conformal calibration.

For each held-out dataset, recalibrates the split-conformal interval
using k labeled target cycles (k = 0, 5, 10, 20, 50), sampled from each
dataset's own EARLY cycles (cycle_idx <= that dataset's own 30th
percentile cycle_idx - a per-dataset early-cycle window, since datasets
vary hugely in how many cycles a battery runs). Evaluated on the
REMAINING cycles of that same dataset - the k sampled rows are removed
from the evaluation set, never double-counted (verified via a disjoint-
index assertion, not just described).

This is about COVERAGE RECOVERY, not accuracy - distinct from
run_researchpass2_item1_fewshot_adaptation.py's k-shot AFFINE
RECALIBRATION of the point prediction itself (that item never touched
conformal intervals). Here, the point-prediction model is NEVER
retrained or adapted - only the conformal calibration SET changes: for
k>0, the k target cycles' own (prediction, true SOH) pairs are POOLED
with the existing in-domain calibration half's residuals (the project's
own standing convention, e.g. stage1_common.calce_coverage) to form a
combined split-conformal calibration set. k=0 is the existing baseline
exactly (in-domain-only calibration) - a built-in continuity check.

Uses whichever model is CURRENTLY, ACTUALLY deployed/routed for each
dataset (extended-reformulation for CALCE/Oxford/HUST, base for XJTU
and every BatteryLife source - matches src/live_inference.py's real
EXTENDED_ROUTED_DATASETS positive list exactly), so the reported
numbers describe what a real user of the live app would actually see.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    load_all_heldout_base, load_all_heldout_extended,
    base_feature_cols, extended_feature_cols, fit_medians,
    load_base_model, load_extended_model, build_X, OUT_DIR, PROC_DIR,
)
from run_conformal import split_conformal, calib_eval_battery_split, ALPHA

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
K_VALUES = [0, 5, 10, 20, 50]
EARLY_CYCLE_PCTL = 30
SEED = 42


def main():
    t0 = time.time()
    print("=== Final pass item 3: few-shot conformal calibration (coverage recovery vs. k) ===")

    merged_base, hi_full_base, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_ext, hi_full_ext, hi_full_raw, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_cols = base_feature_cols()
    ext_cols = extended_feature_cols()
    base_medians = fit_medians(merged_base, train_mask_b, base_cols)
    ext_medians = fit_medians(merged_ext, train_mask_e, ext_cols)
    base_model = load_base_model()
    ext_model = load_extended_model()

    # In-domain CALIB-half residuals for each of the two models (the
    # existing "calibrate on one half of the NASA+MIT test batteries"
    # convention - same battery split for both, since it's the same
    # underlying battery set, only the FEATURE representation differs).
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

    print(f"[item3] in-domain calib half: {indomain_calib_mask_b.sum()} rows (base repr), "
          f"{indomain_calib_mask_e.sum()} rows (extended repr), from {len(calib_ids)} calib batteries")

    heldout_base = load_all_heldout_base(hi_full_base)
    heldout_ext = load_all_heldout_extended(hi_full_raw)

    def get_target_frame(name):
        if name in {"CALCE", "Oxford", "HUST", "XJTU"}:
            routed = name in EXTENDED_ROUTED_DATASETS
            df = heldout_ext[name] if routed else heldout_base[name]
            cols, medians, model = (ext_cols, ext_medians, ext_model) if routed else (base_cols, base_medians, base_model)
            indomain_pred = indomain_calib_pred_ext if routed else indomain_calib_pred_base
            indomain_y = indomain_calib_y_ext if routed else indomain_calib_y_base
        else:
            df = pd.read_parquet(PROC_DIR / f"batterylife_{name}_merged.parquet")
            cols, medians, model = base_cols, base_medians, base_model
            indomain_pred, indomain_y = indomain_calib_pred_base, indomain_calib_y_base
        return df, cols, medians, model, indomain_pred, indomain_y

    all_datasets = ["CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_SOURCES
    rows = []

    for name in all_datasets:
        if name in BATTERYLIFE_SOURCES and not (PROC_DIR / f"batterylife_{name}_merged.parquet").exists():
            print(f"[item3] {name}: no local merged table - skipped")
            continue
        df, cols, medians, model, indomain_pred, indomain_y = get_target_frame(name)

        X_target = build_X(df, cols, medians)
        pred_target = model.predict(X_target)
        y_target = df["SOH"].to_numpy()
        n_total = len(y_target)

        cutoff = np.percentile(df["cycle_idx"].to_numpy(), EARLY_CYCLE_PCTL)
        early_idx = np.where(df["cycle_idx"].to_numpy() <= cutoff)[0]
        rng = np.random.default_rng(SEED)

        for k in K_VALUES:
            if k == 0:
                calib_pred = indomain_pred
                calib_y = indomain_y
                eval_idx = np.arange(n_total)
                n_sampled = 0
            else:
                n_sampled = min(k, len(early_idx))
                sampled_idx = rng.choice(early_idx, size=n_sampled, replace=False) if n_sampled > 0 else np.array([], dtype=int)
                eval_idx = np.setdiff1d(np.arange(n_total), sampled_idx, assume_unique=False)
                assert len(np.intersect1d(sampled_idx, eval_idx)) == 0, "calibration/eval overlap - BUG"
                calib_pred = np.concatenate([indomain_pred, pred_target[sampled_idx]])
                calib_y = np.concatenate([indomain_y, y_target[sampled_idx]])

            _, lo, hi, method = split_conformal(calib_pred, calib_y, pred_target[eval_idx], ALPHA)
            covered = (y_target[eval_idx] >= lo) & (y_target[eval_idx] <= hi)
            coverage = float(covered.mean())
            width = float(np.mean(hi - lo))
            print(f"[item3] {name} k={k} (n_sampled={n_sampled}): coverage={coverage:.3f} "
                  f"width={width:.3f} n_eval={len(eval_idx)} method={method}")
            rows.append({"dataset": name, "k": k, "n_sampled_actual": n_sampled,
                         "coverage": coverage, "avg_width": width, "n_eval": len(eval_idx),
                         "n_total": n_total, "method": method})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "finalpass_item3_fewshot_conformal.csv", index=False)

    print("\n=== SUMMARY: coverage by dataset x k ===")
    cov_pivot = results_df.pivot(index="dataset", columns="k", values="coverage")
    print(cov_pivot.to_string())
    cov_pivot.to_csv(OUT_DIR / "finalpass_item3_coverage_pivot.csv")

    print("\n=== SUMMARY: avg_width by dataset x k ===")
    width_pivot = results_df.pivot(index="dataset", columns="k", values="avg_width")
    print(width_pivot.to_string())
    width_pivot.to_csv(OUT_DIR / "finalpass_item3_width_pivot.csv")

    target_coverage = 1 - ALPHA
    mean_cov_by_k = results_df.groupby("k")["coverage"].mean()
    mean_width_by_k = results_df.groupby("k")["avg_width"].mean()
    print(f"\n[item3] MEAN coverage across all {len(all_datasets)} held-out datasets, by k "
          f"(target={target_coverage:.0%}):")
    for k in K_VALUES:
        print(f"    k={k}: mean coverage={mean_cov_by_k.get(k, float('nan')):.3f}, "
              f"mean width={mean_width_by_k.get(k, float('nan')):.3f}")

    print(f"\n[item3] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
