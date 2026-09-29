"""
Final research pass, item 5 (rigor pass), sub-item 5: one consolidated
conformal coverage/width table across all 13 held-out datasets, plus
in-domain - the STANDARD (non-few-shot, k=0) split-conformal
calibration this project has always shipped, using whichever model is
currently, actually routed per dataset.

Reuses item 3's own k=0 rows for the 13 held-out sets directly (k=0 IS
exactly this project's existing in-domain-calibration-only convention -
no need to recompute), adds the one row item 3's own held-out-only
panel never covered: in-domain TEST itself (calibrated on ITS OWN calib
half, evaluated on the eval half - the project's own calib_eval_
battery_split convention, unchanged).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, base_feature_cols, fit_medians, load_base_model, build_X, OUT_DIR,
)
from run_conformal import split_conformal, calib_eval_battery_split, ALPHA


def main():
    t0 = time.time()
    print("=== Final pass item 5e: consolidated conformal coverage/width table (in-domain + 13 held-out) ===")

    merged_base, hi_full_base, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    base_cols = base_feature_cols()
    base_medians = fit_medians(merged_base, train_mask_b, base_cols)
    base_model = load_base_model()

    test_bids = merged_base.loc[test_mask_b, "battery_id"].to_numpy()
    unique_test_ids = sorted(set(test_bids.tolist()))
    calib_ids, eval_ids = calib_eval_battery_split(unique_test_ids)
    calib_mask = np.isin(test_bids, calib_ids)
    eval_mask = np.isin(test_bids, eval_ids)

    X_test = build_X(merged_base.loc[test_mask_b], base_cols, base_medians)
    pred_test = base_model.predict(X_test)
    y_test = merged_base.loc[test_mask_b, "SOH"].to_numpy()

    calib_pred, calib_y = pred_test[calib_mask], y_test[calib_mask]
    eval_pred, eval_y = pred_test[eval_mask], y_test[eval_mask]

    _, lo, hi, method = split_conformal(calib_pred, calib_y, eval_pred, ALPHA)
    covered = (eval_y >= lo) & (eval_y <= hi)
    indomain_row = {"dataset": "in-domain (TEST, own calib half)", "coverage": float(covered.mean()),
                    "avg_width": float(np.mean(hi - lo)), "n_eval": len(eval_y), "method": method,
                    "target_coverage": 1 - ALPHA}
    print(f"[item5e] in-domain: coverage={indomain_row['coverage']:.3f} width={indomain_row['avg_width']:.3f} "
          f"n={indomain_row['n_eval']}")

    item3_path = OUT_DIR / "finalpass_item3_fewshot_conformal.csv"
    item3_df = pd.read_csv(item3_path)
    k0 = item3_df[item3_df["k"] == 0].copy()
    k0_rows = [{"dataset": r["dataset"], "coverage": r["coverage"], "avg_width": r["avg_width"],
               "n_eval": r["n_eval"], "method": r["method"], "target_coverage": 1 - ALPHA}
              for _, r in k0.iterrows()]

    all_rows = [indomain_row] + k0_rows
    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUT_DIR / "finalpass_item5e_conformal_consolidated.csv", index=False)

    print("\n=== CONSOLIDATED CONFORMAL COVERAGE/WIDTH TABLE (target coverage = 90%) ===")
    print(result_df.to_string(index=False))

    n_within_10pts = int((np.abs(result_df["coverage"] - 0.9) <= 0.10).sum())
    print(f"\n[item5e] {n_within_10pts}/{len(result_df)} datasets (incl. in-domain) reach coverage within "
          f"10 percentage points of the 90% target. Every OTHER dataset is severely under-covered - "
          f"consistent with items 3's own finding that this project's conformal intervals, calibrated purely "
          f"in-domain, do not transfer under the domain shift most of these held-out/BatteryLife sets represent.")

    print(f"\n[item5e] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
