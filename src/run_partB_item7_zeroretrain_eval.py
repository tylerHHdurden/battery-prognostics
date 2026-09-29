"""
Part B, item 7: zero-retrain evaluation of the CURRENT deployed setup
(dataset-aware routing, `src/live_inference.py`'s own
`EXTENDED_ROUTED_DATASETS = {"CALCE","Oxford","HUST"}`) against every
BatteryLife sub-source this pass actually has usable local data for
(built by `src/build_batterylife_hi_table.py`, from
`data/processed/batterylife_<source>_merged.parquet`).

WHAT "CURRENT DEPLOYED" MEANS FOR A BRAND-NEW DATASET NAME, STATED
EXPLICITLY: `EXTENDED_ROUTED_DATASETS` is a POSITIVE list - any dataset
name not literally in it falls through to the BASE model
(`models/xgb_soh_fusion.json`), by construction of the actual deployed
code, not a judgment call made here. Since none of BatteryLife's source
names are in that list, the base model IS what the live app would
actually run today for every one of these - reported as the PRIMARY,
real "current deployed" number. The extended-reformulation model's own
number is ALSO reported alongside, for context (whether extending the
routing list to include a new source would help), never as the primary
"deployed" figure - a real distinction, not blurred.

No retraining, no fitting on this data - both models loaded already-
trained (identical to Part A), imputation medians computed from the
SAME NASA+MIT train-pool split the deployed models actually use (never
from the new data itself - the SAME convention Part A and
`live_inference.py` both already use).
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    base_feature_cols, extended_feature_cols, fit_medians,
    load_base_model, load_extended_model, build_X, score, OUT_DIR, PROC_DIR,
)

SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth", "stanford", "stanford_2", "isu_ilcc"]
ROUTED_BASELINE = {"in-domain (fixed split)": 0.974, "CALCE": 0.740, "Oxford": 0.953, "HUST": 0.800, "XJTU": -1.062}


def main():
    t0 = time.time()
    print("=== Part B, item 7: zero-retrain eval of the CURRENT deployed setup on new BatteryLife sources ===")

    merged_base, hi_full_base, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_ext, hi_full_ext, hi_full_raw, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_cols = base_feature_cols()
    ext_cols = extended_feature_cols()
    base_medians = fit_medians(merged_base, train_mask_b, base_cols)
    ext_medians = fit_medians(merged_ext, train_mask_e, ext_cols)
    base_model = load_base_model()
    ext_model = load_extended_model()

    results = []
    for source in SOURCES:
        path = PROC_DIR / f"batterylife_{source}_merged.parquet"
        if not path.exists():
            print(f"[item7] {source}: no local merged table - skipped (not yet downloaded/built)")
            continue
        df = pd.read_parquet(path)
        n_batt = df["battery_id"].nunique()

        X_base = build_X(df, base_cols, base_medians)
        pred_base = base_model.predict(X_base)
        r_base = score(df["SOH"].to_numpy(), pred_base)

        X_ext = build_X(df, ext_cols, ext_medians)
        pred_ext = ext_model.predict(X_ext)
        r_ext = score(df["SOH"].to_numpy(), pred_ext)

        print(f"[item7] {source} ({n_batt} batteries, {len(df)} cycles): "
              f"CURRENT DEPLOYED (base model, actual routing) R2={r_base['r2']:.4f} RMSE={r_base['rmse']:.4f} | "
              f"extended-reformulation model (context only, NOT currently routed here) R2={r_ext['r2']:.4f} RMSE={r_ext['rmse']:.4f}")
        results.append({"source": source, "n_batteries": n_batt, "n_cycles": len(df),
                         "deployed_base_r2": r_base["r2"], "deployed_base_rmse": r_base["rmse"],
                         "extended_context_r2": r_ext["r2"], "extended_context_rmse": r_ext["rmse"]})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "partB_item7_zeroretrain_eval.csv", index=False)
    print("\n=== Item 7 FULL RESULTS ===")
    print(results_df.to_string(index=False))
    print(f"\n[item7] For reference, the EXISTING 4 held-out sets' CURRENT deployed (routed) numbers: {ROUTED_BASELINE}")
    print(f"\n[item7] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
