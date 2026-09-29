"""
Final research pass, item 5 (rigor pass), sub-item 3: RUL cross-domain
results for every held-out dataset where RUL labels CAN be derived -
and an explicit statement of where they can't.

TWO SEPARATE QUESTIONS, kept distinct (conflating them would overstate
or understate this section):
  (1) Can RUL LABELS be derived at all? YES for all 13 held-out
      datasets - every one of them has a `discharge_capacity` column
      per cycle (confirmed directly), and rul_labels.compute_eol_and_
      rul/compute_eol_and_rul_severson_aware are dataset-agnostic
      (operate on any ordered cycle_idx/discharge_capacity sequence).
  (2) Can the DEPLOYED RUL MODEL (JointSOHRULModelFusion,
      models/joint_adaptive_fusion.pt) be SCORED zero-retrain on that
      dataset? Only for CALCE/Oxford/HUST/XJTU - it consumes a raw
      200-timestep, 6-channel (V/I/T/dQdV/dVdQ/dIdV) per-cycle tensor
      (sequence_features.build_dataset_tensors), which stage7_common.py
      already builds for these 4 datasets (reused unchanged here, same
      RUL labels every other Stage 7 item scores against). The 9
      BatteryLife sources were only ever adapted into the AGGREGATED
      tabular HI-feature pipeline (data_adapters_batterylife.py /
      build_batterylife_hi_table.py) - no raw per-cycle V/I/T tensor
      builder exists for them anywhere in this project. Building one
      from scratch (per-source raw-file parsing into the exact
      200-bin sequence format) is a real, non-trivial infrastructure
      project of its own, out of this item's scope - disclosed here,
      not silently skipped.

No retraining: the joint_fusion model is loaded exactly as
live_inference.py loads it for production, scored zero-retrain,
batched (not the live app's slow per-cycle loop).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage1_common import canonical_feature_cols, build_calce_merged, load_nasa_mit_pool, OUT_DIR, PROC_DIR, ROOT
from stage7_common import load_heldout_keyed
from sequence_features import apply_channel_norm
from run_stage1_followup_partB_joint_rul import JointSOHRULModelFusion

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]


def metrics(y_true, pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return {"r2": float(r2_score(y_true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred))}


def main():
    t0 = time.time()
    print("=== Final pass item 5c: RUL cross-domain (labels derivable everywhere; model scorable on 4/13) ===")

    canonical_rel = canonical_feature_cols(reformulated=True)  # 8 features, no cycle_idx - the RUL model's own input
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    joint_hi_norm = json.loads((PROC_DIR / "joint_hi_norm_stats.json").read_text())
    destd = json.loads((PROC_DIR / "destandardization_constants.json").read_text())
    hi_mean = np.array(joint_hi_norm["hi_mean"])
    hi_std = np.array(joint_hi_norm["hi_std"])
    rul_mean, rul_std = destd["rul_mean"], destd["rul_std"]

    joint_fusion = JointSOHRULModelFusion(n_hi_features=len(canonical_rel))
    joint_fusion.load_state_dict(torch.load(ROOT / "models" / "joint_adaptive_fusion.pt"))
    joint_fusion.eval()

    _, hi_full = load_nasa_mit_pool(reformulated=True)
    heldout_hi = {
        "CALCE": build_calce_merged(hi_full),
        "Oxford": pd.read_parquet(PROC_DIR / "stage5_1_oxford_merged.parquet"),
        "HUST": pd.read_parquet(PROC_DIR / "stage5_1_hust_merged.parquet"),
        "XJTU": pd.read_parquet(PROC_DIR / "stage5_1_xjtu_merged.parquet"),
    }

    rows = []
    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        print(f"\n[item5c] === {name}: building raw tensors + scoring RUL zero-retrain ===")
        pool = load_heldout_keyed(name)
        hi_df = heldout_hi[name].drop_duplicates(subset=["battery_id", "cycle_idx"])[
            ["battery_id", "cycle_idx"] + canonical_rel]

        # Build a lookup frame keyed exactly like the tensor pool (one row
        # per (battery_id, cycle_idx)), then a single vectorized merge -
        # avoids a per-row Python .loc loop, which is far too slow for
        # HUST's 146k rows (the earlier version of this script timed out).
        key_rows = []
        for bid, (X, soh, rul, idxs) in pool.items():
            for i, cyc in enumerate(idxs):
                key_rows.append((bid, int(cyc), bid, i))
        key_df = pd.DataFrame(key_rows, columns=["battery_id", "cycle_idx", "battery_key", "pos_idx"])
        merged_keys = key_df.merge(hi_df, on=["battery_id", "cycle_idx"], how="inner")
        n_dropped_no_hi = len(key_df) - len(merged_keys)

        if len(merged_keys) == 0:
            print(f"[item5c] {name}: no rows survived the HI join - skipped")
            continue

        bkeys = merged_keys["battery_key"].to_numpy()
        positions = merged_keys["pos_idx"].to_numpy()
        X_arr = np.stack([pool[bk][0][p] for bk, p in zip(bkeys, positions)]).astype(np.float32)
        rul_arr = np.array([pool[bk][2][p] for bk, p in zip(bkeys, positions)], dtype=float)
        hi_arr = merged_keys[canonical_rel].to_numpy(dtype=float)
        hi_arr = np.where(np.isinf(hi_arr), np.nan, hi_arr)
        with np.errstate(invalid="ignore"):
            col_medians = np.nanmedian(hi_arr, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)  # entirely-NaN column -> 0.0
        inds = np.where(np.isnan(hi_arr))
        hi_arr[inds] = np.take(col_medians, inds[1])

        X_norm = apply_channel_norm(X_arr, norm_stats)
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
        hi_z = (hi_arr - hi_mean) / hi_std
        hi_z = np.nan_to_num(hi_z, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            _, pred_rul_z = joint_fusion(torch.tensor(X_norm), torch.tensor(hi_z, dtype=torch.float32))
        pred_rul = pred_rul_z.numpy().reshape(-1) * rul_std + rul_mean

        valid = np.isfinite(pred_rul) & np.isfinite(rul_arr)
        n_invalid = int((~valid).sum())
        if n_invalid > 0:
            print(f"[item5c] {name}: WARNING {n_invalid} rows had non-finite pred/true RUL - excluded from metrics")
        rul_arr, pred_rul = rul_arr[valid], pred_rul[valid]

        m = metrics(rul_arr, pred_rul)
        n_batt = len(pool)
        print(f"[item5c] {name}: RUL R2={m['r2']:.4f} RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} "
              f"n={len(rul_arr)} n_batteries={n_batt} (dropped {n_dropped_no_hi} rows w/ no HI match)")
        rows.append({"dataset": name, "rul_r2": m["r2"], "rul_rmse": m["rmse"], "rul_mae": m["mae"],
                     "n": len(rul_arr), "n_batteries": n_batt, "n_dropped_no_hi_match": n_dropped_no_hi})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "finalpass_item5c_rul_crossdomain.csv", index=False)

    deployed_indomain_rul_r2 = 0.6656929850578308  # stage1_followup_partB_joint_rul_results.csv, in-domain TEST
    print(f"\n=== SUMMARY: RUL zero-retrain, deployed joint_fusion model (in-domain reference R2={deployed_indomain_rul_r2:.4f}) ===")
    print(results_df.to_string(index=False))

    print(f"\n[item5c] RUL labels ARE derivable (via rul_labels.compute_eol_and_rul, from each dataset's own "
          f"discharge_capacity column) for all 13 held-out datasets, including the 9 BatteryLife sources "
          f"({', '.join(BATTERYLIFE_SOURCES)}).")
    print(f"[item5c] RUL MODEL SCORING is NOT possible zero-retrain for those 9 BatteryLife sources: the "
          f"deployed joint_fusion model requires a raw 200-timestep/6-channel per-cycle tensor "
          f"(sequence_features.build_dataset_tensors), which no adapter in this project builds for BatteryLife "
          f"sources - data_adapters_batterylife.py only ever produced the AGGREGATED tabular HI-feature pipeline "
          f"(used by the SOH XGBoost-fusion items throughout this pass). Building a raw-tensor adapter per "
          f"BatteryLife source is a genuine, non-trivial infrastructure project, disclosed here as out of this "
          f"item's scope, not silently skipped.")

    print(f"\n[item5c] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
