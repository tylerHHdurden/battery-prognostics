"""
Stage 5.1: zero-retrain generalization eval of the frozen Stage 4
XGBoost-fusion model on 3 newly-integrated, never-trained-on datasets
(Oxford, HUST, XJTU), evaluated EXACTLY as CALCE has always been
evaluated in this project - same frozen model, same frozen
channel_norm_stats.json ("clip bounds", fit on NASA+MIT+recovered only,
never refit per new dataset), same frozen ica_encoder.pt, same reformulated
8-feature + cycle_idx + 16-fusion-embedding (25-dim) input, never included
in any training split.

Reuses stage1_common.py's exact CALCE-eval code path (build_calce_merged/
eval_calce's logic, generalized to any dataset name) rather than
reimplementing it, so results are directly comparable apples-to-apples -
this project's established practice (see stage1_common.py's own docstring
on why Stage 0/1's CALCE checks share one code path).

Verified BEFORE this script was trusted: reconstructing Stage 4's exact
train-column medians via load_nasa_mit_pool+battery_split_masks and
re-evaluating the frozen model on the frozen in-domain test split
reproduces Stage 4's own numbers (R2=0.9739568832487542, RMSE=
0.780492685553031) to full float precision - confirms this script's
feature/median/model pipeline is bit-identical to Stage 4's, not a
close-but-different reimplementation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters import (
    oxford_cell_ids, iterate_oxford_cycles,
    hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids, iterate_xjtu_cycles,
)

# ROOT-CAUSED DATA ISSUE (not a code bug): XJTU's Batch-6 ("Sim_satellite")
# cells use a simulated variable-load discharge profile - unlike every
# other protocol in this whole project (NASA/MIT/CALCE/Oxford/HUST/XJTU's
# other 5 batches), each cycle's discharge depth is NOT roughly constant
# (verified directly: cell Batch-6/Sim_satellite_battery-1's raw per-
# cycle discharge_capacity is 1.991/0.111/0.445/0.756/... Ah - a >10x
# swing between adjacent cycles, by design, simulating real satellite
# power draw). rul_labels.soh_per_cycle's project-wide convention
# (nominal capacity = median of the battery's own first 3 cycles) is
# silently invalid here: it can lock onto an atypically shallow early
# cycle as "100% SOH", producing SOH ratios up to 457% for later, more-
# fully-discharged cycles - confirmed by direct inspection (all 8
# Batch-6 cells share an identical, suspicious minimum SOH of
# 24.9438202247191%, a strong signature of a shared labeling artifact,
# not real physical variance). Excluded from every SOH-based comparison
# in this stage, disclosed rather than silently dropped - these 8 cells'
# RAW cycling data remain real and usable for a future non-SOH-ratio
# analysis, just not this one.
XJTU_EXCLUDED_PREFIX = "Batch-6/Sim_satellite"


def xjtu_cell_ids_soh_valid():
    return [c for c in xjtu_cell_ids() if not c.startswith(XJTU_EXCLUDED_PREFIX)]
from health_indicators import compute_health_indicators
from rul_labels import soh_per_cycle
from sequence_features import build_dataset_tensors, apply_channel_norm
from models.ica_encoder import ICAEncoder
from stage1_common import (
    canonical_feature_cols, add_reformulated_duration_features, fusion_cols,
    load_nasa_mit_pool, battery_split_masks, calce_coverage,
    build_calce_merged, eval_calce, OUT_DIR, PROC_DIR, ROOT, ALPHA,
)

ICA_CHANNEL_SLICE = slice(3, 6)


def build_dataset_hi_and_tensors(dataset_name: str, cell_ids, iterate_fn, min_cycles=5):
    """Generalization of stage1_common.build_calce_tensors + the
    HI-table-row construction from run_stage4_feature_regen.py's
    process_battery, unified into one function so it works for any new
    dataset following the standard cycle-record contract."""
    hi_rows = []
    all_X, all_bid, all_cyc = [], [], []
    skipped = []
    for cid in cell_ids:
        cycles = list(iterate_fn(cid))
        if len(cycles) < min_cycles:
            skipped.append((cid, len(cycles)))
            continue

        soh_map = soh_per_cycle(cycles)
        for c in cycles:
            his = compute_health_indicators(c)
            row = {"dataset": dataset_name, "battery_id": cid, "cycle_idx": c["cycle_idx"],
                   "discharge_capacity": c["discharge_capacity"], "SOH": soh_map[c["cycle_idx"]]}
            row.update(his)
            hi_rows.append(row)

        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            print(f"[stage5-1] WARNING: {dataset_name}/{cid} produced 0 valid tensors "
                  f"(all {len(cycles)} cycles failed get_cycle_tensor) - excluded from fusion "
                  f"embeddings (and thus from the eval, since fusion embeddings are required).")
            continue
        all_X.append(X.astype(np.float32))
        all_bid += [cid] * len(idxs)
        all_cyc += list(idxs)

    if skipped:
        print(f"[stage5-1] {dataset_name}: skipped {len(skipped)} cells with <{min_cycles} cycles: {skipped}")

    hi_df = pd.DataFrame(hi_rows)
    X_all = np.concatenate(all_X) if all_X else None
    return hi_df, X_all, all_bid, all_cyc


def eval_new_dataset(dataset_name: str, cell_ids, iterate_fn, model, medians, feature_cols,
                      norm_stats, encoder):
    hi_df, X_all, bid, cyc = build_dataset_hi_and_tensors(dataset_name, cell_ids, iterate_fn)
    if X_all is None or len(hi_df) == 0:
        return None, None, hi_df

    n_cells = hi_df["battery_id"].nunique()
    n_cycles_raw = len(hi_df)
    n_tensor_cycles = len(bid)
    print(f"[stage5-1] {dataset_name}: {n_cells} cells, {n_cycles_raw} HI rows, "
          f"{n_tensor_cycles} tensor-valid cycles ({n_cycles_raw - n_tensor_cycles} dropped by get_cycle_tensor)")

    X_norm = apply_channel_norm(X_all, norm_stats)
    with torch.no_grad():
        fusion_emb = encoder.encode(torch.tensor(X_norm[:, :, ICA_CHANNEL_SLICE])).numpy()

    hi_reformulated = add_reformulated_duration_features(hi_df)
    seq_df = pd.DataFrame({"battery_id": bid, "cycle_idx": cyc})
    for i in range(16):
        seq_df[f"fusion_{i}"] = fusion_emb[:, i]
    merged = pd.merge(hi_reformulated, seq_df, on=["battery_id", "cycle_idx"], how="inner")
    print(f"[stage5-1] {dataset_name}: {len(merged)} rows after HI/fusion-embedding merge "
          f"({n_cycles_raw - len(merged)} lost to the merge)")

    result = eval_calce(model, medians, feature_cols, merged)  # generic despite the name - see stage1_common.py
    return result, merged, hi_df


def main():
    print("=== Stage 5.1: loading frozen Stage 4 resources (NOT retraining anything) ===")
    model = XGBRegressor()
    model.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()

    # exact reproduction of Stage 4's train-column medians (verified to
    # reproduce Stage 4's own in-domain R2/RMSE to full float precision
    # before this script was trusted - see module docstring)
    merged_nm, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged_nm)
    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    cols = feature_cols + fusion_cols()
    X_nm = merged_nm[cols].to_numpy(dtype=float, copy=True)
    medians = np.nanmedian(X_nm[train_mask], axis=0)

    Xc = X_nm.copy()
    inds = np.where(np.isnan(Xc))
    Xc[inds] = np.take(medians, inds[1])
    sanity_pred = model.predict(Xc[test_mask])
    sanity_r2 = r2_score(merged_nm.loc[test_mask, "SOH"], sanity_pred)
    print(f"[stage5-1] sanity check - reproduced in-domain R2={sanity_r2:.10f} "
          f"(must match Stage 4's 0.9739568832487542)")
    assert abs(sanity_r2 - 0.9739568832487542) < 1e-9, "median/feature reconstruction does not match Stage 4!"

    # CALCE, recomputed fresh here (unchanged model/pool) for a like-for-like
    # comparison row in the SAME table - not re-deriving a new number, just
    # confirming this script's own pipeline reproduces Stage 4's own CALCE line.
    calce_merged = build_calce_merged(hi_full)
    calce_result = eval_calce(model, medians, feature_cols, calce_merged)
    print(f"[stage5-1] CALCE (reproduced) R2={calce_result['r2']:.4f} RMSE={calce_result['rmse']:.4f} "
          f"(Stage 4 reported R2=0.5679 RMSE=14.155)")

    datasets = [
        ("Oxford", oxford_cell_ids(), iterate_oxford_cycles),
        ("HUST", hust_cell_ids(), iterate_hust_cycles),
        ("XJTU", xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles),
    ]

    summary_rows = [{
        "dataset": "CALCE", "n_cells": len(set(calce_merged["battery_id"])),
        "n_cycles": len(calce_merged), "r2": calce_result["r2"], "rmse": calce_result["rmse"],
        "mae": calce_result["mae"],
    }]
    all_results = {"CALCE": calce_result}

    for name, ids, iterate_fn in datasets:
        print(f"\n=== {name} ===")
        result, merged, hi_df = eval_new_dataset(name, ids, iterate_fn, model, medians, feature_cols,
                                                   norm_stats, encoder)
        if result is None:
            print(f"[stage5-1] {name}: NO USABLE DATA - skipped entirely, reported as such.")
            summary_rows.append({"dataset": name, "n_cells": 0, "n_cycles": 0,
                                  "r2": np.nan, "rmse": np.nan, "mae": np.nan})
            continue
        print(f"[stage5-1] {name}: R2={result['r2']:.4f} RMSE={result['rmse']:.4f} MAE={result['mae']:.4f} "
              f"(n={len(result['pred'])}, {len(set(result['battery_id']))} cells)")
        all_results[name] = result
        summary_rows.append({
            "dataset": name, "n_cells": len(set(result["battery_id"])),
            "n_cycles": len(result["pred"]), "r2": result["r2"], "rmse": result["rmse"],
            "mae": result["mae"],
        })
        merged.to_parquet(PROC_DIR / f"stage5_1_{name.lower()}_merged.parquet")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "stage5_1_zero_retrain_generalization.csv", index=False)
    print("\n=== SUMMARY (zero-retrain generalization, individually per dataset) ===")
    print(summary.to_string(index=False))

    # bonus: conformal coverage per item 5 ("if time permits, don't block")
    print("\n=== bonus: conformal coverage (plain split-conformal, same convention as CALCE's) ===")
    cov_rows = []
    indomain_full = load_indomain_result(merged_nm, model, medians, feature_cols, test_mask)
    for name, result in all_results.items():
        cov = calce_coverage(indomain_full, result, alpha=ALPHA)
        cov_rows.append({"dataset": name, **cov})
        print(f"[stage5-1] {name} coverage={cov['empirical_coverage']*100:.2f}% "
              f"width={cov['avg_interval_width']:.3f} (target {(1-ALPHA)*100:.0f}%)")
    pd.DataFrame(cov_rows).to_csv(OUT_DIR / "stage5_1_conformal_coverage.csv", index=False)


def load_indomain_result(merged_nm, model, medians, feature_cols, test_mask):
    cols = feature_cols + fusion_cols()
    X = merged_nm[cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    pred = model.predict(X[test_mask])
    y_true = merged_nm.loc[test_mask, "SOH"].to_numpy(dtype=float)
    return {"pred": pred, "y_true": y_true, "battery_id": merged_nm.loc[test_mask, "battery_id"].to_numpy()}


if __name__ == "__main__":
    main()
