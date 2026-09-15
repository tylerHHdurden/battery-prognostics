"""
Follow-on: saves full per-cycle (battery_id, y_true, pred) triplets for
each (pool, held-out dataset) combination - needed for a proper
battery-level cluster bootstrap (session 21/Stage 0.4's methodology),
which requires resampling raw rows, not just summary R2/RMSE numbers
(the only things the earlier pool-comparison runs saved to disk).

Three pools: "" (deployed 42-battery, models/xgb_soh_fusion.json),
"_pool204" (204-battery), "_nasaheavy" (NASA-heavy) - suffix selects
which artifacts to load (channel_norm_stats{suffix}.json, ica_encoder
{suffix}.pt, xgb_soh_fusion{suffix}.json). No retraining - pure
inference with each pool's own already-trained, already-saved model.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from models.ica_encoder import ICAEncoder
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks, fusion_cols,
    build_calce_merged, eval_calce, OUT_DIR, PROC_DIR, ROOT,
)
from run_pool204_step2_retrain_eval import build_new_dataset_merged
from run_stage5_1_new_datasets_eval import (
    oxford_cell_ids, iterate_oxford_cycles, hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids_soh_valid, iterate_xjtu_cycles,
)

EMBED_DIM = 16
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]
PRED_DIR = PROC_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)


def main(pool_suffix: str, pool_label: str):
    t0 = time.time()
    print(f"=== Saving per-cycle predictions for pool: {pool_label} (suffix='{pool_suffix}') ===")

    norm_stats_file = PROC_DIR / f"channel_norm_stats{pool_suffix}.json"
    encoder_file = ROOT / "models" / f"ica_encoder{pool_suffix}.pt"
    model_file = ROOT / "models" / f"xgb_soh_fusion{pool_suffix}.json"

    norm_stats = json.loads(norm_stats_file.read_text())
    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder.load_state_dict(torch.load(encoder_file))
    encoder.eval()
    model = XGBRegressor()
    model.load_model(str(model_file))

    # reconstruct this pool's own medians (deterministic from its own
    # hi_table + fusion_embeddings + battery_split - no retraining)
    if pool_suffix == "":
        merged_nm, hi_full = load_nasa_mit_pool(reformulated=True)
        train_mask, test_mask, split = battery_split_masks(merged_nm)
        feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
        fcols = fusion_cols()
        X_nm = merged_nm[[*feature_cols, *fcols]].to_numpy(dtype=float, copy=True)
        X_nm = np.where(np.isinf(X_nm), np.nan, X_nm)
        medians = np.nanmedian(X_nm[train_mask], axis=0)
        calce_merged = build_calce_merged(hi_full)
    else:
        from stage1_common import add_reformulated_duration_features
        hi_table_file = PROC_DIR / f"hi_table{pool_suffix}.parquet"
        fusion_file = PROC_DIR / f"fusion_embeddings{pool_suffix}.csv"
        split_file = PROC_DIR / f"battery_split{pool_suffix}.json" if pool_suffix == "_pool204" else None
        if pool_suffix == "_pool204":
            split_file = PROC_DIR / "battery_split_expanded_b0018pinned.json"
        else:
            split_file = PROC_DIR / f"battery_split{pool_suffix}.json"
        split = json.loads(split_file.read_text())
        train_ids = split["train_ids"]

        hi_full = pd.read_parquet(hi_table_file)
        hi_reformulated = add_reformulated_duration_features(hi_full)
        fusion_df = pd.read_csv(fusion_file)
        train_datasets = ["NASA", "MIT"] if pool_suffix == "_pool204" else ["NASA", "NASA_RANDOMIZED", "MIT"]
        nasa_mit = hi_reformulated[hi_reformulated["dataset"].isin(train_datasets)]
        merged = pd.merge(nasa_mit, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
        train_mask = merged["battery_id"].isin(train_ids).to_numpy()

        feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
        fcols = [f"fusion_{i}" for i in range(EMBED_DIM)]
        X = merged[[*feature_cols, *fcols]].to_numpy(dtype=float, copy=True)
        X = np.where(np.isinf(X), np.nan, X)
        medians = np.nanmedian(X[train_mask], axis=0)

        # CALCE: own tensors, this pool's own norm_stats/encoder (NOT
        # looked up in fusion_df, which never contains CALCE - the
        # exact bug already found and fixed once for pool204)
        calce_merged = build_new_dataset_merged("CALCE", CALCE_CELLS,
                                                 __import__("data_adapters").iterate_calce_cycles,
                                                 norm_stats, encoder)

    def predict_df(df, cols):
        X = df[cols].to_numpy(dtype=float, copy=True)
        X = np.where(np.isinf(X), np.nan, X)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(medians, inds[1])
        pred = model.predict(X)
        return pred

    cols = feature_cols + fcols

    datasets = {"CALCE": calce_merged}
    print("[save-preds] building Oxford/HUST/XJTU under this pool's own norm_stats/encoder...")
    for name, ids, iterate_fn in [
        ("Oxford", oxford_cell_ids(), iterate_oxford_cycles),
        ("HUST", hust_cell_ids(), iterate_hust_cycles),
        ("XJTU", xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles),
    ]:
        datasets[name] = build_new_dataset_merged(name, ids, iterate_fn, norm_stats, encoder)
        print(f"[save-preds] {name}: {len(datasets[name])} rows, {datasets[name]['battery_id'].nunique()} cells")

    for name, df in datasets.items():
        pred = predict_df(df, cols)
        out = pd.DataFrame({
            "battery_id": df["battery_id"].to_numpy(),
            "y_true": df["SOH"].to_numpy(dtype=float),
            "pred": pred,
        })
        out_path = PRED_DIR / f"percycle_{name.lower()}{pool_suffix or '_deployed'}.csv"
        out.to_csv(out_path, index=False)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(out["y_true"], out["pred"])
        rmse = float(np.sqrt(mean_squared_error(out["y_true"], out["pred"])))
        print(f"[save-preds] {name}: R2={r2:.4f} RMSE={rmse:.4f} n={len(out)} -> saved {out_path.name}")

    print(f"\n[save-preds] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    pool_suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    pool_label = sys.argv[2] if len(sys.argv) > 2 else "deployed"
    main(pool_suffix, pool_label)
