"""
Final completion of Part A Step 3, after the resume run's CALCE eval
crashed on a real, root-caused bug: fusion_embeddings_pool204.csv only
ever contains NASA/MIT rows (pool204_tensors.py's training-pool loader
never includes CALCE, correctly - CALCE is never trained on), so
`fusion_df[fusion_df["dataset"]=="CALCE"]` was always empty and the
merge produced 0 rows. Fixed by computing CALCE's own fusion
embeddings the same way Oxford/HUST/XJTU's already correctly are
(build_new_dataset_merged - build CALCE's own tensors, apply the SAME
frozen channel_norm_stats_pool204, encode via the SAME trained
encoder), not by looking them up in a training-pool file that was
never going to contain them.

Reloads the already-trained, already-saved xgb_soh_fusion_pool204.json
(no retraining - GroupKFold's own 5 numbers are already complete and
saved) and recomputes ONLY the (cheap, deterministic) column medians
needed for imputation before evaluating all 4 held-out datasets
correctly.
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
from stage1_common import canonical_feature_cols, add_reformulated_duration_features, OUT_DIR, PROC_DIR, ROOT
from run_pool204_step2_retrain_eval import eval_xgb_local, build_new_dataset_merged
from run_stage5_1_new_datasets_eval import (
    oxford_cell_ids, iterate_oxford_cycles, hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids_soh_valid, iterate_xjtu_cycles,
)
from data_adapters import iterate_calce_cycles

EMBED_DIM = 16
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]


def main():
    t_start = time.time()
    print("=== Finishing Part A Step 3 (CALCE eval bug fixed) ===")
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_pool204.json").read_text())
    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder_pool204.pt"))
    encoder.eval()
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings_pool204.csv")
    fcols = [f"fusion_{i}" for i in range(EMBED_DIM)]

    split = json.loads((PROC_DIR / "battery_split_expanded_b0018pinned.json").read_text())
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]

    hi_full = pd.read_parquet(PROC_DIR / "hi_table_pool204.parquet")
    hi_reformulated = add_reformulated_duration_features(hi_full)
    nasa_mit = hi_reformulated[hi_reformulated["dataset"].isin(["NASA", "MIT"])]
    merged = pd.merge(nasa_mit, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    train_mask = merged["battery_id"].isin(train_ids).to_numpy()
    test_mask = merged["battery_id"].isin(test_ids).to_numpy()

    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    cols = feature_cols + fcols
    X = merged[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    medians = np.nanmedian(X[train_mask], axis=0)

    model = XGBRegressor()
    model.load_model(str(ROOT / "models" / "xgb_soh_fusion_pool204.json"))

    indomain = eval_xgb_local(model, medians, feature_cols, fcols, merged[test_mask])
    print(f"[pool204-finish] IN-DOMAIN (fixed split): R2={indomain['r2']:.4f} RMSE={indomain['rmse']:.4f} "
          f"(sanity check - must match the resume run's own 0.9966/0.4214)")
    assert abs(indomain["r2"] - 0.9966) < 0.01, "medians/model reconstruction mismatch!"

    fold_df = pd.read_csv(OUT_DIR / "pool204_groupkfold.csv")
    print(f"[pool204-finish] GroupKFold (already completed): mean R2={fold_df['r2'].mean():.4f} "
          f"(std {fold_df['r2'].std():.4f}) mean RMSE={fold_df['rmse'].mean():.4f} (std {fold_df['rmse'].std():.4f})")

    print("\n=== zero-retrain eval on all 4 held-out datasets (CALCE included correctly this time) ===")
    summary_rows = [{"dataset": "in-domain (fixed split)", "n_cycles": len(indomain["pred"]),
                      "r2": indomain["r2"], "rmse": indomain["rmse"]}]
    for name, ids, iterate_fn in [
        ("CALCE", CALCE_CELLS, iterate_calce_cycles),
        ("Oxford", oxford_cell_ids(), iterate_oxford_cycles),
        ("HUST", hust_cell_ids(), iterate_hust_cycles),
        ("XJTU", xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles),
    ]:
        m = build_new_dataset_merged(name, ids, iterate_fn, norm_stats, encoder)
        r = eval_xgb_local(model, medians, feature_cols, fcols, m)
        print(f"[pool204-finish] {name}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={len(r['pred'])} "
              f"({m['battery_id'].nunique()} cells)")
        summary_rows.append({"dataset": name, "n_cycles": len(r["pred"]), "r2": r["r2"], "rmse": r["rmse"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "pool204_zero_retrain_eval.csv", index=False)
    print("\n=== FINAL SUMMARY (204-battery pool, vs. Stage 4's 42-battery deployed model) ===")
    print(summary.to_string(index=False))
    print("\nStage 4 (42-battery, deployed) reference: in-domain R2=0.9740(fixed)/0.9658(CV), "
          "CALCE=0.568, Oxford=-2.694, HUST=-0.152, XJTU=-1.059")
    print(f"\n[pool204-finish] TOTAL TIME: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
