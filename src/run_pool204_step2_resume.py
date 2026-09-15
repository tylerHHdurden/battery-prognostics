"""
Resume of run_pool204_step2_retrain_eval.py after the first run
crashed at the XGBoost-fit step (a real bug - `inf` values not
sanitized on the training path, fixed in the main script too, see its
own comment). Loads the 3 expensive artifacts that DID complete and
save successfully before the crash (channel_norm_stats_pool204.json,
ica_encoder_pool204.pt, fusion_embeddings_pool204.csv - confirmed
present and correctly saved) rather than repeating the ~48-minute
tensor-loading + encoder-training work. Everything from the XGBoost
retrain onward is fast and re-run fresh here.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from models.ica_encoder import ICAEncoder
from stage1_common import canonical_feature_cols, add_reformulated_duration_features, OUT_DIR, PROC_DIR, ROOT
from run_pool204_step2_retrain_eval import fit_xgb_local, eval_xgb_local, build_new_dataset_merged
from run_stage5_1_new_datasets_eval import (
    oxford_cell_ids, iterate_oxford_cycles, hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids_soh_valid, iterate_xjtu_cycles,
)
from data_adapters import iterate_calce_cycles
from health_indicators import compute_health_indicators
from rul_labels import soh_per_cycle

EMBED_DIM = 16
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]


def main():
    t_start = time.time()
    print("=== Resuming Part A Step 2/3 from saved artifacts ===")
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_pool204.json").read_text())
    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder_pool204.pt"))
    encoder.eval()
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings_pool204.csv")
    fcols = [f"fusion_{i}" for i in range(EMBED_DIM)]

    split = json.loads((PROC_DIR / "battery_split_expanded_b0018pinned.json").read_text())
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]

    print("[pool204-resume] === building HI-reformulated + fusion merged frame ===")
    hi_full = pd.read_parquet(PROC_DIR / "hi_table_pool204.parquet")
    hi_reformulated = add_reformulated_duration_features(hi_full)
    nasa_mit = hi_reformulated[hi_reformulated["dataset"].isin(["NASA", "MIT"])]
    merged = pd.merge(nasa_mit, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[pool204-resume] merged: {len(merged)} rows")

    train_mask = merged["battery_id"].isin(train_ids).to_numpy()
    test_mask = merged["battery_id"].isin(test_ids).to_numpy()

    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    monotone = tuple([0] * (len(feature_cols) - 1) + [-1] + [0] * len(fcols))

    print("\n[pool204-resume] === retraining XGBoost-fusion (canonical 1.1+1.5 config) ===")
    model, medians, cols = fit_xgb_local(merged, train_mask, feature_cols, fcols,
                                          xgb_extra_kwargs={"monotone_constraints": monotone})
    model.save_model(str(ROOT / "models" / "xgb_soh_fusion_pool204.json"))

    indomain = eval_xgb_local(model, medians, feature_cols, fcols, merged[test_mask])
    print(f"[pool204-resume] IN-DOMAIN (fixed split): R2={indomain['r2']:.4f} RMSE={indomain['rmse']:.4f} "
          f"(Stage 4 42-battery: R2=0.9740 RMSE=0.7805)")

    print("\n[pool204-resume] === GroupKFold(5) CV ===")
    gkf = GroupKFold(n_splits=5)
    unique_batteries = merged["battery_id"].unique()
    fold_rows = []
    for fold_i, (train_bidx, test_bidx) in enumerate(gkf.split(unique_batteries, groups=unique_batteries)):
        tb = set(unique_batteries[train_bidx]); eb = set(unique_batteries[test_bidx])
        tm = merged["battery_id"].isin(tb).to_numpy()
        em = merged["battery_id"].isin(eb).to_numpy()
        m2, med2, c2 = fit_xgb_local(merged, tm, feature_cols, fcols,
                                      xgb_extra_kwargs={"monotone_constraints": monotone})
        r = eval_xgb_local(m2, med2, feature_cols, fcols, merged[em])
        print(f"[pool204-resume] fold {fold_i}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={em.sum()}")
        fold_rows.append({"fold": fold_i, "r2": r["r2"], "rmse": r["rmse"], "n_test": int(em.sum())})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "pool204_groupkfold.csv", index=False)
    print(f"[pool204-resume] GroupKFold mean R2={fold_df['r2'].mean():.4f} (std {fold_df['r2'].std():.4f}) "
          f"mean RMSE={fold_df['rmse'].mean():.4f} (std {fold_df['rmse'].std():.4f})")
    print(f"[pool204-resume] vs. Stage 4 42-battery GroupKFold: R2=0.9658 (std 0.0207) RMSE=1.1388 (std 0.5673)")

    print("\n=== Part A, Step 3: zero-retrain eval on all 4 held-out datasets ===")
    calce_rows = []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        soh_map = soh_per_cycle(cycles)
        for c in cycles:
            his = compute_health_indicators(c)
            row = {"dataset": "CALCE", "battery_id": cid, "cycle_idx": c["cycle_idx"], "SOH": soh_map[c["cycle_idx"]]}
            row.update(his)
            calce_rows.append(row)
    calce_hi = pd.DataFrame(calce_rows)
    calce_hi_ref = add_reformulated_duration_features(calce_hi)
    calce_fusion = fusion_df[fusion_df["dataset"] == "CALCE"]
    calce_merged = pd.merge(calce_hi_ref, calce_fusion, on=["battery_id", "cycle_idx"], how="inner")

    results = {"CALCE": eval_xgb_local(model, medians, feature_cols, fcols, calce_merged)}
    summary_rows = [{"dataset": "in-domain (fixed split)", "n_cycles": len(indomain["pred"]),
                      "r2": indomain["r2"], "rmse": indomain["rmse"]},
                     {"dataset": "CALCE", "n_cycles": len(results["CALCE"]["pred"]),
                      "r2": results["CALCE"]["r2"], "rmse": results["CALCE"]["rmse"]}]

    for name, ids, iterate_fn in [
        ("Oxford", oxford_cell_ids(), iterate_oxford_cycles),
        ("HUST", hust_cell_ids(), iterate_hust_cycles),
        ("XJTU", xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles),
    ]:
        m = build_new_dataset_merged(name, ids, iterate_fn, norm_stats, encoder)
        r = eval_xgb_local(model, medians, feature_cols, fcols, m)
        print(f"[pool204-resume] {name}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={len(r['pred'])}")
        results[name] = r
        summary_rows.append({"dataset": name, "n_cycles": len(r["pred"]), "r2": r["r2"], "rmse": r["rmse"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "pool204_zero_retrain_eval.csv", index=False)
    print("\n=== SUMMARY (204-battery pool, vs. Stage 4's 42-battery deployed model) ===")
    print(summary.to_string(index=False))
    print("\nStage 4 (42-battery, deployed) reference: in-domain R2=0.9740/0.9658(CV), "
          "CALCE=0.568, Oxford=-2.694, HUST=-0.152, XJTU=-1.059")
    print(f"\n[pool204-resume] TOTAL TIME: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
