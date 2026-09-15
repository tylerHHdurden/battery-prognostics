"""
Data-expansion pass, item 1: the designed test of the specialization
mechanism. Builds the NASA-heavy pool (see nasaheavy_tensors.py's
docstring for composition/reasoning: 57 NASA-family + 28 MIT = 85
batteries, 67.1% NASA), regenerates its hi_table, retrains XGBoost-
fusion through the exact same pipeline as the 204-battery pool run
(reusing its own fit_xgb_local/eval_xgb_local/build_new_dataset_merged
functions directly rather than re-deriving them - both known bugs
from that run, the inf-on-training-path crash and the CALCE-fusion-
lookup bug, are already fixed there and inherited fixed here), and
evaluates in-domain + zero-retrain on all four held-out sets.

NOT a deployment candidate regardless of outcome - a scientific test
of a mechanism, stated explicitly. Writes ONLY to its own, separate
files - does not touch any deployed file.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from nasaheavy_tensors import load_battery_tensors_nasaheavy, build_hi_rows_nasaheavy
from train_deep_models import make_xy
from sequence_features import compute_channel_norm_stats, apply_channel_norm
from models.ica_encoder import ICAEncoder
from stage1_common import canonical_feature_cols, add_reformulated_duration_features, OUT_DIR, PROC_DIR, ROOT
from run_pool204_step2_retrain_eval import fit_xgb_local, eval_xgb_local, build_new_dataset_merged, train_encoder
from run_stage5_1_new_datasets_eval import (
    oxford_cell_ids, iterate_oxford_cycles, hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids_soh_valid, iterate_xjtu_cycles,
)
from data_adapters import iterate_calce_cycles

EMBED_DIM = 16
ICA_CHANNEL_SLICE = slice(3, 6)
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]
RNG_SEED = 42


def main():
    t_start = time.time()
    print("=== NASA-heavy pool: feature regeneration ===")
    hi_df = build_hi_rows_nasaheavy()
    dup = hi_df.duplicated(subset=["dataset", "battery_id", "cycle_idx"]).sum()
    print(f"[nasaheavy] duplicate rows: {dup}")
    assert dup == 0, "duplicates found - not trusting this pool"
    n_batteries = hi_df.groupby("dataset")["battery_id"].nunique()
    print(f"[nasaheavy] pool composition: {n_batteries.to_dict()}, {len(hi_df)} total cycles")
    hi_df.to_parquet(PROC_DIR / "hi_table_nasaheavy.parquet")

    print("\n=== NASA-heavy pool: tensor loading + retrain ===")
    battery_data = load_battery_tensors_nasaheavy()
    all_ids = sorted(battery_data.keys())
    rng = np.random.default_rng(RNG_SEED)
    shuffled = list(all_ids)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * 0.2))
    test_ids = sorted(shuffled[:n_test])
    train_ids = sorted(shuffled[n_test:])
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[nasaheavy] fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)} "
          f"({len(battery_data)} total batteries)")
    with open(PROC_DIR / "battery_split_nasaheavy.json", "w") as f:
        json.dump({"train_ids": train_ids, "test_ids": test_ids,
                    "_note": "80/20 random split, seed=42, battery-level, constructed for this pool only"}, f, indent=2)

    all_ids_ordered = fit_ids + val_ids + test_ids
    X_all, y_all, rul_all, ds_all, bid_all, cyc_all = make_xy(battery_data, all_ids_ordered)
    n_fit = sum(len(battery_data[b][1]) for b in fit_ids)
    n_val_cyc = sum(len(battery_data[b][1]) for b in val_ids)
    X_fit_raw = X_all[:n_fit]

    print("\n[nasaheavy] === refitting channel_norm_stats (own file) ===")
    norm_stats = compute_channel_norm_stats(X_fit_raw)
    with open(PROC_DIR / "channel_norm_stats_nasaheavy.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    X_all_norm = apply_channel_norm(X_all, norm_stats)
    X_fit_ica = X_all_norm[:n_fit, :, ICA_CHANNEL_SLICE]
    X_val_ica = X_all_norm[n_fit:n_fit + n_val_cyc, :, ICA_CHANNEL_SLICE]
    y_fit, y_val = y_all[:n_fit], y_all[n_fit:n_fit + n_val_cyc]

    print("\n[nasaheavy] === training ICA fusion encoder (own file) ===")
    encoder, hist = train_encoder(X_fit_ica, y_fit, X_val_ica, y_val)
    torch.save(encoder.state_dict(), ROOT / "models" / "ica_encoder_nasaheavy.pt")

    print("\n[nasaheavy] === generating fusion embeddings for the full pool ===")
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder.encode(torch.tensor(X_all_norm[:, :, ICA_CHANNEL_SLICE])).numpy()
    fusion_df = pd.DataFrame({"dataset": ds_all, "battery_id": bid_all, "cycle_idx": cyc_all})
    for i in range(EMBED_DIM):
        fusion_df[f"fusion_{i}"] = embeddings[:, i]
    fusion_df.to_csv(PROC_DIR / "fusion_embeddings_nasaheavy.csv", index=False)
    fcols = [f"fusion_{i}" for i in range(EMBED_DIM)]

    print("\n[nasaheavy] === building HI-reformulated + fusion merged frame ===")
    hi_reformulated = add_reformulated_duration_features(hi_df)
    nasa_mit = hi_reformulated[hi_reformulated["dataset"].isin(["NASA", "NASA_RANDOMIZED", "MIT"])]
    merged = pd.merge(nasa_mit, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[nasaheavy] merged: {len(merged)} rows ({len(nasa_mit)} HI rows, {len(nasa_mit)-len(merged)} lost)")

    train_mask = merged["battery_id"].isin(train_ids).to_numpy()
    test_mask = merged["battery_id"].isin(test_ids).to_numpy()

    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    monotone = tuple([0] * (len(feature_cols) - 1) + [-1] + [0] * len(fcols))

    print("\n[nasaheavy] === retraining XGBoost-fusion (canonical 1.1+1.5 config) ===")
    model, medians, cols = fit_xgb_local(merged, train_mask, feature_cols, fcols,
                                          xgb_extra_kwargs={"monotone_constraints": monotone})
    model.save_model(str(ROOT / "models" / "xgb_soh_fusion_nasaheavy.json"))

    indomain = eval_xgb_local(model, medians, feature_cols, fcols, merged[test_mask])
    print(f"[nasaheavy] IN-DOMAIN (fixed 80/20 split): R2={indomain['r2']:.4f} RMSE={indomain['rmse']:.4f}")

    print("\n[nasaheavy] === GroupKFold(5) CV ===")
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
        print(f"[nasaheavy] fold {fold_i}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={em.sum()}")
        fold_rows.append({"fold": fold_i, "r2": r["r2"], "rmse": r["rmse"], "n_test": int(em.sum())})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "nasaheavy_groupkfold.csv", index=False)
    print(f"[nasaheavy] GroupKFold mean R2={fold_df['r2'].mean():.4f} (std {fold_df['r2'].std():.4f}) "
          f"mean RMSE={fold_df['rmse'].mean():.4f} (std {fold_df['rmse'].std():.4f})")

    print("\n=== zero-retrain eval on all 4 held-out datasets ===")
    summary_rows = [{"dataset": "in-domain (fixed 80/20 split)", "n_cycles": len(indomain["pred"]),
                      "r2": indomain["r2"], "rmse": indomain["rmse"]}]
    for name, ids, iterate_fn in [
        ("CALCE", CALCE_CELLS, iterate_calce_cycles),
        ("Oxford", oxford_cell_ids(), iterate_oxford_cycles),
        ("HUST", hust_cell_ids(), iterate_hust_cycles),
        ("XJTU", xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles),
    ]:
        m = build_new_dataset_merged(name, ids, iterate_fn, norm_stats, encoder)
        r = eval_xgb_local(model, medians, feature_cols, fcols, m)
        print(f"[nasaheavy] {name}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={len(r['pred'])} "
              f"({m['battery_id'].nunique()} cells)")
        summary_rows.append({"dataset": name, "n_cycles": len(r["pred"]), "r2": r["r2"], "rmse": r["rmse"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "nasaheavy_zero_retrain_eval.csv", index=False)
    print("\n=== FINAL SUMMARY (NASA-heavy pool) ===")
    print(summary.to_string(index=False))
    print(f"\n[nasaheavy] TOTAL TIME: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
