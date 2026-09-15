"""
Stage 6.1 (MANDATORY): real, implemented baselines - Severson et al.
2019's own published variance-model method, and a richer Attia-et-al.-
2020-style extension - trained on the EXACT SAME 42-battery pool as
this project's deployed model, evaluated on the EXACT SAME GroupKFold
splits and the same 4 held-out zero-retrain datasets.

"Deployed model" comparison point, per this stage's own canonical-
configuration instruction (Stage 4's 42-battery pool + Stage 1.1 +
Stage 5's extended SCV/MATD/VIECT/MET reformulation + Stage 1.5
monotone constraints): this is the model already trained and saved as
models/_experimental_xgb_soh_fusion_extended_reformulation.json
(Stage 5 follow-on work) - NOT literally what's running in the live
Streamlit app right now (still Stage-1.1-only reformulation, since the
extended-reformulation model was never promoted, due to its own
XJTU regression - see that stage's entry). Stated explicitly here per
instruction, not glossed over: this stage's "deployed XGBoost-fusion"
comparison numbers are this experimental-but-canonical-for-Stage-6
model's own numbers (already computed and saved in outputs/stage5_
extended_reformulation_eval.csv), reused directly rather than retrained.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles, iterate_calce_cycles
from rul_labels import soh_per_cycle
from severson_features import build_severson_rows
from stage4_recovered_batteries import get_recovered_battery, ALL_RECOVERED
from run_stage5_1_new_datasets_eval import (
    oxford_cell_ids, iterate_oxford_cycles, hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids_soh_valid, iterate_xjtu_cycles,
)

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]

VARIANCE_FEATS = ["log_var_dq"]
RICH_FEATS = ["log_var_dq", "min_dq", "skew_dq", "early_fade_slope"]


def build_42pool_severson_df():
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    recovered_ids = {bid for _, bid in ALL_RECOVERED}

    all_rows = []
    for cid in NASA_CELLS:
        if cid in recovered_ids:
            continue
        cycles = list(iterate_nasa_cycles(cid))
        soh_map = soh_per_cycle(cycles)
        all_rows += build_severson_rows("NASA", cid, cycles, soh_map)
        print(f"[severson-feat] NASA/{cid}: {len(cycles)} cycles")

    for entry in mit_subset:
        cid = entry["global_id"]
        if cid in recovered_ids:
            continue
        cycles = list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))
        soh_map = soh_per_cycle(cycles)
        all_rows += build_severson_rows("MIT", cid, cycles, soh_map)
        print(f"[severson-feat] MIT/{cid}: {len(cycles)} cycles")

    for ds, bid in ALL_RECOVERED:
        kept_cycles, soh_map, rul_map, eol_cycle, censored = get_recovered_battery(ds, bid)
        all_rows += build_severson_rows(ds, bid, kept_cycles, soh_map)
        print(f"[severson-feat] RECOVERED {ds}/{bid}: {len(kept_cycles)} cycles")

    return pd.DataFrame(all_rows)


def build_dataset_severson_df(name, cell_ids, iterate_fn):
    rows = []
    for cid in cell_ids:
        cycles = list(iterate_fn(cid))
        if len(cycles) < 5:
            continue
        soh_map = soh_per_cycle(cycles)
        rows += build_severson_rows(name, cid, cycles, soh_map)
    return pd.DataFrame(rows)


def fit_eval(train_df, test_df, feat_cols, label):
    X_train = train_df[feat_cols].to_numpy(dtype=float)
    y_train = train_df["SOH"].to_numpy(dtype=float)
    col_medians = np.nanmedian(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train = X_train.copy()
    X_train[inds] = np.take(col_medians, inds[1])

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=5000, random_state=42)
    model.fit(X_train_s, y_train)

    X_test = test_df[feat_cols].to_numpy(dtype=float).copy()
    inds2 = np.where(np.isnan(X_test))
    X_test[inds2] = np.take(col_medians, inds2[1])
    X_test_s = scaler.transform(X_test)
    pred = model.predict(X_test_s)
    y_test = test_df["SOH"].to_numpy(dtype=float)
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {"label": label, "r2": r2, "rmse": rmse, "n": len(y_test),
            "n_cells": test_df["battery_id"].nunique()}, model, scaler, col_medians


def main():
    t0 = time.time()
    print("=== Stage 6.1: building Severson-style features for the 42-battery pool ===")
    pool_df = build_42pool_severson_df()
    pool_df.to_parquet(PROC_DIR / "severson_features_42pool.parquet")
    print(f"[severson] pool: {len(pool_df)} rows, {pool_df['battery_id'].nunique()} batteries, "
          f"NaN log_var_dq rows: {pool_df['log_var_dq'].isna().sum()}")

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    train_mask = pool_df["battery_id"].isin(train_ids)
    test_mask = pool_df["battery_id"].isin(test_ids)
    train_df, test_df = pool_df[train_mask], pool_df[test_mask]
    print(f"[severson] fixed split: train={len(train_df)} rows/{train_df['battery_id'].nunique()} batt, "
          f"test={len(test_df)} rows/{test_df['battery_id'].nunique()} batt")

    print("\n=== held-out datasets: building Severson-style features ===")
    calce_df = build_dataset_severson_df("CALCE", CALCE_CELLS, iterate_calce_cycles)
    oxford_df = build_dataset_severson_df("Oxford", oxford_cell_ids(), iterate_oxford_cycles)
    hust_df = build_dataset_severson_df("HUST", hust_cell_ids(), iterate_hust_cycles)
    xjtu_df = build_dataset_severson_df("XJTU", xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles)
    for name, df in [("CALCE", calce_df), ("Oxford", oxford_df), ("HUST", hust_df), ("XJTU", xjtu_df)]:
        print(f"[severson] {name}: {len(df)} rows, {df['battery_id'].nunique()} cells")
        df.to_parquet(PROC_DIR / f"severson_features_{name.lower()}.parquet")

    held_out = {"CALCE": calce_df, "Oxford": oxford_df, "HUST": hust_df, "XJTU": xjtu_df}

    results_rows = []
    for label, feat_cols in [("Severson variance model (1 feat)", VARIANCE_FEATS),
                              ("Attia-style rich model (4 feat)", RICH_FEATS)]:
        print(f"\n=== {label} ===")
        r_indomain, model, scaler, medians = fit_eval(train_df, test_df, feat_cols, f"{label} in-domain")
        print(f"[severson] in-domain (fixed split): R2={r_indomain['r2']:.4f} RMSE={r_indomain['rmse']:.4f}")
        results_rows.append({"method": label, "eval_set": "in-domain (fixed split)", **{
            k: v for k, v in r_indomain.items() if k != "label"}})

        print(f"[severson] === GroupKFold(5) for {label} ===")
        gkf = GroupKFold(n_splits=5)
        unique_b = pool_df["battery_id"].unique()
        fold_r2, fold_rmse = [], []
        for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(unique_b, groups=unique_b)):
            tb, eb = set(unique_b[tr_idx]), set(unique_b[te_idx])
            tr_df = pool_df[pool_df["battery_id"].isin(tb)]
            te_df = pool_df[pool_df["battery_id"].isin(eb)]
            r, _, _, _ = fit_eval(tr_df, te_df, feat_cols, f"{label} fold{fold_i}")
            print(f"    fold {fold_i}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={r['n']}")
            fold_r2.append(r["r2"]); fold_rmse.append(r["rmse"])
        gkf_mean_r2, gkf_std_r2 = float(np.mean(fold_r2)), float(np.std(fold_r2))
        gkf_mean_rmse, gkf_std_rmse = float(np.mean(fold_rmse)), float(np.std(fold_rmse))
        print(f"[severson] GroupKFold mean R2={gkf_mean_r2:.4f} (std {gkf_std_r2:.4f}) "
              f"mean RMSE={gkf_mean_rmse:.4f} (std {gkf_std_rmse:.4f})")
        results_rows.append({"method": label, "eval_set": "in-domain (GroupKFold mean)",
                              "r2": gkf_mean_r2, "rmse": gkf_mean_rmse, "n": None, "n_cells": None,
                              "r2_std": gkf_std_r2, "rmse_std": gkf_std_rmse})

        for name, df in held_out.items():
            X = df[feat_cols].to_numpy(dtype=float).copy()
            inds = np.where(np.isnan(X))
            X[inds] = np.take(medians, inds[1])
            Xs = scaler.transform(X)
            pred = model.predict(Xs)
            y_true = df["SOH"].to_numpy(dtype=float)
            r2 = r2_score(y_true, pred)
            rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
            print(f"[severson] {name}: R2={r2:.4f} RMSE={rmse:.4f} n={len(df)}")
            results_rows.append({"method": label, "eval_set": name, "r2": r2, "rmse": rmse,
                                  "n": len(df), "n_cells": df["battery_id"].nunique()})

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(OUT_DIR / "stage6_1_severson_attia_results.csv", index=False)
    print("\n=== FULL RESULTS ===")
    print(results_df.to_string(index=False))
    print(f"\n[severson] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
