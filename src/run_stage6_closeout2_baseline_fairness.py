"""
Stage 6 closeout, item 2: 6.1 baseline-fairness self-audit. The most
consequential possible unfairness identified on inspection: Severson/
Attia's own published methods use a linear/elastic-net model (faithful
to their own papers - their contribution WAS "a few physics-motivated
features + a simple linear model"), while this project's deployed
model uses a gradient-boosted tree ensemble (XGBoost) PLUS a much
richer, learned feature set (8 domain-engineered HI ratios + 16
neural-encoder fusion embeddings vs. Severson/Attia's 1-4 hand-
computed features). Two different axes of advantage are entangled in
6.1's original comparison: model CLASS (linear vs. tree ensemble) and
FEATURE richness (few vs. many/learned).

This script isolates them: re-runs Severson's and Attia's OWN feature
sets through XGBoost (the SAME model class, SAME hyperparameters as
the deployed model) instead of ElasticNet - if the gap narrows a lot,
model class was doing much of the work; if it stays wide, the win is
genuinely about this project's own feature engineering, a fairer,
more specific claim for the paper. Cheap - reuses the already-computed
Severson/Attia feature parquets from 6.1, no feature recomputation.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import OUT_DIR, PROC_DIR, ROOT
from run_stage6_1_severson_attia_baselines import VARIANCE_FEATS, RICH_FEATS

# SAME hyperparameters as this project's own deployed XGBoost-fusion
# model (stage1_common.fit_xgb) - not a separately-tuned config, for a
# genuine apples-to-apples "same model class, same settings" test.
XGB_KWARGS = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, reg_lambda=1.0)


def fit_eval_xgb(train_df, test_df, feat_cols, label):
    X_train = train_df[feat_cols].to_numpy(dtype=float)
    y_train = train_df["SOH"].to_numpy(dtype=float)
    col_medians = np.nanmedian(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train = X_train.copy()
    X_train[inds] = np.take(col_medians, inds[1])

    model = XGBRegressor(**XGB_KWARGS)
    model.fit(X_train, y_train)

    X_test = test_df[feat_cols].to_numpy(dtype=float).copy()
    inds2 = np.where(np.isnan(X_test))
    X_test[inds2] = np.take(col_medians, inds2[1])
    pred = model.predict(X_test)
    y_test = test_df["SOH"].to_numpy(dtype=float)
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {"label": label, "r2": r2, "rmse": rmse, "n": len(y_test),
            "n_cells": test_df["battery_id"].nunique()}, model, col_medians


def main():
    t0 = time.time()
    print("=== Stage 6 closeout, item 2: baseline-fairness re-run (Severson/Attia features -> XGBoost) ===")
    pool_df = pd.read_parquet(PROC_DIR / "severson_features_42pool.parquet")
    calce_df = pd.read_parquet(PROC_DIR / "severson_features_calce.parquet")
    oxford_df = pd.read_parquet(PROC_DIR / "severson_features_oxford.parquet")
    hust_df = pd.read_parquet(PROC_DIR / "severson_features_hust.parquet")
    xjtu_df = pd.read_parquet(PROC_DIR / "severson_features_xjtu.parquet")
    held_out = {"CALCE": calce_df, "Oxford": oxford_df, "HUST": hust_df, "XJTU": xjtu_df}

    import json
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    train_mask = pool_df["battery_id"].isin(train_ids)
    test_mask = pool_df["battery_id"].isin(test_ids)
    train_df, test_df = pool_df[train_mask], pool_df[test_mask]

    results_rows = []
    for label, feat_cols in [("Severson variance model, XGBoost (fairness re-run)", VARIANCE_FEATS),
                              ("Attia-style rich model, XGBoost (fairness re-run)", RICH_FEATS)]:
        print(f"\n=== {label} ===")
        r_indomain, model, medians = fit_eval_xgb(train_df, test_df, feat_cols, f"{label} in-domain")
        print(f"[fairness] in-domain (fixed split): R2={r_indomain['r2']:.4f} RMSE={r_indomain['rmse']:.4f}")
        results_rows.append({"method": label, "eval_set": "in-domain (fixed split)", **{
            k: v for k, v in r_indomain.items() if k != "label"}})

        gkf = GroupKFold(n_splits=5)
        unique_b = pool_df["battery_id"].unique()
        fold_r2, fold_rmse = [], []
        for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(unique_b, groups=unique_b)):
            tb, eb = set(unique_b[tr_idx]), set(unique_b[te_idx])
            tr_df = pool_df[pool_df["battery_id"].isin(tb)]
            te_df = pool_df[pool_df["battery_id"].isin(eb)]
            r, _, _ = fit_eval_xgb(tr_df, te_df, feat_cols, f"{label} fold{fold_i}")
            fold_r2.append(r["r2"]); fold_rmse.append(r["rmse"])
        gkf_mean_r2, gkf_std_r2 = float(np.mean(fold_r2)), float(np.std(fold_r2))
        gkf_mean_rmse, gkf_std_rmse = float(np.mean(fold_rmse)), float(np.std(fold_rmse))
        print(f"[fairness] GroupKFold mean R2={gkf_mean_r2:.4f} (std {gkf_std_r2:.4f})")
        results_rows.append({"method": label, "eval_set": "in-domain (GroupKFold mean)",
                              "r2": gkf_mean_r2, "rmse": gkf_mean_rmse, "n": None, "n_cells": None,
                              "r2_std": gkf_std_r2, "rmse_std": gkf_std_rmse})

        for name, df in held_out.items():
            X = df[feat_cols].to_numpy(dtype=float).copy()
            inds = np.where(np.isnan(X))
            X[inds] = np.take(medians, inds[1])
            pred = model.predict(X)
            y_true = df["SOH"].to_numpy(dtype=float)
            r2 = r2_score(y_true, pred)
            rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
            print(f"[fairness] {name}: R2={r2:.4f} RMSE={rmse:.4f} n={len(df)}")
            results_rows.append({"method": label, "eval_set": name, "r2": r2, "rmse": rmse,
                                  "n": len(df), "n_cells": df["battery_id"].nunique()})

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(OUT_DIR / "stage6_closeout2_baseline_fairness_results.csv", index=False)
    print("\n=== FULL RESULTS ===")
    print(results_df.to_string(index=False))
    print(f"\n[fairness] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
