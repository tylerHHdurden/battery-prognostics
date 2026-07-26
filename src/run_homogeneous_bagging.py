"""
Experiment 3/3 (remaining evaluation protocol): homogeneous-bagging
baseline. Trains XGBoost-fusion 5 times on the identical train data and
feature set (7 BFA HIs + 16 fusion dims), differing only in
random_state, averages the 5 models' test predictions, and compares
against (a) a single XGBoost-fusion fit and (b) the heterogeneous
Stacking-Ridge-fusion ensemble - to test whether simply bagging 5 seeds
of the SAME model type captures anything like the (in this project's
case, minimal - see the drop-branch ablation) benefit the heterogeneous
ensemble was hoped to provide.

Same hyperparameters as train_xgboost_fusion.py's single fit, so any
difference is attributable to the seed-averaging itself, not a
different model configuration.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"

SEEDS = [42, 0, 1, 7, 123]


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    df = df[df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        selected = [l.strip() for l in f if l.strip()]
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]
    merged = pd.merge(df, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")

    feature_cols = selected + fusion_cols
    X = merged[feature_cols].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = merged["SOH"].to_numpy(dtype=float)

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_mask = merged["battery_id"].isin(split["train_ids"]).to_numpy()
    test_mask = ~train_mask

    all_preds = []
    single_metrics = None
    for i, seed in enumerate(SEEDS):
        model = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=-1, reg_lambda=1.0,
        )
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        all_preds.append(pred)
        m = metrics(y[test_mask], pred)
        print(f"[bagging] seed={seed}: {m}")
        if i == 0:
            single_metrics = m  # seed=42, matches the original single-fit model

    bagged_pred = np.mean(all_preds, axis=0)
    bagged_metrics = metrics(y[test_mask], bagged_pred)
    print(f"\n[bagging] Homogeneous bag of {len(SEEDS)} XGBoost seeds: {bagged_metrics}")
    print(f"[bagging] Single XGBoost-fusion (seed=42): {single_metrics}")

    ridge_metrics_path = PRED_DIR / "ensemble_fusion_metrics.csv"
    ridge_row = pd.read_csv(ridge_metrics_path).iloc[0]
    print(f"[bagging] Heterogeneous Stacking-Ridge-fusion (reference): "
          f"rmse={ridge_row['rmse']:.4f} mae={ridge_row['mae']:.4f} r2={ridge_row['r2']:.4f}")

    rows = [
        {"model": "Single XGBoost-fusion (seed=42)", **single_metrics},
        {"model": f"Homogeneous bag ({len(SEEDS)} XGBoost seeds)", **bagged_metrics},
        {"model": "Heterogeneous Stacking-Ridge-fusion", "rmse": ridge_row["rmse"],
         "mae": ridge_row["mae"], "r2": ridge_row["r2"]},
    ]
    out = pd.DataFrame(rows).sort_values("rmse")
    out.to_csv(PRED_DIR / "homogeneous_bagging_comparison.csv", index=False)
    print("\n[bagging] === COMPARISON TABLE ===")
    print(out.to_string(index=False))
    print("[bagging] DONE")


if __name__ == "__main__":
    main()
