"""
Dataset Expansion Phase 1: fusion-enabled XGBoost retrained on the
expanded pool (7 BFA-selected HIs from the EXPANDED BFA re-run + the
16-dim expanded fusion embedding, 23 features total, same as the
original). Additive - train_xgboost_fusion.py / xgb_soh_fusion.json /
xgb_fusion_preds.csv untouched.
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


def main():
    df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    df = df[df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)

    with open(PROC_DIR / "bfa_selected_features_expanded.txt") as f:
        selected = [l.strip() for l in f if l.strip()]
    print(f"[xgb-fusion-exp] expanded-BFA-selected HIs: {selected}")

    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings_expanded.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]

    merged = pd.merge(df, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[xgb-fusion-exp] hi_table rows={len(df)}, fusion rows={len(fusion)}, "
          f"merged rows={len(merged)}")

    feature_cols = selected + fusion_cols
    X = merged[feature_cols].to_numpy(dtype=float, copy=True)
    X[np.isinf(X)] = np.nan  # see run_bfa_expanded.py's comment: one MIT/b2c30 cycle's VDEDT is inf
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = merged["SOH"].to_numpy(dtype=float)

    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    train_mask = merged["battery_id"].isin(train_ids).to_numpy()
    test_mask = ~train_mask
    print(f"[xgb-fusion-exp] train rows={train_mask.sum()}, test rows={test_mask.sum()}")

    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, reg_lambda=1.0,
    )
    model.fit(X[train_mask], y[train_mask])

    pred_test = model.predict(X[test_mask])
    pred_train = model.predict(X[train_mask])

    rmse = np.sqrt(mean_squared_error(y[test_mask], pred_test))
    mae = mean_absolute_error(y[test_mask], pred_test)
    r2 = r2_score(y[test_mask], pred_test)
    print(f"[xgb-fusion-exp] TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    print("[xgb-fusion-exp] (original 32-battery XGBoost-fusion was RMSE=1.392 MAE=0.959 R2=0.917)")

    out = merged[["dataset", "battery_id", "cycle_idx", "SOH", "RUL"]].copy()
    out["y_pred_soh_fusion"] = np.nan
    out.loc[train_mask, "y_pred_soh_fusion"] = pred_train
    out.loc[test_mask, "y_pred_soh_fusion"] = pred_test
    out["split"] = np.where(train_mask, "train", "test")
    out.to_csv(PRED_DIR / "xgb_fusion_expanded_preds.csv", index=False)

    model.save_model(str(ROOT / "models" / "xgb_soh_fusion_expanded.json"))
    pd.DataFrame([{"model": "XGBoost-fusion-expanded", "rmse": rmse, "mae": mae, "r2": r2,
                   "n_features": len(feature_cols)}]).to_csv(
        PRED_DIR / "xgb_fusion_expanded_metrics.csv", index=False
    )
    print("[xgb-fusion-exp] DONE")


if __name__ == "__main__":
    main()
