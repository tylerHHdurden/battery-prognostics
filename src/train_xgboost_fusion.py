"""
Fusion-enabled XGBoost: same 7 BFA-selected Health Indicators as
train_xgboost.py, PLUS the 16-dim ICA/DV/DC fusion embedding
(data/processed/fusion_embeddings.csv) concatenated in as additional
features (23 total). Additive: does not modify or overwrite
train_xgboost.py / xgb_soh.json / xgb_preds.csv - this is a separate,
parallel script and output set so the original (non-fusion) pipeline
stays fully intact and reproducible.

Same battery-level split (data/processed/battery_split.json) as every
other base learner, for a fair comparison against the non-fusion
XGBoost's RMSE=1.478/MAE=0.990/R2=0.907 reported in FINAL_SUMMARY.md.
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
    df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    df = df[df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)

    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        selected = [l.strip() for l in f if l.strip()]
    print(f"[xgb-fusion] BFA-selected HIs: {selected}")

    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]
    print(f"[xgb-fusion] fusion embedding dims: {len(fusion_cols)}")

    merged = pd.merge(df, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[xgb-fusion] hi_table rows={len(df)}, fusion rows={len(fusion)}, "
          f"merged (inner) rows={len(merged)}")

    feature_cols = selected + fusion_cols
    X = merged[feature_cols].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = merged["SOH"].to_numpy(dtype=float)

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    train_mask = merged["battery_id"].isin(train_ids).to_numpy()
    test_mask = ~train_mask
    print(f"[xgb-fusion] train rows={train_mask.sum()}, test rows={test_mask.sum()}")

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
    print(f"[xgb-fusion] TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    print(f"[xgb-fusion] (non-fusion XGBoost was RMSE=1.478 MAE=0.990 R2=0.907)")

    out = merged[["dataset", "battery_id", "cycle_idx", "SOH", "RUL"]].copy()
    out["y_pred_soh_fusion"] = np.nan
    out.loc[train_mask, "y_pred_soh_fusion"] = pred_train
    out.loc[test_mask, "y_pred_soh_fusion"] = pred_test
    out["split"] = np.where(train_mask, "train", "test")
    out.to_csv(PRED_DIR / "xgb_fusion_preds.csv", index=False)

    model.save_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    pd.DataFrame([{"model": "XGBoost-fusion", "rmse": rmse, "mae": mae, "r2": r2,
                   "n_features": len(feature_cols)}]).to_csv(
        PRED_DIR / "xgb_fusion_metrics.csv", index=False
    )
    print("[xgb-fusion] DONE")


if __name__ == "__main__":
    main()
