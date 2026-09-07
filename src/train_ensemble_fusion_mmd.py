"""
MMD-aligned counterpart of train_ensemble_fusion.py: identical stacking
setup (pred_XGBoost_fusion + VLSTM/CNNLSTM/PiFormer + 16-dim fusion
embedding as Ridge meta-features), but sourced entirely from the MMD
variant's files (xgb_fusion_mmd_preds.csv, fusion_embeddings_mmd.csv).
Additive: writes a parallel `*_fusion_mmd` file set, does not touch the
non-MMD ensemble outputs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"


def load_merged_fusion_mmd(split_name: str) -> pd.DataFrame:
    xgb = pd.read_csv(PRED_DIR / "xgb_fusion_mmd_preds.csv")
    xgb = xgb[xgb["split"] == split_name][
        ["dataset", "battery_id", "cycle_idx", "SOH", "RUL", "y_pred_soh_fusion"]
    ].rename(columns={"y_pred_soh_fusion": "pred_XGBoost_fusion"})

    deep_file = "deep_models_test_preds.csv" if split_name == "test" else "deep_models_train_preds.csv"
    deep = pd.read_csv(PRED_DIR / deep_file).rename(columns={
        "y_pred_VLSTM": "pred_VLSTM", "y_pred_CNNLSTM": "pred_CNNLSTM",
        "y_pred_PiFormer": "pred_PiFormer",
    })

    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings_mmd.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]

    merged = pd.merge(
        xgb, deep[["dataset", "battery_id", "cycle_idx", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]],
        on=["dataset", "battery_id", "cycle_idx"], how="inner",
    )
    merged = pd.merge(merged, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[ensemble-fusion-mmd] {split_name}: merged rows={len(merged)}")
    return merged, fusion_cols


def main():
    train_df, fusion_cols = load_merged_fusion_mmd("train")
    test_df, _ = load_merged_fusion_mmd("test")

    base_cols = ["pred_XGBoost_fusion", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
    meta_cols = base_cols + fusion_cols
    print(f"[ensemble-fusion-mmd] meta-learner input dims: {len(meta_cols)}")

    X_train, y_train = train_df[meta_cols].to_numpy(), train_df["SOH"].to_numpy()
    X_test, y_test = test_df[meta_cols].to_numpy(), test_df["SOH"].to_numpy()

    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    pred_test = ridge.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))
    print(f"[ensemble-fusion-mmd] Stacking-Ridge-fusion-MMD TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    print(f"[ensemble-fusion-mmd] (non-MMD Stacking-Ridge-fusion was R2=0.917)")

    test_df["pred_Stacking_Ridge_fusion"] = pred_test
    test_df.to_csv(PRED_DIR / "ensemble_fusion_mmd_test_preds.csv", index=False)
    pd.DataFrame([{"model": "Stacking-Ridge-fusion-MMD", "rmse": rmse, "mae": mae, "r2": r2,
                   "n_meta_features": len(meta_cols)}]).to_csv(
        PRED_DIR / "ensemble_fusion_mmd_metrics.csv", index=False
    )

    import pickle
    with open(ROOT / "models" / "ridge_meta_fusion_mmd.pkl", "wb") as f:
        pickle.dump(ridge, f)

    print("[ensemble-fusion-mmd] DONE")


if __name__ == "__main__":
    main()
