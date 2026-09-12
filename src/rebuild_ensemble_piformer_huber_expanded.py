"""
Targeted improvement pass, Part 1, item 3: does the Huber-loss PiFormer
fix also fix the 4-branch ensemble's inherited RMSE regression?

Refits the Ridge meta-learner with PiFormer's predictions swapped from
the MSE-trained checkpoint to the Huber-trained one (VLSTM/CNN-LSTM/
XGBoost-fusion unchanged) - the exact train_ensemble_fusion_expanded.py
recipe, one column swapped. Additive: does not touch
ensemble_fusion_expanded_{metrics,test_preds}.csv or
ridge_meta_fusion_expanded.pkl.
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


def load_merged(split_name: str) -> tuple[pd.DataFrame, list[str]]:
    xgb = pd.read_csv(PRED_DIR / "xgb_fusion_expanded_preds.csv")
    xgb = xgb[xgb["split"] == split_name][
        ["dataset", "battery_id", "cycle_idx", "SOH", "RUL", "y_pred_soh_fusion"]
    ].rename(columns={"y_pred_soh_fusion": "pred_XGBoost_fusion"})

    deep_file = ("deep_models_expanded_test_preds.csv" if split_name == "test"
                 else "deep_models_expanded_train_preds.csv")
    deep = pd.read_csv(PRED_DIR / deep_file).rename(columns={
        "y_pred_VLSTM": "pred_VLSTM", "y_pred_CNNLSTM": "pred_CNNLSTM",
    })[["dataset", "battery_id", "cycle_idx", "pred_VLSTM", "pred_CNNLSTM"]]

    huber_file = ("piformer_huber_expanded_test_preds.csv" if split_name == "test"
                  else "piformer_huber_expanded_train_preds.csv")
    huber = pd.read_csv(PRED_DIR / huber_file).rename(
        columns={"y_pred_PiFormer_huber": "pred_PiFormer"}
    )[["dataset", "battery_id", "cycle_idx", "pred_PiFormer"]]

    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings_expanded.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]

    merged = pd.merge(xgb, deep, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    merged = pd.merge(merged, huber, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    merged = pd.merge(merged, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[ensemble-huber] {split_name}: merged rows={len(merged)}")
    return merged, fusion_cols


def main():
    train_df, fusion_cols = load_merged("train")
    test_df, _ = load_merged("test")

    base_cols = ["pred_XGBoost_fusion", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
    meta_cols = base_cols + fusion_cols

    X_train, y_train = train_df[meta_cols].to_numpy(), train_df["SOH"].to_numpy()
    X_test, y_test = test_df[meta_cols].to_numpy(), test_df["SOH"].to_numpy()

    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    pred_test = ridge.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))
    print(f"[ensemble-huber] Stacking-Ridge-fusion (PiFormer=Huber) TEST "
          f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    print(f"[ensemble-huber] (existing ensemble, PiFormer=MSE, was RMSE=1.4834 MAE=0.5276 R2=0.9579)")
    print(f"[ensemble-huber] (original 32-battery ensemble was RMSE=1.394 MAE=0.966 R2=0.917)")
    print(f"[ensemble-huber] Ridge coefficients: {dict(zip(base_cols, ridge.coef_[:4]))}")

    # Per-battery breakdown - does swapping in Huber-PiFormer actually
    # fix B0053's contribution to the STACK's RMSE, or does the Ridge
    # meta-learner's small PiFormer weight (established in the Dataset
    # Expansion session as tiny: +0.1748) mean it barely matters here?
    test_df["pred_ensemble_huber"] = pred_test
    test_df["abs_err"] = (test_df["SOH"] - test_df["pred_ensemble_huber"]).abs()
    g = test_df.groupby(["dataset", "battery_id"]).agg(
        n=("SOH", "size"), rmse=("abs_err", lambda e: float(np.sqrt((e ** 2).mean())))
    ).reset_index().sort_values("rmse", ascending=False)
    print("\n[ensemble-huber] Per-battery ensemble RMSE, worst 5:")
    print(g.head(5).to_string(index=False))

    test_df.to_csv(PRED_DIR / "ensemble_fusion_piformerhuber_expanded_test_preds.csv", index=False)
    pd.DataFrame([{"model": "Stacking-Ridge-fusion-expanded-PiFormerHuber",
                    "rmse": rmse, "mae": mae, "r2": r2}]).to_csv(
        PRED_DIR / "ensemble_fusion_piformerhuber_expanded_metrics.csv", index=False
    )
    print("\n[ensemble-huber] DONE")


if __name__ == "__main__":
    main()
