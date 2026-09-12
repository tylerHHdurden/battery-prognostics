"""
Dataset Expansion Phase 1: 5-branch drop-branch ablation on the expanded
pool - additive companion, mirrors run_drop_branch_ablation_5branch.py
exactly but reads the `*_expanded` prediction files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ensemble_fusion_expanded import load_merged_fusion_expanded

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"

BASE_COLS_5 = ["pred_XGBoost_fusion", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer", "pred_CNNBiGRU"]


def load_merged_fusion_5branch_expanded(split_name: str):
    merged, fusion_cols = load_merged_fusion_expanded(split_name)
    cnn_bigru_file = ("cnn_bigru_expanded_test_preds.csv" if split_name == "test"
                       else "cnn_bigru_expanded_train_preds.csv")
    cnn_bigru = pd.read_csv(PRED_DIR / cnn_bigru_file)
    merged5 = pd.merge(
        merged, cnn_bigru[["dataset", "battery_id", "cycle_idx", "y_pred_CNNBiGRU"]],
        on=["dataset", "battery_id", "cycle_idx"], how="inner",
    ).rename(columns={"y_pred_CNNBiGRU": "pred_CNNBiGRU"})
    print(f"[drop-branch-5-exp] {split_name}: 4-branch merged rows={len(merged)}, "
          f"CNN-BiGRU rows={len(cnn_bigru)}, 5-branch merged rows={len(merged5)}")
    return merged5, fusion_cols


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    train_df, fusion_cols = load_merged_fusion_5branch_expanded("train")
    test_df, _ = load_merged_fusion_5branch_expanded("test")

    rows = []
    full_cols = BASE_COLS_5 + fusion_cols
    ridge_full = Ridge(alpha=1.0).fit(train_df[full_cols], train_df["SOH"])
    pred_full = ridge_full.predict(test_df[full_cols])
    m_full = metrics(test_df["SOH"], pred_full)
    print(f"[drop-branch-5-exp] FULL (all 5 base learners): {m_full}")
    rows.append({"variant": "full_5_branch", "dropped": "none", **m_full,
                 "delta_rmse": 0.0, "delta_r2": 0.0})

    for dropped in BASE_COLS_5:
        cols = [c for c in BASE_COLS_5 if c != dropped] + fusion_cols
        ridge = Ridge(alpha=1.0).fit(train_df[cols], train_df["SOH"])
        pred = ridge.predict(test_df[cols])
        m = metrics(test_df["SOH"], pred)
        delta_rmse = m["rmse"] - m_full["rmse"]
        delta_r2 = m["r2"] - m_full["r2"]
        print(f"[drop-branch-5-exp] DROP {dropped}: {m} (delta RMSE={delta_rmse:+.4f}, "
              f"delta R2={delta_r2:+.4f})")
        rows.append({"variant": f"drop_{dropped}", "dropped": dropped, **m,
                     "delta_rmse": delta_rmse, "delta_r2": delta_r2})

    df = pd.DataFrame(rows).sort_values("rmse")
    df.to_csv(PRED_DIR / "drop_branch_ablation_5branch_expanded.csv", index=False)
    print("\n[drop-branch-5-exp] === FULL TABLE (sorted by RMSE) ===")
    print(df.to_string(index=False))
    print("[drop-branch-5-exp] DONE")


if __name__ == "__main__":
    main()
