"""
Experiment 2/3 (remaining evaluation protocol): drop-one-branch ablation.

For each of the 4 base learners (XGBoost-fusion, VLSTM, CNN-LSTM,
PiFormer), refits ONLY the Ridge meta-learner with that one base
learner's prediction column removed (the other 3 base predictions + all
16 fusion-embedding dims kept), and measures the resulting test-set
performance drop vs. the full 4-branch ensemble (R2=0.917). No base
learner is retrained - this measures the meta-learner's reliance on
each branch's prediction, not each base learner's standalone quality
(that's already covered by the base-learner comparison table in
FINAL_SUMMARY.md).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ensemble_fusion import load_merged_fusion

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"

BASE_COLS = ["pred_XGBoost_fusion", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    train_df, fusion_cols = load_merged_fusion("train")
    test_df, _ = load_merged_fusion("test")

    rows = []

    # full 4-branch baseline (re-fit here rather than reusing the saved
    # model, so it's evaluated identically to the ablated variants below)
    full_cols = BASE_COLS + fusion_cols
    ridge_full = Ridge(alpha=1.0).fit(train_df[full_cols], train_df["SOH"])
    pred_full = ridge_full.predict(test_df[full_cols])
    m_full = metrics(test_df["SOH"], pred_full)
    print(f"[drop-branch] FULL (all 4 base learners): {m_full}")
    rows.append({"variant": "full_4_branch", "dropped": "none", **m_full,
                 "delta_rmse": 0.0, "delta_r2": 0.0})

    for dropped in BASE_COLS:
        cols = [c for c in BASE_COLS if c != dropped] + fusion_cols
        ridge = Ridge(alpha=1.0).fit(train_df[cols], train_df["SOH"])
        pred = ridge.predict(test_df[cols])
        m = metrics(test_df["SOH"], pred)
        delta_rmse = m["rmse"] - m_full["rmse"]
        delta_r2 = m["r2"] - m_full["r2"]
        print(f"[drop-branch] DROP {dropped}: {m} (delta RMSE={delta_rmse:+.4f}, "
              f"delta R2={delta_r2:+.4f})")
        rows.append({"variant": f"drop_{dropped}", "dropped": dropped, **m,
                     "delta_rmse": delta_rmse, "delta_r2": delta_r2})

    df = pd.DataFrame(rows).sort_values("rmse")
    df.to_csv(PRED_DIR / "drop_branch_ablation.csv", index=False)
    print("\n[drop-branch] === FULL TABLE (sorted by RMSE) ===")
    print(df.to_string(index=False))
    print("[drop-branch] DONE")


if __name__ == "__main__":
    main()
