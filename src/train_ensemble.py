"""
Phase 3: stacking ensemble over the 4 base learners' SOH predictions.

Meta-features = [xgb_pred, vlstm_pred, cnn_lstm_pred, piformer_pred].
Two meta-learners tried: Ridge regression (the task's primary ask) and a
small XGBoost (n_estimators=50, max_depth=2 - deliberately shallow/few
trees since the meta-learner only has 4 input features and a few thousand
rows; a full-depth/full-tree-count XGBoost here would just overfit to
which base learner "won" on which few batteries).

Base-learner prediction tables are merged on (dataset, battery_id,
cycle_idx). This is an INNER join, not outer: sequence_features drops a
few cycles that health_indicators keeps (e.g. cycles too short for the
Savitzky-Golay window in ica_dv_dc), so the merge naturally restricts the
ensemble to cycles all 4 base learners actually scored. Row-count drop
from this is logged below rather than silently absorbed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"


def load_merged(split_name: str) -> pd.DataFrame:
    xgb = pd.read_csv(PRED_DIR / "xgb_preds.csv")
    xgb = xgb[xgb["split"] == split_name][
        ["dataset", "battery_id", "cycle_idx", "SOH", "RUL", "y_pred_soh"]
    ].rename(columns={"y_pred_soh": "pred_XGBoost"})

    deep_file = "deep_models_test_preds.csv" if split_name == "test" else "deep_models_train_preds.csv"
    deep = pd.read_csv(PRED_DIR / deep_file).rename(columns={
        "y_pred_VLSTM": "pred_VLSTM", "y_pred_CNNLSTM": "pred_CNNLSTM",
        "y_pred_PiFormer": "pred_PiFormer",
    })

    merged = pd.merge(
        xgb, deep[["dataset", "battery_id", "cycle_idx", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]],
        on=["dataset", "battery_id", "cycle_idx"], how="inner",
    )
    print(f"[ensemble] {split_name}: xgb rows={len(xgb)}, deep rows={len(deep)}, "
          f"merged (inner) rows={len(merged)}")
    return merged


def main():
    train_df = load_merged("train")
    test_df = load_merged("test")

    base_cols = ["pred_XGBoost", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
    X_train, y_train = train_df[base_cols].to_numpy(), train_df["SOH"].to_numpy()
    X_test, y_test = test_df[base_cols].to_numpy(), test_df["SOH"].to_numpy()

    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    print(f"[ensemble] Ridge meta-learner coefficients: "
          f"{dict(zip(base_cols, ridge.coef_.round(3)))}, intercept={ridge.intercept_:.3f}")

    xgb_meta = XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.1,
                             random_state=42, n_jobs=-1)
    xgb_meta.fit(X_train, y_train)
    xgb_meta_pred = xgb_meta.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    rows = []
    for col in base_cols:
        m = metrics(y_test, test_df[col].to_numpy())
        rows.append({"model": col.replace("pred_", ""), **m})
    rows.append({"model": "Stacking-Ridge", **metrics(y_test, ridge_pred)})
    rows.append({"model": "Stacking-XGBoost", **metrics(y_test, xgb_meta_pred)})

    comparison = pd.DataFrame(rows).sort_values("rmse")
    print("\n[ensemble] === FULL COMPARISON TABLE (test set, SOH%) ===")
    print(comparison.to_string(index=False))

    comparison.to_csv(PROC_DIR / "predictions" / "ensemble_comparison.csv", index=False)

    test_df["pred_Stacking_Ridge"] = ridge_pred
    test_df["pred_Stacking_XGBoost"] = xgb_meta_pred
    test_df.to_csv(PROC_DIR / "predictions" / "ensemble_test_preds.csv", index=False)

    import pickle
    with open(ROOT / "models" / "ridge_meta.pkl", "wb") as f:
        pickle.dump(ridge, f)
    xgb_meta.save_model(str(ROOT / "models" / "xgb_meta.json"))

    print("[ensemble] DONE")


if __name__ == "__main__":
    main()
