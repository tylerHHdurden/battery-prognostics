"""
Base learner 1/4: XGBoost on the BFA-selected HI subset.

Per the task ("train all four base learners ... on NASA + MIT"), this
trains/evaluates only on NASA+MIT rows of hi_table.parquet, even though
BFA feature selection (Phase 1) ran on the full NASA+CALCE+MIT pool.
Target: SOH (%). Split: battery-level (src/split_utils.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import battery_level_split

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(exist_ok=True)
(PROC_DIR / "predictions").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)


def main():
    df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    df = df[df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)

    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        selected = [l.strip() for l in f if l.strip()]
    print(f"[xgb] using BFA-selected features: {selected}")

    X = df[selected].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = df["SOH"].to_numpy(dtype=float)

    import json
    split_path = PROC_DIR / "battery_split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        train_ids, test_ids = split["train_ids"], split["test_ids"]
        print("[xgb] reusing existing battery_split.json (shared across all base learners)")
    else:
        dataset_of = dict(zip(df["battery_id"], df["dataset"]))
        train_ids, test_ids = battery_level_split(df["battery_id"].tolist(), dataset_of=dataset_of)
        split_path.write_text(json.dumps({"train_ids": train_ids, "test_ids": test_ids}, indent=2))
    print(f"[xgb] train batteries ({len(train_ids)}): {train_ids}")
    print(f"[xgb] test batteries ({len(test_ids)}): {test_ids}")

    train_mask = df["battery_id"].isin(train_ids).to_numpy()
    test_mask = ~train_mask

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
    print(f"[xgb] TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    out = df[["dataset", "battery_id", "cycle_idx", "SOH", "RUL"]].copy()
    out["y_pred_soh"] = np.nan
    out.loc[train_mask, "y_pred_soh"] = pred_train
    out.loc[test_mask, "y_pred_soh"] = pred_test
    out["split"] = np.where(train_mask, "train", "test")
    out.to_csv(PROC_DIR / "predictions" / "xgb_preds.csv", index=False)

    model.save_model(str(ROOT / "models" / "xgb_soh.json"))

    metrics = pd.DataFrame([{"model": "XGBoost", "rmse": rmse, "mae": mae, "r2": r2}])
    metrics.to_csv(PROC_DIR / "predictions" / "xgb_metrics.csv", index=False)
    print("[xgb] DONE")


if __name__ == "__main__":
    main()
