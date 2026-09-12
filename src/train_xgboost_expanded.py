"""
Dataset Expansion Phase 1: XGBoost base learner retrained on the
expanded NASA+MIT pool (hi_table_expanded.parquet, ~219 batteries vs.
the original 32). Additive - train_xgboost.py / battery_split.json /
xgb_soh.json / xgb_preds.csv are all completely untouched; this script
writes its own parallel `*_expanded` files.

Split: battery_level_split from split_utils.py, REUSED UNCHANGED (same
per-dataset stratification that caught and fixed the original
"zero NASA batteries in test set" bug) - computed fresh here from the
expanded battery/dataset list and saved to battery_split_expanded.json,
never overwriting the original battery_split.json every other existing
script still depends on.
"""

import json
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
PRED_DIR = PROC_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    df = df[df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    print(f"[xgb-exp] {len(df)} rows, {df['battery_id'].nunique()} batteries "
          f"({df[df.dataset=='NASA']['battery_id'].nunique()} NASA, "
          f"{df[df.dataset=='MIT']['battery_id'].nunique()} MIT)")

    with open(PROC_DIR / "bfa_selected_features_expanded.txt") as f:
        selected = [l.strip() for l in f if l.strip()]
    print(f"[xgb-exp] using expanded-BFA-selected features: {selected}")

    X = df[selected].to_numpy(dtype=float, copy=True)
    X[np.isinf(X)] = np.nan  # see run_bfa_expanded.py's comment: one MIT/b2c30 cycle's VDEDT is inf
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = df["SOH"].to_numpy(dtype=float)

    split_path = PROC_DIR / "battery_split_expanded.json"
    dataset_of = dict(zip(df["battery_id"], df["dataset"]))
    train_ids, test_ids = battery_level_split(df["battery_id"].tolist(), dataset_of=dataset_of)
    split_path.write_text(json.dumps({"train_ids": train_ids, "test_ids": test_ids}, indent=2))
    n_nasa_train = sum(1 for b in train_ids if dataset_of[b] == "NASA")
    n_nasa_test = sum(1 for b in test_ids if dataset_of[b] == "NASA")
    print(f"[xgb-exp] train batteries ({len(train_ids)}): {n_nasa_train} NASA, "
          f"{len(train_ids)-n_nasa_train} MIT")
    print(f"[xgb-exp] test batteries ({len(test_ids)}): {n_nasa_test} NASA, "
          f"{len(test_ids)-n_nasa_test} MIT")
    assert n_nasa_train > 0 and n_nasa_test > 0, \
        "NASA must be represented in BOTH splits - the exact bug caught in the original build"

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
    print(f"[xgb-exp] TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} "
          f"(n_test_cycles={test_mask.sum()}, n_train_cycles={train_mask.sum()})")

    model.save_model(str(ROOT / "models" / "xgb_soh_expanded.json"))
    pd.DataFrame([{"model": "XGBoost-expanded", "rmse": rmse, "mae": mae, "r2": r2,
                    "n_train_cycles": int(train_mask.sum()), "n_test_cycles": int(test_mask.sum()),
                    "n_train_batteries": len(train_ids), "n_test_batteries": len(test_ids)}]
                 ).to_csv(PRED_DIR / "xgb_expanded_metrics.csv", index=False)

    out = df.loc[test_mask, ["dataset", "battery_id", "cycle_idx", "SOH", "RUL"]].copy()
    out["y_pred_soh"] = pred_test
    out["split"] = "test"
    out_train = df.loc[train_mask, ["dataset", "battery_id", "cycle_idx", "SOH", "RUL"]].copy()
    out_train["y_pred_soh"] = pred_train
    out_train["split"] = "train"
    pd.concat([out, out_train], ignore_index=True).to_csv(PRED_DIR / "xgb_expanded_preds.csv", index=False)

    print("[xgb-exp] DONE.")


if __name__ == "__main__":
    main()
