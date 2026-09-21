"""
Re-audit, step 2: the ONE authoritative, fresh recomputation of the
TRUE currently-deployed model's zero-retrain numbers on all 4 held-out
datasets plus in-domain TEST - computed directly from the exact file
app.py/live_inference.py load in production
(models/xgb_soh_fusion.json, confirmed by direct grep of
live_inference.py's own loading code: `MODELS_DIR / "xgb_soh_fusion.
json"`, MODELS_DIR = ROOT/"models", no env override, no indirection -
the third independent confirmation of this file, after (1) this
session's own established convention across items 8/9/14/16/etc, which
all used this same file+feature pipeline, and (2) run_save_percycle_
predictions.py's own docstring, which already documented pool_suffix=""
as "deployed 42-battery, models/xgb_soh_fusion.json").

Uses the EXACT SAME feature pipeline (stage1_common.canonical_
feature_cols(reformulated=True) + cycle_idx + fusion_cols(), train-only
median imputation) as every other item in both research passes that
scored this model - not a different pipeline that could itself
introduce a discrepancy. This script's own numbers are NOT pulled from
any DEVELOPMENT_LOG.md table, any outputs/*.csv, or any prior research-
pass script's printed output - every number here is computed fresh,
right now, from the model file and the held-out data files directly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols,
    build_calce_merged, OUT_DIR, PROC_DIR, ROOT,
)

MODEL_PATH = ROOT / "models" / "xgb_soh_fusion.json"


def impute(X, medians):
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    return X


def main():
    print("=== Re-audit: TRUE deployed model (models/xgb_soh_fusion.json) fresh zero-retrain numbers ===")
    print(f"[audit] scoring model file: {MODEL_PATH}")
    assert MODEL_PATH.exists(), f"model file not found at {MODEL_PATH}"

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    fcols = fusion_cols()
    cols = feature_cols + fcols
    print(f"[audit] feature columns ({len(cols)}): {cols}")

    X_train = merged.loc[train_mask, cols].to_numpy(dtype=float, copy=True)
    train_medians = np.nanmedian(np.where(np.isinf(X_train), np.nan, X_train), axis=0)

    results = []

    def score(name, df):
        X = impute(df[cols].to_numpy(dtype=float, copy=True), train_medians)
        y = df["SOH"].to_numpy(dtype=float)
        pred = model.predict(X)
        r2 = float(r2_score(y, pred))
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        mae = float(mean_absolute_error(y, pred))
        n = len(y)
        n_batt = df["battery_id"].nunique()
        print(f"[audit] {name}: R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f} n={n} n_batteries={n_batt}")
        results.append({"eval_set": name, "r2": r2, "rmse": rmse, "mae": mae, "n": n, "n_batteries": n_batt})

    score("in-domain (TEST)", merged[test_mask])

    calce_merged = build_calce_merged(hi_full)
    score("CALCE", calce_merged)

    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"),
                         ("HUST", "stage5_1_hust_merged.parquet"),
                         ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        path = PROC_DIR / fname
        assert path.exists(), f"{fname} not found - cannot audit {name}"
        df = pd.read_parquet(path)
        missing = [c for c in cols if c not in df.columns]
        assert not missing, f"{name} missing columns {missing}"
        score(name, df)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "audit_true_deployed_baseline.csv", index=False)
    print("\n=== AUTHORITATIVE TRUE DEPLOYED MODEL NUMBERS (fresh, this run) ===")
    print(results_df.to_string(index=False))
    print(f"\n[audit] Saved to {OUT_DIR / 'audit_true_deployed_baseline.csv'}")


if __name__ == "__main__":
    main()
