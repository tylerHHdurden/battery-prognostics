"""
Stage 1, Item 1.3 (XGBoost half): retrains XGBoost-fusion on the
EXPANDED (204-battery) pool with objective='reg:pseudohubererror'
replacing the default squared-error objective - same pool as the other
1.3/1.4 deep-model retrains specifically so B0053 (which does not exist
in the original 32-battery pool - confirmed before writing any of this
stage's code) can be checked, per this item's own instruction.

CANONICAL FEATURE SET: uses the Stage 1 canonical (Check 0.3, NASA+MIT-
only) 8-feature set on hi_table_expanded.parquet, NOT session 33's own
separately-reselected 204-battery 8-feature set - the canonical set is
now used everywhere in Stage 1 by decision, so this retrains BOTH an
MSE baseline and the pseudohuber variant on the canonical features
(neither exists yet on canonical features at expanded-pool scale),
rather than reusing the existing xgb_fusion_expanded_metrics.csv, which
used a different, non-canonical feature set and would not be a clean
apples-to-apples comparison.
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
OUT_DIR = ROOT / "outputs"

CANONICAL_FEATURES = ["ICHV", "SCV", "VDEDT", "VIECT", "MATD", "MET", "TEVD", "TEVI"]


def fit_eval(feature_cols, objective, merged, train_mask, test_mask, label):
    X = merged[feature_cols].to_numpy(dtype=float, copy=True)
    X[np.isinf(X)] = np.nan
    col_medians = np.nanmedian(X[train_mask], axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = merged["SOH"].to_numpy(dtype=float)

    kwargs = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  n_jobs=-1, reg_lambda=1.0)
    if objective:
        kwargs["objective"] = objective
        if objective == "reg:pseudohubererror":
            # ROOT-CAUSED (not a tuned hyperparameter, and max_delta_step
            # was tried first and did NOT fix it - kept failing
            # identically): isolated with a tiny synthetic example before
            # touching this dataset again. XGBoost 3.2.0's automatic
            # base_score estimation ("boost_from_average") for
            # reg:pseudohubererror is badly broken - on data with true
            # mean ~80, it estimated base_score=508737 instead of ~80
            # (confirmed directly via booster.save_config()), and 500
            # small-learning-rate boosting rounds never claw back from an
            # initial constant that wrong. Fix: bypass the broken
            # auto-estimation by passing the target's own training mean
            # as an explicit base_score - confirmed this alone fixes the
            # synthetic reproduction (RMSE 8544->0.54).
            kwargs["base_score"] = float(merged.loc[train_mask, "SOH"].mean())
    model = XGBRegressor(**kwargs)
    model.fit(X[train_mask], y[train_mask])
    pred_test = model.predict(X[test_mask])
    y_test = y[test_mask]
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))
    print(f"[xgb-pseudohuber] [{label}] TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    per_batt = pd.DataFrame({
        "battery_id": merged.loc[test_mask, "battery_id"].to_numpy(),
        "y_true": y_test, "pred": pred_test,
    })
    per_batt["abs_err"] = (per_batt.y_true - per_batt.pred).abs()
    batt_rmse = per_batt.groupby("battery_id").apply(
        lambda g: float(np.sqrt((g["abs_err"] ** 2).mean())), include_groups=False
    ).sort_values(ascending=False)
    return {"label": label, "rmse": rmse, "mae": mae, "r2": r2}, batt_rmse


def main():
    hi = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    nasa_mit = hi[hi["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings_expanded.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]
    merged = pd.merge(nasa_mit, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    feature_cols = CANONICAL_FEATURES + fusion_cols
    print(f"[xgb-pseudohuber] canonical features: {CANONICAL_FEATURES} (+{len(fusion_cols)} fusion dims)")
    print(f"[xgb-pseudohuber] expanded NASA+MIT merged pool: {len(merged)} rows, {merged['battery_id'].nunique()} batteries")

    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_mask = merged["battery_id"].isin(split["train_ids"]).to_numpy()
    test_mask = ~train_mask
    print(f"[xgb-pseudohuber] train rows={train_mask.sum()}, test rows={test_mask.sum()}, "
          f"B0053 in test: {'B0053' in merged.loc[test_mask, 'battery_id'].unique()}")

    mse_metrics, mse_batt = fit_eval(feature_cols, None, merged, train_mask, test_mask,
                                      "MSE (canonical features, expanded pool)")
    huber_metrics, huber_batt = fit_eval(feature_cols, "reg:pseudohubererror", merged, train_mask, test_mask,
                                          "pseudohuber (canonical features, expanded pool)")

    print("\n[xgb-pseudohuber] === B0053 individual RMSE ===")
    b0053_mse = float(mse_batt.get("B0053", float("nan")))
    b0053_huber = float(huber_batt.get("B0053", float("nan")))
    print(f"[xgb-pseudohuber] B0053: MSE-objective={b0053_mse:.4f}  pseudohuber-objective={b0053_huber:.4f}  "
          f"(delta={b0053_huber-b0053_mse:+.4f})")

    print("\n[xgb-pseudohuber] === worst 5 batteries, MSE-objective ===")
    print(mse_batt.head(5).to_string())
    print("\n[xgb-pseudohuber] === worst 5 batteries, pseudohuber-objective ===")
    print(huber_batt.head(5).to_string())

    # explicit regression check: any battery meaningfully WORSE under pseudohuber?
    common = mse_batt.index.intersection(huber_batt.index)
    delta = (huber_batt.loc[common] - mse_batt.loc[common]).sort_values(ascending=False)
    print("\n[xgb-pseudohuber] === largest REGRESSIONS (pseudohuber worse than MSE), top 5 ===")
    print(delta.head(5).to_string())

    print(f"\n[xgb-pseudohuber] === POOLED COMPARISON ===")
    print(f"[xgb-pseudohuber] R2:   MSE={mse_metrics['r2']:.4f}  pseudohuber={huber_metrics['r2']:.4f}  "
          f"(delta={huber_metrics['r2']-mse_metrics['r2']:+.4f})")
    print(f"[xgb-pseudohuber] RMSE: MSE={mse_metrics['rmse']:.4f}  pseudohuber={huber_metrics['rmse']:.4f}  "
          f"(delta={huber_metrics['rmse']-mse_metrics['rmse']:+.4f})")

    pd.DataFrame([mse_metrics, huber_metrics]).to_csv(OUT_DIR / "stage1_3_xgb_pseudohuber_pooled.csv", index=False)
    pd.DataFrame({"battery_id": mse_batt.index, "rmse_mse": mse_batt.values}).merge(
        pd.DataFrame({"battery_id": huber_batt.index, "rmse_pseudohuber": huber_batt.values}),
        on="battery_id", how="outer"
    ).to_csv(OUT_DIR / "stage1_3_xgb_pseudohuber_per_battery.csv", index=False)
    print("\n[xgb-pseudohuber] saved outputs/stage1_3_xgb_pseudohuber_{pooled,per_battery}.csv")
    print("[xgb-pseudohuber] DONE")


if __name__ == "__main__":
    main()
