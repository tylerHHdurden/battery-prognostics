"""
Stage 2, Item 2.3: GroupKFold (grouped by battery ID) replacing the
deterministic "every Nth battery" split, for the XGBoost-fusion
pipeline specifically (deep models explicitly out of scope for full
k-fold, per the original plan's cost note - lean pipeline only).

SCOPE DECISION on the "recovered pool from 2.1", stated explicitly
rather than silently skipped: fully integrating 2.1's 9 newly-
recovered batteries into this CV run would require recomputing all 16
health indicators + fusion embeddings from raw cycles for each of
them, respecting the corrected SOH/dropped-cycle baseline - a real,
non-trivial pool-rebuild step in its own right, touching the same
canonical data pool every later stage depends on. Given this session's
own explicit warning ("2.1 and 2.2 involve real data correction work
that needs care - a mistake here would propagate into every subsequent
stage"), rushing that integration alongside everything else in this
stage risks exactly that outcome. This run uses the EXISTING, already-
verified 204-battery expanded pool; full recovered-pool integration is
deferred to a dedicated future step, not silently dropped.

k=5: a reasonable default given 204 batteries (~41 batteries held out
per fold) - large enough per fold to give a meaningful test-set size,
small enough that 5 distinct folds each get genuine variety.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import canonical_feature_cols, add_reformulated_duration_features, fusion_cols

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

K_FOLDS = 5


def main():
    t0 = time.time()
    hi = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    nasa_mit = hi[hi["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    nasa_mit = add_reformulated_duration_features(nasa_mit)

    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings_expanded.csv")
    fcols = [c for c in fusion.columns if c.startswith("fusion_")]
    merged = pd.merge(nasa_mit, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan)  # known VDEDT inf on one MIT/b2c30 cycle

    base_features = canonical_feature_cols(reformulated=True)  # Stage 1's 1.1 config
    feature_cols = base_features + ["cycle_idx"]  # + 1.5's monotone-constrained feature
    n_fusion = len(fcols)
    monotone = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)
    cols = feature_cols + fcols

    print(f"[groupkfold] canonical Stage 1.1+1.5 feature set: {feature_cols}")
    print(f"[groupkfold] pool: {len(merged)} rows, {merged['battery_id'].nunique()} batteries")

    X_all = merged[cols].to_numpy(dtype=float, copy=True)
    y_all = merged["SOH"].to_numpy(dtype=float)
    groups = merged["battery_id"].to_numpy()

    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_results = []
    b0018_result = None

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=groups)):
        t_fold = time.time()
        X_train, X_test = X_all[train_idx].copy(), X_all[test_idx].copy()
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        test_battery_ids = groups[test_idx]

        col_medians = np.nanmedian(X_train, axis=0)
        inds = np.where(np.isnan(X_train))
        X_train[inds] = np.take(col_medians, inds[1])
        inds = np.where(np.isnan(X_test))
        X_test[inds] = np.take(col_medians, inds[1])

        model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=-1, reg_lambda=1.0, monotone_constraints=monotone)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
        n_test_batteries = len(set(test_battery_ids))
        elapsed = time.time() - t_fold
        print(f"[groupkfold] fold {fold_idx}: {n_test_batteries} test batteries, "
              f"{len(test_idx)} test rows, RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} "
              f"({elapsed:.1f}s)")
        fold_results.append({"fold": fold_idx, "n_test_batteries": n_test_batteries,
                             "n_test_rows": len(test_idx), "rmse": rmse, "mae": mae, "r2": r2})

        if "B0018" in test_battery_ids:
            b0018_mask = test_battery_ids == "B0018"
            b0018_rmse = float(np.sqrt(mean_squared_error(y_test[b0018_mask], pred[b0018_mask])))
            b0018_r2 = float(r2_score(y_test[b0018_mask], pred[b0018_mask])) if y_test[b0018_mask].std() > 0 else float("nan")
            print(f"[groupkfold]   B0018 lands in fold {fold_idx}: n={b0018_mask.sum()}, "
                  f"RMSE={b0018_rmse:.4f}, R2={b0018_r2:.4f}")
            b0018_result = {"fold": fold_idx, "n": int(b0018_mask.sum()), "rmse": b0018_rmse, "r2": b0018_r2}

    results_df = pd.DataFrame(fold_results)
    print(f"\n[groupkfold] === AGGREGATE across {K_FOLDS} folds ===")
    print(results_df.to_string(index=False))
    print(f"\n[groupkfold] mean RMSE={results_df.rmse.mean():.4f} (std={results_df.rmse.std():.4f}, "
          f"range=[{results_df.rmse.min():.4f}, {results_df.rmse.max():.4f}])")
    print(f"[groupkfold] mean R2={results_df.r2.mean():.4f} (std={results_df.r2.std():.4f}, "
          f"range=[{results_df.r2.min():.4f}, {results_df.r2.max():.4f}])")

    print(f"\n[groupkfold] === B0018 check ===")
    if b0018_result:
        others_r2 = results_df[results_df.fold != b0018_result["fold"]]["r2"]
        print(f"[groupkfold] B0018's own fold ({b0018_result['fold']}) R2={b0018_result['r2']:.4f} "
              f"(RMSE={b0018_result['rmse']:.4f}, n={b0018_result['n']} cycles) vs. "
              f"other folds' mean R2={others_r2.mean():.4f}")
        weak = b0018_result["r2"] < others_r2.mean() - others_r2.std()
        print(f"[groupkfold] B0018's fold notably weaker than other folds "
              f"(more than 1 std below their mean): {weak}")
    else:
        print("[groupkfold] B0018 not found in any test fold (unexpected)")

    print(f"\n[groupkfold] NOTE (per instruction): this GroupKFold run is for POINT-PREDICTOR "
          f"variance estimation / battery-level significance testing ONLY - it does NOT replace "
          f"the existing train/calibration/test split used for conformal work (Stage 1.6), which "
          f"remains a separate, fixed split for calibration exchangeability reasons.")

    results_df.to_csv(OUT_DIR / "stage2_groupkfold_cv_results.csv", index=False)
    if b0018_result:
        pd.DataFrame([b0018_result]).to_csv(OUT_DIR / "stage2_groupkfold_b0018_check.csv", index=False)
    print(f"\n[groupkfold] saved outputs/stage2_groupkfold_cv_results.csv")
    print(f"[groupkfold] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
