"""
Research pass 2, item 1: few-shot test-time adaptation. Every held-out
evaluation on record so far (CALCE/Oxford/HUST/XJTU) has been strict
zero-retrain - the deployed model never sees a single cycle from the
target battery before being scored on it. This tests the more realistic
alternative: given the first K cycles of a target-domain battery, adapt
before predicting the REST of that same battery's trajectory.

TWO adaptation methods, both applied PER BATTERY using only that
battery's own first K cycles (no cross-battery leakage):
  (a) recalibration: fit y = a*raw_pred + b (plain 1-D linear
      regression) on the K adapt cycles' (deployed-model raw
      prediction, true SOH) pairs, apply the affine correction to the
      deployed model's raw predictions on the remaining eval cycles.
      Cheap, low-variance, can't really overfit with only 2 free
      parameters even at K=5.
  (b) continued XGBoost training: warm-starts from the deployed
      booster (xgb_model=) and adds a SMALL number of extra trees
      (20, vs. the deployed model's 500 - disclosed, chosen to limit
      overfitting risk on a K=5..10-row fit set) fit ONLY on the K
      adapt cycles, using the exact same hyperparameters as every
      other XGBoost-fusion variant in this project.

K tested at both 5 and 10 (the task's own "5-10 cycles" range) - cheap
to report both rather than picking one arbitrarily.

Batteries with fewer than K+5 total cycles are skipped (not enough
eval cycles left to score meaningfully) - disclosed via the skip count
per dataset, not silently dropped.

Reports BOTH a dataset-level pooled comparison (baseline zero-retrain
vs. recalibrated vs. continued-trained, R2/RMSE over all eval rows) AND
a full per-battery breakdown, directly against the deployed model's own
existing zero-retrain numbers on the SAME eval-cycle rows (not the
whole-battery numbers on record elsewhere, which include the K adapt
cycles too - apples-to-apples with THIS experiment's own eval set).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols,
    build_calce_merged, OUT_DIR, PROC_DIR, ROOT,
)

SEED = 42
K_VALUES = [5, 10]
MIN_EVAL_CYCLES = 5
CONTINUED_N_ESTIMATORS = 20


def load_dataset_frames():
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    calce = build_calce_merged(hi_full)
    frames = {"CALCE": calce}
    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"),
                         ("HUST", "stage5_1_hust_merged.parquet"),
                         ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        path = PROC_DIR / fname
        if path.exists():
            frames[name] = pd.read_parquet(path)
        else:
            print(f"[fewshot] WARNING: {fname} not found, skipping {name}")
    return frames, merged


def impute(X, medians):
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    return X


def main():
    t0 = time.time()
    print("=== Research pass 2, item 1: few-shot test-time adaptation ===")
    frames, merged_nm = load_dataset_frames()
    train_mask, test_mask, split = battery_split_masks(merged_nm)
    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    fcols = fusion_cols()
    cols = feature_cols + fcols

    deployed = XGBRegressor()
    deployed.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    booster = deployed.get_booster()

    X_train_all = merged_nm.loc[train_mask, cols].to_numpy(dtype=float, copy=True)
    train_medians = np.nanmedian(np.where(np.isinf(X_train_all), np.nan, X_train_all), axis=0)

    per_battery_rows = []
    dataset_summary_rows = []

    for ds_name, df in frames.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[fewshot] {ds_name}: missing columns {missing}, skipping entirely")
            continue
        df = df.copy()
        for k in K_VALUES:
            n_skipped = 0
            n_used = 0
            pooled = {"baseline": {"pred": [], "true": []},
                      "recal": {"pred": [], "true": []},
                      "continued": {"pred": [], "true": []}}
            for bid, bdf in df.groupby("battery_id"):
                bdf = bdf.sort_values("cycle_idx").reset_index(drop=True)
                if len(bdf) < k + MIN_EVAL_CYCLES:
                    n_skipped += 1
                    continue
                n_used += 1
                adapt = bdf.iloc[:k]
                eval_ = bdf.iloc[k:]

                X_adapt = impute(adapt[cols].to_numpy(dtype=float, copy=True), train_medians)
                y_adapt = adapt["SOH"].to_numpy(dtype=float)
                X_eval = impute(eval_[cols].to_numpy(dtype=float, copy=True), train_medians)
                y_eval = eval_["SOH"].to_numpy(dtype=float)

                pred_baseline_adapt = deployed.predict(X_adapt)
                pred_baseline_eval = deployed.predict(X_eval)

                # (a) recalibration: y = a*raw_pred + b, fit on adapt cycles only
                lr = LinearRegression().fit(pred_baseline_adapt.reshape(-1, 1), y_adapt)
                pred_recal_eval = lr.predict(pred_baseline_eval.reshape(-1, 1))

                # (b) continued XGBoost training: warm-start from deployed booster,
                # a few extra trees fit ONLY on the adapt cycles
                extra = XGBRegressor(n_estimators=CONTINUED_N_ESTIMATORS, max_depth=6,
                                      learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                      random_state=SEED, n_jobs=-1, reg_lambda=1.0)
                extra.fit(X_adapt, y_adapt, xgb_model=booster)
                pred_continued_eval = extra.predict(X_eval)

                r2_b = r2_score(y_eval, pred_baseline_eval) if len(y_eval) > 1 else np.nan
                r2_r = r2_score(y_eval, pred_recal_eval) if len(y_eval) > 1 else np.nan
                r2_c = r2_score(y_eval, pred_continued_eval) if len(y_eval) > 1 else np.nan
                rmse_b = float(np.sqrt(mean_squared_error(y_eval, pred_baseline_eval)))
                rmse_r = float(np.sqrt(mean_squared_error(y_eval, pred_recal_eval)))
                rmse_c = float(np.sqrt(mean_squared_error(y_eval, pred_continued_eval)))

                per_battery_rows.append({
                    "dataset": ds_name, "battery_id": bid, "k": k, "n_adapt": len(adapt), "n_eval": len(eval_),
                    "baseline_r2": r2_b, "recal_r2": r2_r, "continued_r2": r2_c,
                    "baseline_rmse": rmse_b, "recal_rmse": rmse_r, "continued_rmse": rmse_c,
                    "best_method": ["baseline", "recal", "continued"][int(np.nanargmax([r2_b, r2_r, r2_c]))],
                })

                pooled["baseline"]["pred"].append(pred_baseline_eval); pooled["baseline"]["true"].append(y_eval)
                pooled["recal"]["pred"].append(pred_recal_eval); pooled["recal"]["true"].append(y_eval)
                pooled["continued"]["pred"].append(pred_continued_eval); pooled["continued"]["true"].append(y_eval)

            if n_used == 0:
                print(f"[fewshot] {ds_name} k={k}: 0 batteries usable ({n_skipped} skipped, too few cycles)")
                continue

            summary = {"dataset": ds_name, "k": k, "n_batteries_used": n_used, "n_batteries_skipped": n_skipped}
            for method in ["baseline", "recal", "continued"]:
                p = np.concatenate(pooled[method]["pred"])
                y = np.concatenate(pooled[method]["true"])
                summary[f"{method}_r2"] = float(r2_score(y, p))
                summary[f"{method}_rmse"] = float(np.sqrt(mean_squared_error(y, p)))
            dataset_summary_rows.append(summary)
            print(f"[fewshot] {ds_name} k={k}: n_batteries={n_used} (skipped {n_skipped}) | "
                  f"baseline R2={summary['baseline_r2']:.4f} recal R2={summary['recal_r2']:.4f} "
                  f"continued R2={summary['continued_r2']:.4f}")

    per_battery_df = pd.DataFrame(per_battery_rows)
    summary_df = pd.DataFrame(dataset_summary_rows)
    per_battery_df.to_csv(OUT_DIR / "researchpass2_item1_fewshot_adaptation_per_battery.csv", index=False)
    summary_df.to_csv(OUT_DIR / "researchpass2_item1_fewshot_adaptation_summary.csv", index=False)

    print("\n=== DATASET-LEVEL SUMMARY (pooled R2 over all eval rows) ===")
    print(summary_df.to_string(index=False))

    print("\n=== VERDICTS (best few-shot method vs. zero-retrain baseline, per dataset/k) ===")
    for _, row in summary_df.iterrows():
        best_fewshot = max(row["recal_r2"], row["continued_r2"])
        verdict = "WIN (few-shot beats zero-retrain)" if best_fewshot > row["baseline_r2"] else "LOSS/TIE"
        best_method = "recal" if row["recal_r2"] >= row["continued_r2"] else "continued"
        print(f"[fewshot] {row['dataset']} k={row['k']}: baseline R2={row['baseline_r2']:.4f} vs "
              f"best few-shot ({best_method}) R2={best_fewshot:.4f} -> {verdict}")

    print(f"\n[fewshot] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
