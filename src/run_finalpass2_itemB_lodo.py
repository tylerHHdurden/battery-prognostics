"""
Final experiment pass 2, item B: leave-one-dataset-out (LODO).

15 sources: NASA, MIT, CALCE, Oxford, HUST, XJTU, ul_pur, hnei, snl,
mich, mich_exp, rwth, stanford, stanford_2, isu_ilcc. For each held-out
source, trains XGBoost-fusion on the POOLED union of all 14 OTHER
sources - SAME features (canonical 8 HI, Stage-1.1-reformulated,
+cycle_idx +16-dim fusion - the deployed BASE model's own recipe) and
SAME fixed hyperparameters/monotone_constraints as Stage 4/Stage 1.5
(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
colsample_bytree=0.8, reg_lambda=1.0, monotone_constraints=-1 on
cycle_idx only) - NO tuning on the held-out set, reusing
`stage1_common.fit_xgb` unchanged so this is byte-identical machinery
to every other retrain in this project, just given a different (much
larger, multi-source) training pool. Evaluated zero-retrain on the
held-out source's own full data (no held-out-source rows ever enter
training).

Compared directly against the EXISTING deployed base model's own
zero-retrain numbers (trained on NASA+MIT ONLY) for every source other
than NASA/MIT themselves (which ARE that baseline's own training pool -
noted explicitly, not force-compared against themselves).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, fit_xgb, fusion_cols,
    build_calce_merged, add_reformulated_duration_features, OUT_DIR, PROC_DIR,
)

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
ALL_SOURCES = ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_SOURCES

# The EXISTING deployed base model's own zero-retrain numbers (trained
# NASA+MIT only) - audit_true_deployed_baseline.csv + this pass's own
# item 5a reproduction. NASA/MIT excluded (they ARE that model's train pool).
NASA_MIT_ONLY_BASELINE = {
    "CALCE": 0.567876, "Oxford": -2.693906, "HUST": -0.152262, "XJTU": -1.061987,
    "ul_pur": 0.116107, "hnei": -0.037714, "snl": 0.146501, "mich": 0.572789,
    "mich_exp": 0.720633, "rwth": -0.484674, "stanford": 0.111290,
    "stanford_2": 0.065504, "isu_ilcc": 0.140112,
}

N_BOOT = 1000
SEED = 42


def metrics(y_true, pred):
    r2 = float(r2_score(y_true, pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    y_safe = np.where(np.abs(y_true) < 1e-6, np.nan, y_true)
    mape = float(np.nanmean(np.abs(y_true - pred) / y_safe) * 100)
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape_pct": mape}


def main():
    t0 = time.time()
    print("=== Final pass 2, item B: leave-one-dataset-out (LODO) ===")

    feature_cols_base = canonical_feature_cols(reformulated=True)  # 8 HI, Stage 1.1 reformulated
    fcols = fusion_cols()
    feature_cols = feature_cols_base + ["cycle_idx"]
    all_cols = feature_cols + fcols
    n_fusion = len(fcols)
    monotone = tuple([0] * len(feature_cols_base) + [-1] + [0] * n_fusion)
    print(f"[itemB] feature cols ({len(all_cols)}): {all_cols}")
    print(f"[itemB] monotone_constraints: {monotone}")

    print("[itemB] building the 15-source pool...")
    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_full = add_reformulated_duration_features(hi_full)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")

    sources = {}
    nasa_mit = pd.merge(hi_full[hi_full["dataset"].isin(["NASA", "MIT"])], fusion_df,
                        on=["dataset", "battery_id", "cycle_idx"], how="inner")
    sources["NASA"] = nasa_mit[nasa_mit["dataset"] == "NASA"][all_cols + ["SOH", "battery_id"]].copy()
    sources["MIT"] = nasa_mit[nasa_mit["dataset"] == "MIT"][all_cols + ["SOH", "battery_id"]].copy()
    sources["CALCE"] = build_calce_merged(hi_full)[all_cols + ["SOH", "battery_id"]].copy()
    sources["Oxford"] = pd.read_parquet(PROC_DIR / "stage5_1_oxford_merged.parquet")[all_cols + ["SOH", "battery_id"]].copy()
    sources["HUST"] = pd.read_parquet(PROC_DIR / "stage5_1_hust_merged.parquet")[all_cols + ["SOH", "battery_id"]].copy()
    sources["XJTU"] = pd.read_parquet(PROC_DIR / "stage5_1_xjtu_merged.parquet")[all_cols + ["SOH", "battery_id"]].copy()
    for source in BATTERYLIFE_SOURCES:
        path = PROC_DIR / f"batterylife_{source}_merged.parquet"
        if path.exists():
            sources[source] = pd.read_parquet(path)[all_cols + ["SOH", "battery_id"]].copy()
        else:
            print(f"[itemB] {source}: no local merged table - excluded from the pool entirely")

    for name, df in sources.items():
        # prefix battery_id with source to avoid cross-source id collisions in bootstrap
        df["battery_id"] = name + "::" + df["battery_id"].astype(str)
        print(f"[itemB] {name}: {len(df)} rows, {df['battery_id'].nunique()} batteries")

    rng = np.random.default_rng(SEED)
    results = []
    for held_out in ALL_SOURCES:
        if held_out not in sources:
            continue
        t_s = time.time()
        train_df = pd.concat([df for name, df in sources.items() if name != held_out], ignore_index=True)
        test_df = sources[held_out]

        X_train = train_df[all_cols].to_numpy(dtype=float, copy=True)
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        col_medians = np.nanmedian(X_train, axis=0)
        inds = np.where(np.isnan(X_train))
        X_train[inds] = np.take(col_medians, inds[1])
        y_train = train_df["SOH"].to_numpy(dtype=float)

        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=-1, reg_lambda=1.0, monotone_constraints=monotone)
        model.fit(X_train, y_train)

        X_test = test_df[all_cols].to_numpy(dtype=float, copy=True)
        X_test = np.where(np.isinf(X_test), np.nan, X_test)
        inds2 = np.where(np.isnan(X_test))
        X_test[inds2] = np.take(col_medians, inds2[1])
        y_test = test_df["SOH"].to_numpy(dtype=float)
        pred = model.predict(X_test)
        m = metrics(y_test, pred)

        bids = test_df["battery_id"].to_numpy()
        unique_b = np.unique(bids)
        idx_by_battery = {b: np.where(bids == b)[0] for b in unique_b}
        boot_r2, boot_rmse, boot_mae, boot_mape = [], [], [], []
        for _ in range(N_BOOT):
            sampled_b = rng.choice(unique_b, size=len(unique_b), replace=True)
            mask = np.concatenate([idx_by_battery[b] for b in sampled_b])
            if len(mask) < 2:
                continue
            bm = metrics(y_test[mask], pred[mask])
            boot_r2.append(bm["r2"]); boot_rmse.append(bm["rmse"]); boot_mae.append(bm["mae"]); boot_mape.append(bm["mape_pct"])
        ci = lambda arr: (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        r2_lo, r2_hi = ci(boot_r2)
        rmse_lo, rmse_hi = ci(boot_rmse)
        mae_lo, mae_hi = ci(boot_mae)
        mape_lo, mape_hi = ci(boot_mape)

        baseline = NASA_MIT_ONLY_BASELINE.get(held_out, None)
        print(f"[itemB] held-out={held_out}: LODO R2={m['r2']:.4f} [{r2_lo:.4f},{r2_hi:.4f}] "
              f"RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} MAPE={m['mape_pct']:.2f}% "
              f"(n={len(y_test)}, {len(unique_b)} batteries, train_n={len(y_train)}) "
              f"| NASA+MIT-only baseline R2={baseline if baseline is not None else 'N/A (this source IS that baseline pool)'} "
              f"({time.time()-t_s:.1f}s)")

        model.save_model(str(OUT_DIR.parent / "models" / f"_experimental_lodo_xgb_holdout_{held_out.lower()}.json"))

        results.append({
            "held_out_source": held_out, "n_test": len(y_test), "n_test_batteries": len(unique_b),
            "n_train": len(y_train), "lodo_r2": m["r2"], "lodo_r2_ci_lo": r2_lo, "lodo_r2_ci_hi": r2_hi,
            "lodo_rmse": m["rmse"], "lodo_rmse_ci_lo": rmse_lo, "lodo_rmse_ci_hi": rmse_hi,
            "lodo_mae": m["mae"], "lodo_mae_ci_lo": mae_lo, "lodo_mae_ci_hi": mae_hi,
            "lodo_mape_pct": m["mape_pct"], "lodo_mape_ci_lo": mape_lo, "lodo_mape_ci_hi": mape_hi,
            "nasa_mit_only_zeroretrain_r2": baseline,
            "lodo_beats_nasa_mit_only": (m["r2"] > baseline) if baseline is not None else None,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "finalpass2_itemB_lodo_results.csv", index=False)

    print("\n=== SUMMARY: LODO vs. NASA+MIT-only zero-retrain ===")
    print(results_df[["held_out_source", "lodo_r2", "nasa_mit_only_zeroretrain_r2", "lodo_beats_nasa_mit_only"]].to_string(index=False))

    comparable = results_df[results_df["nasa_mit_only_zeroretrain_r2"].notna()]
    n_helps = int(comparable["lodo_beats_nasa_mit_only"].sum())
    print(f"\n[itemB] Source diversity (LODO pooling) beats the NASA+MIT-only baseline on "
          f"{n_helps}/{len(comparable)} comparable targets.")

    print(f"\n[itemB] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
