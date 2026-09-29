"""
Final research pass, item 5 (rigor pass), sub-items 1-2:

  - 5 seeds (42 - the deployed model's own seed - plus 1/2/3/4), mean +/-
    std, for the FINAL configuration: the routed deployed setup exactly
    as it ships today (EXTENDED_ROUTED_DATASETS = {CALCE,Oxford,HUST},
    base model for XJTU + everything else). Nothing from items 1-4
    cleared this pass's own promotion bar, so "the final configuration"
    IS the already-shipped routed setup - no new candidate is folded in.
  - Battery-level bootstrap 95% CIs (seed=42, n_boot=1000, resampling
    BATTERIES with replacement, not rows - respects the actual unit of
    independence, same convention as run_bootstrap_significance.py).
  - Extended per-dataset metrics: R2, RMSE, MAE (in % SOH - SOH is
    already stored on a 0-100 scale project-wide, confirmed by every
    existing RMSE/MAE figure in this file, e.g. in-domain RMSE~0.75),
    NRMSE (=RMSE/mean(y_true)*100), NMAE (=MAE/mean(y_true)*100), MAPE
    (=mean(|y_true-pred|/|y_true|)*100).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, build_calce_merged, fusion_cols, OUT_DIR, PROC_DIR,
)
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols

SEEDS = [42, 1, 2, 3, 4]
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
N_BOOT = 1000
BOOT_SEED = 42


def eval_generic(model, medians, cols, df, y_col="SOH"):
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    y_true = df[y_col].to_numpy(dtype=float)
    pred = model.predict(X)
    return y_true, pred, df["battery_id"].to_numpy()


def metrics(y_true, pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = float(r2_score(y_true, pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    mean_y = float(np.mean(y_true))
    nrmse = rmse / abs(mean_y) * 100 if abs(mean_y) > 1e-9 else float("nan")
    nmae = mae / abs(mean_y) * 100 if abs(mean_y) > 1e-9 else float("nan")
    y_safe = np.where(np.abs(y_true) < 1e-6, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - pred) / y_safe)) * 100)
    return {"r2": r2, "rmse": rmse, "mae": mae, "nrmse_pct": nrmse, "nmae_pct": nmae, "mape_pct": mape}


def main():
    t0 = time.time()
    print("=== Final pass item 5a: 5-seed rigor + battery-level bootstrap CIs + extended metrics ===")

    base_features = canonical_feature_cols(reformulated=True)
    ext_features_raw = extended_canonical_feature_cols(base_features)

    merged_base, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged_base)

    hi_full_ext = add_scv_matd_viect_reformulated(hi_full)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    nasa_mit_ext = hi_full_ext[hi_full_ext["dataset"].isin(["NASA", "MIT"])]
    merged_ext = pd.merge(nasa_mit_ext, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    train_mask_e, test_mask_e, _ = battery_split_masks(merged_ext)

    calce_base = build_calce_merged(hi_full)
    calce_ext = add_scv_matd_viect_reformulated(calce_base)
    oxford_base = pd.read_parquet(PROC_DIR / "stage5_1_oxford_merged.parquet")
    oxford_ext = add_scv_matd_viect_reformulated(oxford_base)
    hust_base = pd.read_parquet(PROC_DIR / "stage5_1_hust_merged.parquet")
    hust_ext = add_scv_matd_viect_reformulated(hust_base)
    xjtu_base = pd.read_parquet(PROC_DIR / "stage5_1_xjtu_merged.parquet")

    batterylife_base = {}
    for source in BATTERYLIFE_SOURCES:
        path = PROC_DIR / f"batterylife_{source}_merged.parquet"
        if path.exists():
            batterylife_base[source] = pd.read_parquet(path)

    n_fusion = len(fusion_cols())
    feat_base = base_features + ["cycle_idx"]
    feat_ext = ext_features_raw + ["cycle_idx"]
    mono_base = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)
    mono_ext = tuple([0] * len(ext_features_raw) + [-1] + [0] * n_fusion)

    per_seed_rows = []
    seed42_preds = {}  # dataset -> (y_true, pred, battery_id) for the ROUTED model, seed 42 only

    for seed in SEEDS:
        print(f"\n[item5a] === seed={seed} ===")
        model_b, med_b, cols_b = fit_xgb(merged_base, train_mask, feat_base,
                                          xgb_extra_kwargs={"monotone_constraints": mono_base, "random_state": seed})
        model_e, med_e, cols_e = fit_xgb(merged_ext, train_mask_e, feat_ext,
                                          xgb_extra_kwargs={"monotone_constraints": mono_ext, "random_state": seed})

        indomain_result = eval_indomain(model_b, med_b, cols_b, merged_base, test_mask)
        y_true, pred = indomain_result["y_true"], indomain_result["pred"]
        bids = merged_base.loc[test_mask, "battery_id"].to_numpy()
        m = metrics(y_true, pred)
        per_seed_rows.append({"seed": seed, "dataset": "in-domain (TEST)", **m, "n": len(y_true)})
        if seed == 42:
            seed42_preds["in-domain (TEST)"] = (y_true, pred, bids)

        targets = {
            "CALCE": (calce_ext, calce_base), "Oxford": (oxford_ext, oxford_base),
            "HUST": (hust_ext, hust_base), "XJTU": (None, xjtu_base),
        }
        for name, (df_ext, df_base) in targets.items():
            routed = name in EXTENDED_ROUTED_DATASETS
            if routed:
                y_true, pred, bids = eval_generic(model_e, med_e, cols_e, df_ext)
            else:
                y_true, pred, bids = eval_generic(model_b, med_b, cols_b, df_base)
            m = metrics(y_true, pred)
            per_seed_rows.append({"seed": seed, "dataset": name, **m, "n": len(y_true)})
            if seed == 42:
                seed42_preds[name] = (y_true, pred, bids)

        for source, df in batterylife_base.items():
            y_true, pred, bids = eval_generic(model_b, med_b, cols_b, df)
            m = metrics(y_true, pred)
            per_seed_rows.append({"seed": seed, "dataset": source, **m, "n": len(y_true)})
            if seed == 42:
                seed42_preds[source] = (y_true, pred, bids)

        print(f"[item5a] seed={seed} done")

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(OUT_DIR / "finalpass_item5a_per_seed_metrics.csv", index=False)

    agg = per_seed_df.groupby("dataset").agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        nrmse_pct_mean=("nrmse_pct", "mean"), nrmse_pct_std=("nrmse_pct", "std"),
        nmae_pct_mean=("nmae_pct", "mean"), nmae_pct_std=("nmae_pct", "std"),
        mape_pct_mean=("mape_pct", "mean"), mape_pct_std=("mape_pct", "std"),
        n=("n", "first"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "finalpass_item5a_5seed_aggregate.csv", index=False)
    print("\n=== 5-SEED MEAN +/- STD (routed final configuration) ===")
    print(agg.to_string(index=False))

    print(f"\n[item5a] === battery-level bootstrap 95% CIs (seed=42 only, n_boot={N_BOOT}) ===")
    rng = np.random.default_rng(BOOT_SEED)
    boot_rows = []
    for name, (y_true, pred, bids) in seed42_preds.items():
        unique_b = np.unique(bids)
        idx_by_battery = {b: np.where(bids == b)[0] for b in unique_b}  # built ONCE per dataset
        point = metrics(y_true, pred)
        boot_r2, boot_rmse, boot_mae = [], [], []
        for _ in range(N_BOOT):
            sampled_b = rng.choice(unique_b, size=len(unique_b), replace=True)
            mask = np.concatenate([idx_by_battery[b] for b in sampled_b])
            if len(mask) < 2:
                continue
            bm = metrics(y_true[mask], pred[mask])
            boot_r2.append(bm["r2"]); boot_rmse.append(bm["rmse"]); boot_mae.append(bm["mae"])
        lo_r2, hi_r2 = np.percentile(boot_r2, [2.5, 97.5])
        lo_rmse, hi_rmse = np.percentile(boot_rmse, [2.5, 97.5])
        lo_mae, hi_mae = np.percentile(boot_mae, [2.5, 97.5])
        print(f"[item5a] {name}: R2={point['r2']:.4f} [{lo_r2:.4f},{hi_r2:.4f}] | "
              f"RMSE={point['rmse']:.4f} [{lo_rmse:.4f},{hi_rmse:.4f}] | "
              f"MAE={point['mae']:.4f} [{lo_mae:.4f},{hi_mae:.4f}] (n_batteries={len(unique_b)})")
        boot_rows.append({"dataset": name, "r2": point["r2"], "r2_ci_lo": lo_r2, "r2_ci_hi": hi_r2,
                          "rmse": point["rmse"], "rmse_ci_lo": lo_rmse, "rmse_ci_hi": hi_rmse,
                          "mae": point["mae"], "mae_ci_lo": lo_mae, "mae_ci_hi": hi_mae,
                          "n_batteries": len(unique_b)})
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_DIR / "finalpass_item5a_bootstrap_ci.csv", index=False)

    print(f"\n[item5a] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
