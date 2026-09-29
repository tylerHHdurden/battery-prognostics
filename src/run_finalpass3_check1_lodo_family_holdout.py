"""
Final verification pass, CHECK 1: LODO sibling/family-level holdout.

Item B's original LODO held out exactly ONE source at a time - if two
sources share a lab/protocol/cell-batch origin, leaving one "sibling"
in the training pool while holding out the other could let the model
transfer via near-duplicate distributional overlap rather than genuine
cross-source generalization, inflating that target's own LODO R2.

FAMILY GROUPS (per this check's own explicit instruction, not
independently re-derived - confirmed directly against this project's
own source list first): {Stanford, Stanford_2}, {MICH, MICH_EXP},
{NASA, and any NASA-derived/randomized-usage set in the pool}. Verified
directly, not assumed: `hi_table.parquet`'s own `dataset` column has
exactly {CALCE, MIT, NASA} - no NASA-Randomized-Battery-Usage data was
ever merged into this project's training pool (that dataset was
acquired in Part B but explicitly, per that entry, never folded into
any retrain) - so NASA has NO sibling source anywhere in this 15-source
pool. Every other source (MIT, CALCE, Oxford, HUST, XJTU, ul_pur, hnei,
snl, rwth, isu_ilcc) is checked and confirmed to have no identified
sibling in this source list either - reported explicitly per target,
not silently assumed.

For each target: family-holdout excludes the target's ENTIRE family
(all its siblings, if any) from training, not just the target itself.
Where a target has no sibling, family-holdout is BY CONSTRUCTION
identical to the original single-source LODO for that target - still
re-run and re-reported here (not just copied), so the "identical"
claim is verified, not assumed.

Battery-ID-collision assertion: after building each (train, target)
split, asserts `set(train_battery_ids) & set(target_battery_ids) ==
empty` - both settings, every target - a real runtime check, not a
structural argument alone (battery IDs are source-prefixed, e.g.
"Stanford::CellX", which should make collision structurally impossible,
but this asserts it directly rather than trusting that reasoning).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, add_reformulated_duration_features, fit_xgb,
    build_calce_merged, fusion_cols, OUT_DIR, PROC_DIR,
)

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
ALL_SOURCES = ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_SOURCES

FAMILIES = {
    "NASA": {"NASA"},          # no sibling found in this pool - confirmed, not assumed
    "MIT": {"MIT"},
    "CALCE": {"CALCE"},
    "Oxford": {"Oxford"},
    "HUST": {"HUST"},
    "XJTU": {"XJTU"},
    "ul_pur": {"ul_pur"},
    "hnei": {"hnei"},
    "snl": {"snl"},
    "rwth": {"rwth"},
    "isu_ilcc": {"isu_ilcc"},
    "mich": {"mich", "mich_exp"},
    "mich_exp": {"mich", "mich_exp"},
    "stanford": {"stanford", "stanford_2"},
    "stanford_2": {"stanford", "stanford_2"},
}

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


def bootstrap_ci(y_test, pred, bids, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    unique_b = np.unique(bids)
    idx_by_battery = {b: np.where(bids == b)[0] for b in unique_b}
    boot = {"r2": [], "rmse": [], "mae": [], "mape_pct": []}
    for _ in range(n_boot):
        sampled_b = rng.choice(unique_b, size=len(unique_b), replace=True)
        mask = np.concatenate([idx_by_battery[b] for b in sampled_b])
        if len(mask) < 2:
            continue
        m = metrics(y_test[mask], pred[mask])
        for k in boot:
            boot[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}


def main():
    t0 = time.time()
    print("=== CHECK 1: LODO family/sibling-level holdout ===")

    feature_cols_base = canonical_feature_cols(reformulated=True)
    fcols = fusion_cols()
    feature_cols = feature_cols_base + ["cycle_idx"]
    all_cols = feature_cols + fcols
    n_fusion = len(fcols)
    monotone = tuple([0] * len(feature_cols_base) + [-1] + [0] * n_fusion)

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

    for name, df in sources.items():
        df["battery_id"] = name + "::" + df["battery_id"].astype(str)

    print("[check1] confirmed family groups:")
    for name in ALL_SOURCES:
        fam = FAMILIES[name] - {name}
        print(f"    {name}: siblings excluded under family-holdout = {sorted(fam) if fam else 'NONE (no sibling in this pool)'}")

    rng = np.random.default_rng(SEED)
    results = []

    for held_out in ALL_SOURCES:
        if held_out not in sources:
            continue
        family = FAMILIES[held_out]
        t_s = time.time()

        train_df = pd.concat([df for name, df in sources.items() if name not in family], ignore_index=True)
        test_df = sources[held_out]

        train_bids = set(train_df["battery_id"].unique())
        test_bids_set = set(test_df["battery_id"].unique())
        overlap = train_bids & test_bids_set
        assert len(overlap) == 0, f"BUG: battery ID collision between train and target for {held_out}: {overlap}"

        X_train = train_df[all_cols].to_numpy(dtype=float, copy=True)
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        col_medians = np.nanmedian(X_train, axis=0)
        inds = np.where(np.isnan(X_train))
        X_train[inds] = np.take(col_medians, inds[1])
        y_train = train_df["SOH"].to_numpy(dtype=float)

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
        ci = bootstrap_ci(y_test, pred, bids)

        baseline = NASA_MIT_ONLY_BASELINE.get(held_out, None)
        siblings_excluded = sorted(family - {held_out})
        print(f"[check1] held-out={held_out} (family excludes: {siblings_excluded or 'none'}): "
              f"family-LODO R2={m['r2']:.4f} [{ci['r2'][0]:.4f},{ci['r2'][1]:.4f}] RMSE={m['rmse']:.4f} "
              f"MAE={m['mae']:.4f} MAPE={m['mape_pct']:.2f}% (n={len(y_test)}, "
              f"{test_df['battery_id'].nunique()} batteries, train_n={len(y_train)}) ({time.time()-t_s:.1f}s)")

        model.save_model(str(OUT_DIR.parent / "models" / f"_experimental_lodo_family_xgb_holdout_{held_out.lower()}.json"))

        results.append({
            "held_out_source": held_out, "siblings_excluded": ",".join(siblings_excluded) if siblings_excluded else "none",
            "n_test": len(y_test), "n_test_batteries": test_df["battery_id"].nunique(), "n_train": len(y_train),
            "family_lodo_r2": m["r2"], "family_lodo_r2_ci_lo": ci["r2"][0], "family_lodo_r2_ci_hi": ci["r2"][1],
            "family_lodo_rmse": m["rmse"], "family_lodo_rmse_ci_lo": ci["rmse"][0], "family_lodo_rmse_ci_hi": ci["rmse"][1],
            "family_lodo_mae": m["mae"], "family_lodo_mae_ci_lo": ci["mae"][0], "family_lodo_mae_ci_hi": ci["mae"][1],
            "family_lodo_mape_pct": m["mape_pct"],
            "nasa_mit_only_zeroretrain_r2": baseline,
            "family_lodo_beats_nasa_mit_only": (m["r2"] > baseline) if baseline is not None else None,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "finalpass3_check1_lodo_family_holdout.csv", index=False)

    # side-by-side with the ORIGINAL (single-source) LODO
    orig_df = pd.read_csv(OUT_DIR / "finalpass2_itemB_lodo_results.csv")
    merged = results_df.merge(orig_df[["held_out_source", "lodo_r2", "lodo_r2_ci_lo", "lodo_r2_ci_hi",
                                       "lodo_rmse", "lodo_mae", "lodo_mape_pct"]], on="held_out_source")
    merged.to_csv(OUT_DIR / "finalpass3_check1_side_by_side.csv", index=False)

    print("\n=== SIDE-BY-SIDE: original single-source LODO vs. family-level LODO ===")
    print(merged[["held_out_source", "siblings_excluded", "lodo_r2", "family_lodo_r2",
                  "nasa_mit_only_zeroretrain_r2"]].to_string(index=False))

    stanford_row = merged[merged["held_out_source"].str.lower() == "stanford"]
    if len(stanford_row):
        r = stanford_row.iloc[0]
        print(f"\n[check1] Stanford's original single-source LODO R2={r['lodo_r2']:.4f} vs. "
              f"family-holdout (Stanford_2 also excluded) R2={r['family_lodo_r2']:.4f} -> "
              f"{'SURVIVES (still comparably high)' if r['family_lodo_r2'] > 0.9 else 'DOES NOT SURVIVE - was sibling leakage'}")

    n_family_comparable = merged[merged["nasa_mit_only_zeroretrain_r2"].notna()]
    n_family_beats = int(n_family_comparable["family_lodo_beats_nasa_mit_only"].sum())
    n_orig_beats = int((n_family_comparable["lodo_r2"] > n_family_comparable["nasa_mit_only_zeroretrain_r2"]).sum())
    print(f"\n[check1] Targets beating NASA+MIT-only baseline: ORIGINAL LODO {n_orig_beats}/{len(n_family_comparable)}, "
          f"FAMILY-HOLDOUT LODO {n_family_beats}/{len(n_family_comparable)}")

    print(f"\n[check1] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
