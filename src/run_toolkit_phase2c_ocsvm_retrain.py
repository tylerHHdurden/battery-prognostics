"""
Toolkit Phase 2, OC-SVM retrain step ONLY - completes what `run_toolkit_
phase2_multisource_retrain.py` started but crashed on (a NaN-handling
bug in the OC-SVM feature matrix: `ocsvm_medians` could itself be NaN
for a column that's entirely missing in the sampled fit split, fixed
here). Reuses the ALREADY-SAVED artifacts from that run (`fusion_
embeddings_multisource.csv`, the candidate encoder/XGBoost model) -
does NOT rebuild raw tensors or retrain the encoder again (the
expensive ~25 min step that already completed successfully).
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import canonical_feature_cols, add_reformulated_duration_features, build_calce_merged, OUT_DIR, PROC_DIR, ROOT
from split_utils import battery_level_split

BATTERYLIFE_LOWER = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                     "stanford", "stanford_2", "isu_ilcc", "tongji"]
SEED = 42
TEST_EVERY = 5
EMBED_DIM = 16


def main():
    t0 = time.time()
    print("=== Toolkit Phase 2c: OC-SVM retrain (reusing already-saved candidate artifacts) ===")

    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings_multisource.csv")
    fusion_cols_new = [f"fusion_{i}" for i in range(EMBED_DIM)]

    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_full = add_reformulated_duration_features(hi_full)
    feature_cols_base = canonical_feature_cols(reformulated=True)
    hi_cols_needed = feature_cols_base + ["cycle_idx", "SOH", "battery_id", "dataset"]

    sources_hi = {}
    sources_hi["NASA"] = hi_full[hi_full["dataset"] == "NASA"][hi_cols_needed].copy()
    sources_hi["MIT"] = hi_full[hi_full["dataset"] == "MIT"][hi_cols_needed].copy()
    calce_merged = build_calce_merged(hi_full)
    sources_hi["CALCE"] = calce_merged[feature_cols_base + ["cycle_idx", "SOH", "battery_id"]].copy()
    sources_hi["CALCE"]["dataset"] = "CALCE"
    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"), ("HUST", "stage5_1_hust_merged.parquet"),
                        ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        df = pd.read_parquet(PROC_DIR / fname)
        sources_hi[name] = df[feature_cols_base + ["cycle_idx", "SOH", "battery_id"]].copy()
        sources_hi[name]["dataset"] = name
    for source in BATTERYLIFE_LOWER:
        path = PROC_DIR / ("batterylife_tongji_merged.parquet" if source == "tongji" else f"batterylife_{source}_merged.parquet")
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        sub = df[feature_cols_base + ["cycle_idx", "SOH", "battery_id"]].copy()
        sub["battery_id"] = source + "::" + sub["battery_id"].astype(str)
        sub["dataset"] = source
        sources_hi[source] = sub

    pooled_frames = []
    for name, df in sources_hi.items():
        fsub = fusion_df[fusion_df["dataset"] == name][["battery_id", "cycle_idx"] + fusion_cols_new]
        merged = pd.merge(df, fsub, on=["battery_id", "cycle_idx"], how="inner")
        merged["_source"] = name
        pooled_frames.append(merged)
    pool = pd.concat(pooled_frames, ignore_index=True)
    print(f"[phase2c] rebuilt pool: {len(pool)} rows, {pool['battery_id'].nunique()} batteries "
          f"(from already-saved fusion embeddings, no tensor rebuild)")

    dataset_of = dict(zip(pool["battery_id"], pool["_source"]))
    unique_bids = sorted(pool["battery_id"].unique().tolist())
    train_ids, test_ids = battery_level_split(unique_bids, test_every=TEST_EVERY, dataset_of=dataset_of)
    n_val_batt = max(1, len(train_ids) // 5)
    val_ids_set = set(sorted(train_ids)[-n_val_batt:])
    fit_ids_set = set(train_ids) - val_ids_set

    ocsvm_feature_cols = feature_cols_base + ["cycle_idx"] + fusion_cols_new
    fit_df = pool.loc[pool["battery_id"].isin(fit_ids_set)]
    max_per_battery = 200
    rng = np.random.default_rng(SEED)
    sampled = []
    for bid, g in fit_df.groupby("battery_id"):
        if len(g) > max_per_battery:
            g = g.sample(n=max_per_battery, random_state=SEED)
        sampled.append(g)
    fit_df_sampled = pd.concat(sampled, ignore_index=True)

    Xo = fit_df_sampled[ocsvm_feature_cols].to_numpy(dtype=float, copy=True)
    Xo = np.where(np.isinf(Xo), np.nan, Xo)
    with np.errstate(invalid="ignore"):
        ocsvm_medians = np.nanmedian(Xo, axis=0)
    ocsvm_medians = np.where(np.isnan(ocsvm_medians), 0.0, ocsvm_medians)  # FIX: entirely-NaN column -> 0.0
    inds4 = np.where(np.isnan(Xo))
    Xo[inds4] = np.take(ocsvm_medians, inds4[1])
    assert np.isfinite(Xo).all(), "still non-finite values after imputation - investigate before fitting"

    ocsvm_scaler = StandardScaler().fit(Xo)
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(ocsvm_scaler.transform(Xo))
    with open(ROOT / "models" / "_candidate_ocsvm.pkl", "wb") as f:
        pickle.dump(ocsvm, f)
    with open(ROOT / "models" / "_candidate_ocsvm_scaler.pkl", "wb") as f:
        pickle.dump(ocsvm_scaler, f)
    with open(PROC_DIR / "_candidate_ocsvm_medians.json", "w") as f:
        json.dump({"cols": ocsvm_feature_cols, "medians": ocsvm_medians.tolist()}, f)
    print(f"[phase2c] saved models/_candidate_ocsvm.pkl (trained on {len(fit_df_sampled)} cycles, "
          f"{len(fit_ids_set)} fit batteries)")

    # ---------- flag rate: candidate vs deployed, on the SAME external held-out rows ----------
    test_mask = pool["battery_id"].isin(test_ids).to_numpy()
    test_rows = pool.loc[test_mask].reset_index(drop=True)
    ext_test = test_rows[~test_rows["_source"].isin({"NASA", "MIT"})].reset_index(drop=True)

    Xo_cand = ext_test[ocsvm_feature_cols].to_numpy(dtype=float, copy=True)
    Xo_cand = np.where(np.isinf(Xo_cand), np.nan, Xo_cand)
    inds5 = np.where(np.isnan(Xo_cand))
    Xo_cand[inds5] = np.take(ocsvm_medians, inds5[1])
    cand_flags = ocsvm.predict(ocsvm_scaler.transform(Xo_cand)) == -1
    cand_flag_rate = float(cand_flags.mean())
    print(f"[phase2c] CANDIDATE OC-SVM flags {cand_flag_rate:.1%} of {len(ext_test)} external held-out cycles "
          f"({ext_test['battery_id'].nunique()} batteries) as anomalous")

    from researchpass_partA_common import base_feature_cols
    base_cols_full = base_feature_cols()
    with open(ROOT / "models" / "ocsvm_model.pkl", "rb") as f:
        deployed_ocsvm = pickle.load(f)
    with open(ROOT / "models" / "ocsvm_scaler.pkl", "rb") as f:
        deployed_scaler = pickle.load(f)

    OLD_FUSION_COLS = [f"fusion_{i}" for i in range(16)]
    old_fusion_by_source = {}
    nasa_mit_old_fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    old_fusion_by_source["NASA"] = nasa_mit_old_fusion[nasa_mit_old_fusion["dataset"] == "NASA"][["battery_id", "cycle_idx"] + OLD_FUSION_COLS]
    old_fusion_by_source["MIT"] = nasa_mit_old_fusion[nasa_mit_old_fusion["dataset"] == "MIT"][["battery_id", "cycle_idx"] + OLD_FUSION_COLS]
    old_fusion_by_source["CALCE"] = calce_merged[["battery_id", "cycle_idx"] + OLD_FUSION_COLS]
    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"), ("HUST", "stage5_1_hust_merged.parquet"),
                        ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        df = pd.read_parquet(PROC_DIR / fname)
        old_fusion_by_source[name] = df[["battery_id", "cycle_idx"] + OLD_FUSION_COLS]
    for source in BATTERYLIFE_LOWER:
        path = PROC_DIR / ("batterylife_tongji_merged.parquet" if source == "tongji" else f"batterylife_{source}_merged.parquet")
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        old_fusion_sub = df[["battery_id", "cycle_idx"] + OLD_FUSION_COLS].copy()
        old_fusion_sub["battery_id"] = source + "::" + old_fusion_sub["battery_id"].astype(str)
        old_fusion_by_source[source] = old_fusion_sub

    deployed_flags_per_src = []
    for src in ext_test["_source"].unique():
        src_mask = (ext_test["_source"] == src).to_numpy()
        rows_src = ext_test.loc[src_mask]
        old_fusion_match = old_fusion_by_source.get(src)
        if old_fusion_match is None or len(old_fusion_match) == 0:
            continue
        join_key = rows_src[["battery_id", "cycle_idx"]]
        merged_old = join_key.merge(old_fusion_match, on=["battery_id", "cycle_idx"], how="left")
        Xh = rows_src[base_cols_full].to_numpy(dtype=float, copy=True)
        Xf = merged_old[[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float, copy=True)
        X_d = np.hstack([Xh, Xf])
        X_d = np.where(np.isinf(X_d), np.nan, X_d)
        inds6 = np.where(np.isnan(X_d))
        if inds6[0].size:
            with np.errstate(invalid="ignore"):
                col_med_fallback = np.nanmedian(X_d, axis=0)
            col_med_fallback = np.where(np.isnan(col_med_fallback), 0.0, col_med_fallback)
            X_d[inds6] = np.take(col_med_fallback, inds6[1])
        flags = deployed_ocsvm.predict(deployed_scaler.transform(X_d)) == -1
        deployed_flags_per_src.append(flags)

    if deployed_flags_per_src:
        deployed_flags_all = np.concatenate(deployed_flags_per_src)
        deployed_flag_rate = float(deployed_flags_all.mean())
        print(f"[phase2c] DEPLOYED OC-SVM (currently in production) flags {deployed_flag_rate:.1%} of the "
              f"SAME external held-out cycles as anomalous")
        print(f"[phase2c] OC-SVM flag rate on external batteries: deployed={deployed_flag_rate:.1%} -> "
              f"candidate={cand_flag_rate:.1%} ({'MORE' if cand_flag_rate>deployed_flag_rate else 'FEWER'} "
              f"flagged after multi-source retraining)")
        pd.DataFrame([{"deployed_flag_rate": deployed_flag_rate, "candidate_flag_rate": cand_flag_rate,
                       "n_external_cycles": len(ext_test), "n_external_batteries": ext_test["battery_id"].nunique()}]
                     ).to_csv(OUT_DIR / "toolkit_phase2c_ocsvm_comparison.csv", index=False)
    else:
        print("[phase2c] could not score the deployed OC-SVM on the same rows")

    print(f"\n[phase2c] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
