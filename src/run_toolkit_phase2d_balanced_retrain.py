"""
Toolkit Phase 2d: source-balanced retrain of the Phase 2 candidate
XGBoost-fusion (sample weights so every SOURCE contributes equal TOTAL
weight - sibling families count as one source: {stanford, stanford_2}
and {mich, mich_exp} are each ONE weight group, matching CHECK 1's own
family definition, not re-derived). The retrained encoder is reused
AS-IS (no re-training) - reuses the already-saved `fusion_embeddings_
multisource.csv`, no tensor rebuild.

Reruns the EXACT SAME gate table computation as `run_toolkit_phase2_
multisource_retrain.py`'s own step 5, unchanged, so the two gate
tables are directly comparable.

Decision rule (per instruction, applied and reported, not just
computed): the balanced model becomes THE single candidate iff (a)
NASA and MIT's balanced R2 comes within bootstrap noise of the routed
model's own R2 (routed R2 falls inside the balanced candidate's own
95% CI) OR within 0.02 R2 absolute where no CI exists, AND (b) every
source that WON under the unweighted candidate still wins under the
balanced one. Otherwise: keep the unweighted candidate, and route by
dataset identity - NASA/MIT stay on the CURRENT DEPLOYED model, every
other known source AND any uploaded/unknown battery goes to the
(unweighted) candidate.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import canonical_feature_cols, add_reformulated_duration_features, build_calce_merged, OUT_DIR, PROC_DIR, ROOT
from split_utils import battery_level_split
from researchpass_partA_common import (
    base_feature_cols, extended_feature_cols, load_base_model, load_extended_model,
    load_base_pool_and_split, load_extended_pool_and_split, fit_medians,
)
from stage5_extended_reformulation import add_scv_matd_viect_reformulated

BATTERYLIFE_LOWER = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                     "stanford", "stanford_2", "isu_ilcc", "tongji"]
ALL_SOURCES = ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_LOWER
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
SEED = 42
N_BOOT = 1000
TEST_EVERY = 5
EMBED_DIM = 16

# Weight groups - sibling families collapse to one group, matching
# CHECK 1's own family definition exactly (not re-derived here).
WEIGHT_GROUP = {
    "NASA": "NASA", "MIT": "MIT", "CALCE": "CALCE", "Oxford": "Oxford",
    "HUST": "HUST", "XJTU": "XJTU", "ul_pur": "ul_pur", "hnei": "hnei",
    "snl": "snl", "mich": "mich_family", "mich_exp": "mich_family",
    "rwth": "rwth", "stanford": "stanford_family", "stanford_2": "stanford_family",
    "isu_ilcc": "isu_ilcc", "tongji": "tongji",
}


def metrics(y_true, pred):
    r2 = float(r2_score(y_true, pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    return {"r2": r2, "rmse": rmse, "mae": mae}


def bootstrap_ci(y_test, pred, bids, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    unique_b = np.unique(bids)
    if len(unique_b) < 2:
        return {"r2": (np.nan, np.nan)}
    idx_by_battery = {b: np.where(bids == b)[0] for b in unique_b}
    boot = []
    for _ in range(n_boot):
        sampled_b = rng.choice(unique_b, size=len(unique_b), replace=True)
        mask = np.concatenate([idx_by_battery[b] for b in sampled_b])
        if len(mask) < 2:
            continue
        boot.append(metrics(y_test[mask], pred[mask])["r2"])
    return {"r2": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (np.nan, np.nan)}


def main():
    t0 = time.time()
    print("=== Toolkit Phase 2d: source-balanced retrain (sample weights, reused encoder) ===")

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

    all_cols = feature_cols_base + ["cycle_idx"] + fusion_cols_new
    pooled_frames = []
    for name, df in sources_hi.items():
        fsub = fusion_df[fusion_df["dataset"] == name][["battery_id", "cycle_idx"] + fusion_cols_new]
        merged = pd.merge(df, fsub, on=["battery_id", "cycle_idx"], how="inner")
        merged["_source"] = name
        pooled_frames.append(merged)
    pool = pd.concat(pooled_frames, ignore_index=True)
    print(f"[phase2d] pooled: {len(pool)} rows, {pool['battery_id'].nunique()} batteries")

    dataset_of = dict(zip(pool["battery_id"], pool["_source"]))
    unique_bids = sorted(pool["battery_id"].unique().tolist())
    train_ids, test_ids = battery_level_split(unique_bids, test_every=TEST_EVERY, dataset_of=dataset_of)
    pool_train_mask = pool["battery_id"].isin(train_ids).to_numpy()
    pool_test_mask = ~pool_train_mask

    # ---------- sample weights: equal total weight per WEIGHT GROUP ----------
    pool["_weight_group"] = pool["_source"].map(WEIGHT_GROUP)
    n_groups = pool.loc[pool_train_mask, "_weight_group"].nunique()
    group_sizes = pool.loc[pool_train_mask].groupby("_weight_group").size()
    print(f"[phase2d] {n_groups} weight groups on the train split: {group_sizes.to_dict()}")
    sample_weight = np.ones(len(pool))
    train_group = pool.loc[pool_train_mask, "_weight_group"]
    w = (1.0 / n_groups) / train_group.map(group_sizes)
    sample_weight[pool_train_mask] = w.to_numpy()
    print(f"[phase2d] per-group total weight (should be ~equal, ~{1/n_groups:.4f} each): "
          f"{pd.Series(sample_weight[pool_train_mask], index=train_group).groupby(level=0).sum().to_dict()}")

    n_fusion = len(fusion_cols_new)
    monotone = tuple([0] * len(feature_cols_base) + [-1] + [0] * n_fusion)
    X_pool = pool[all_cols].to_numpy(dtype=float, copy=True)
    X_pool = np.where(np.isinf(X_pool), np.nan, X_pool)
    col_medians = np.nanmedian(X_pool[pool_train_mask], axis=0)
    inds = np.where(np.isnan(X_pool))
    X_pool[inds] = np.take(col_medians, inds[1])
    y_pool = pool["SOH"].to_numpy(dtype=float)

    print("[phase2d] training source-balanced candidate XGBoost-fusion...")
    xgb_balanced = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                                 colsample_bytree=0.8, random_state=SEED, n_jobs=-1, reg_lambda=1.0,
                                 monotone_constraints=monotone)
    xgb_balanced.fit(X_pool[pool_train_mask], y_pool[pool_train_mask], sample_weight=sample_weight[pool_train_mask])
    xgb_balanced.save_model(str(ROOT / "models" / "_candidate_multisource_balanced.json"))
    print("[phase2d] saved models/_candidate_multisource_balanced.json")

    pred_candidate = xgb_balanced.predict(X_pool[pool_test_mask])
    y_test = y_pool[pool_test_mask]
    bids_test = pool.loc[pool_test_mask, "battery_id"].to_numpy()
    src_test = pool.loc[pool_test_mask, "_source"].to_numpy()

    # ---------- gate vs. current deployed routed model (identical to Phase 2's own step 5) ----------
    base_cols_full = base_feature_cols()
    ext_cols_full = extended_feature_cols()
    base_model = load_base_model()
    ext_model = load_extended_model()
    merged_b, hi_full_b, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_e, hi_full_e, hi_full_raw_e, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_medians = fit_medians(merged_b, train_mask_b, base_cols_full)
    ext_medians = fit_medians(merged_e, train_mask_e, ext_cols_full)

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

    test_rows = pool.loc[pool_test_mask].reset_index(drop=True)
    pred_routed = np.full(len(test_rows), np.nan)
    routed_which = np.empty(len(test_rows), dtype=object)
    for src in test_rows["_source"].unique():
        src_mask = (test_rows["_source"] == src).to_numpy()
        rows_src = test_rows.loc[src_mask].copy()
        old_fusion_match = old_fusion_by_source.get(src)
        if old_fusion_match is None or len(old_fusion_match) == 0:
            pred_routed[np.where(src_mask)[0]] = np.nan
            routed_which[src_mask] = "no_old_embedding"
            continue
        join_key = rows_src[["battery_id", "cycle_idx"]]
        merged_old = join_key.merge(old_fusion_match, on=["battery_id", "cycle_idx"], how="left")
        if src in EXTENDED_ROUTED_DATASETS:
            rows_src_ext = rows_src.reset_index(drop=True).copy()
            rows_src_ext = add_scv_matd_viect_reformulated(rows_src_ext)
            Xh = rows_src_ext[ext_cols_full].to_numpy(dtype=float, copy=True)
            Xf = merged_old[[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float, copy=True)
            X_r = np.hstack([Xh, Xf])
            X_r = np.where(np.isinf(X_r), np.nan, X_r)
            inds3 = np.where(np.isnan(X_r))
            ref_med = np.concatenate([ext_medians[:len(ext_cols_full)], np.zeros(16)])
            X_r[inds3] = np.take(ref_med, inds3[1])
            pred_routed[np.where(src_mask)[0]] = ext_model.predict(X_r)
            routed_which[src_mask] = "extended"
        else:
            Xh = rows_src[base_cols_full].to_numpy(dtype=float, copy=True)
            Xf = merged_old[[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float, copy=True)
            X_r = np.hstack([Xh, Xf])
            X_r = np.where(np.isinf(X_r), np.nan, X_r)
            inds3 = np.where(np.isnan(X_r))
            ref_med = np.concatenate([base_medians[:len(base_cols_full)], np.zeros(16)])
            X_r[inds3] = np.take(ref_med, inds3[1])
            pred_routed[np.where(src_mask)[0]] = base_model.predict(X_r)
            routed_which[src_mask] = "base"

    gate_rows = []
    for name in ALL_SOURCES:
        mask = src_test == name
        if mask.sum() == 0:
            continue
        y_s = y_test[mask]
        pred_c_s = pred_candidate[mask]
        pred_r_s = pred_routed[mask]
        valid_r = np.isfinite(pred_r_s)
        bids_s = bids_test[mask]

        m_c = metrics(y_s, pred_c_s)
        ci_c = bootstrap_ci(y_s, pred_c_s, bids_s)
        if valid_r.sum() > 1:
            m_r = metrics(y_s[valid_r], pred_r_s[valid_r])
        else:
            m_r = {"r2": np.nan, "rmse": np.nan, "mae": np.nan}

        which = routed_which[np.where(mask)[0][0]]
        wins = bool(np.isfinite(m_r["r2"]) and m_c["r2"] > m_r["r2"])
        print(f"[phase2d] {name} (n={mask.sum()}, {len(set(bids_s))} batt): BALANCED CANDIDATE R2={m_c['r2']:.4f} "
              f"[{ci_c['r2'][0]:.4f},{ci_c['r2'][1]:.4f}] | ROUTED({which}) R2={m_r['r2']:.4f} -> "
              f"{'CANDIDATE WINS' if wins else ('N/A' if not np.isfinite(m_r['r2']) else 'ROUTED WINS')}")
        gate_rows.append({"source": name, "n": int(mask.sum()), "n_batteries": len(set(bids_s)),
                          "balanced_r2": m_c["r2"], "balanced_r2_ci_lo": ci_c["r2"][0], "balanced_r2_ci_hi": ci_c["r2"][1],
                          "balanced_rmse": m_c["rmse"], "balanced_mae": m_c["mae"],
                          "routed_r2": m_r["r2"], "routed_model_used": which, "balanced_wins": wins})

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(OUT_DIR / "toolkit_phase2d_balanced_gate_table.csv", index=False)

    # ---------- decision rule ----------
    print("\n=== DECISION RULE ===")
    unweighted_gate = pd.read_csv(OUT_DIR / "toolkit_phase2_gate_table.csv")
    nasa_row = gate_df[gate_df["source"] == "NASA"].iloc[0]
    mit_row = gate_df[gate_df["source"] == "MIT"].iloc[0]

    def within_noise(row, orig_row):
        routed_r2 = row["routed_r2"]
        if np.isfinite(row["balanced_r2_ci_lo"]) and np.isfinite(row["balanced_r2_ci_hi"]):
            ok = row["balanced_r2_ci_lo"] <= routed_r2 <= row["balanced_r2_ci_hi"]
            basis = f"CI check: routed R2={routed_r2:.4f} in [{row['balanced_r2_ci_lo']:.4f},{row['balanced_r2_ci_hi']:.4f}]? {ok}"
        else:
            ok = abs(row["balanced_r2"] - routed_r2) <= 0.02
            basis = f"0.02-absolute check: |{row['balanced_r2']:.4f}-{routed_r2:.4f}|={abs(row['balanced_r2']-routed_r2):.4f} <= 0.02? {ok}"
        return ok, basis

    nasa_ok, nasa_basis = within_noise(nasa_row, None)
    mit_ok, mit_basis = within_noise(mit_row, None)
    print(f"[phase2d] NASA: {nasa_basis}")
    print(f"[phase2d] MIT: {mit_basis}")

    orig_wins = set(unweighted_gate[unweighted_gate["candidate_wins"] == True]["source"])
    balanced_still_wins = all(bool(gate_df[gate_df["source"] == s]["balanced_wins"].iloc[0]) for s in orig_wins)
    lost_wins = [s for s in orig_wins if not bool(gate_df[gate_df["source"] == s]["balanced_wins"].iloc[0])]
    print(f"[phase2d] sources that won under the UNWEIGHTED candidate: {sorted(orig_wins)}")
    print(f"[phase2d] all of those STILL win under the balanced candidate? {balanced_still_wins}"
          + (f" (LOST: {lost_wins})" if lost_wins else ""))

    adopt_balanced = nasa_ok and mit_ok and balanced_still_wins
    print(f"\n[phase2d] === FINAL DECISION: {'ADOPT BALANCED MODEL as the single candidate' if adopt_balanced else 'KEEP UNWEIGHTED CANDIDATE + route by dataset identity (NASA/MIT->deployed, else->candidate)'} ===")
    print(f"[phase2d]   NASA within noise: {nasa_ok} | MIT within noise: {mit_ok} | all original wins preserved: {balanced_still_wins}")

    with open(OUT_DIR / "toolkit_phase2d_decision.txt", "w") as f:
        f.write(f"adopt_balanced={adopt_balanced}\nnasa_ok={nasa_ok}\nmit_ok={mit_ok}\n"
                f"balanced_still_wins={balanced_still_wins}\nlost_wins={lost_wins}\n")

    print(f"\n[phase2d] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
