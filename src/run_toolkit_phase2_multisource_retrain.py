"""
Toolkit Phase 2: multi-source promotion candidate - retrains the ICA
fusion encoder AND XGBoost-fusion AND the OC-SVM anomaly detector on
ALL 16 sources (NASA, MIT, CALCE, Oxford, HUST, XJTU + 9 BatteryLife
sources + Tongji), full (not subsampled) data.

GATE (per this phase's own explicit instruction): a battery-level
split WITHIN every source (fixed seed, `split_utils.battery_level_
split`'s own per-dataset stratification - same convention as this
pass's own earlier Phase 1 attempt, this time with a real, fixed
feature-shape bug from that attempt corrected: the routed-model
scoring path now correctly includes the 16-dim fusion columns it was
missing before). Reports R2/RMSE/MAE with battery-level bootstrap CIs
per source, against the CURRENT DEPLOYED ROUTED model (loaded
unchanged, scored fresh on these exact held-out rows).

Expected performance on genuinely NEW datasets is NOT re-measured with
the OLD encoder's LODO/family-holdout numbers here - a SEPARATE script
(`run_toolkit_phase2b_lodo_rerun.py`) reruns that check with the NEW
encoder's own embeddings, since the embedding representation itself
changed.

Saves models/_candidate_ica_encoder.pt, models/_candidate_multisource.
json (XGBoost-fusion), models/_candidate_ocsvm.pkl + models/_candidate_
ocsvm_scaler.pkl - all PROMOTION CANDIDATES, NOT wired into app.py/
live_inference.py. STOPS after printing the gate table, per instruction.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage1_common import (
    canonical_feature_cols, add_reformulated_duration_features,
    build_calce_merged, OUT_DIR, PROC_DIR, ROOT,
)
from data_adapters import iterate_calce_cycles
from data_adapters_batterylife import batterylife_cell_ids, iterate_batterylife_cycles
from sequence_features import build_dataset_tensors, compute_channel_norm_stats, apply_channel_norm
from stage7_common import (
    oxford_cell_ids, iterate_oxford_cycles, hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids_soh_valid, iterate_xjtu_cycles,
)
from models.ica_encoder import ICAEncoder
from train_deep_models import train_one_model
from split_utils import battery_level_split
from researchpass_partA_common import base_feature_cols, extended_feature_cols, load_base_model, load_extended_model

BATTERYLIFE_LOWER = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                     "stanford", "stanford_2", "isu_ilcc", "tongji"]
BATTERYLIFE_RAW = {"ul_pur": "UL_PUR", "hnei": "HNEI", "snl": "SNL", "mich": "MICH",
                   "mich_exp": "MICH_EXP", "rwth": "RWTH", "stanford": "Stanford",
                   "stanford_2": "Stanford_2", "isu_ilcc": "ISU_ILCC", "tongji": "Tongji"}
ALL_SOURCES = ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_LOWER
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
ICA_CHANNEL_SLICE = slice(3, 6)
EMBED_DIM = 16
SEED = 42
N_BOOT = 1000
TEST_EVERY = 5


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
    print("=== Toolkit Phase 2: multi-source ICA encoder + XGBoost-fusion + OC-SVM candidate ===")

    # ---------- 1. RAW TENSORS, every source, every cycle ----------
    print("\n[phase2] === step 1: building raw 200-bin/6-channel tensors for all 16 sources ===")
    all_X, all_soh, all_ds, all_bid, all_cyc = [], [], [], [], []

    t_s = time.time()
    # stage4_pool's own loader doesn't carry per-cycle idxs, needed to join
    # against fusion embeddings later - rebuilt directly via the same
    # iterate_fn + build_dataset_tensors path every other source below uses,
    # rather than mixing two different tensor-loading conventions.
    from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
    _ORIGINAL_NASA = ["B0005", "B0006", "B0007", "B0018"]
    for cid in _ORIGINAL_NASA:
        cycles = list(iterate_nasa_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        all_X.append(X); all_soh.append(soh)
        all_ds += ["NASA"] * len(soh); all_bid += [cid] * len(soh); all_cyc += list(idxs)
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    for entry in mit_subset:
        gid = entry["global_id"]
        cycles = list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        all_X.append(X); all_soh.append(soh)
        all_ds += ["MIT"] * len(soh); all_bid += [gid] * len(soh); all_cyc += list(idxs)
    from stage4_recovered_batteries import ALL_RECOVERED, get_recovered_battery
    for ds_name, bid in ALL_RECOVERED:
        kept_cycles, soh_map, rul_map, eol_cycle, censored = get_recovered_battery(ds_name, bid)
        X, soh, rul, idxs, _ = build_dataset_tensors(kept_cycles)
        if X is None:
            continue
        all_X.append(X); all_soh.append(soh)
        all_ds += [ds_name] * len(soh); all_bid += [bid] * len(soh); all_cyc += list(idxs)
    print(f"[phase2] NASA+MIT (incl. recovered): {sum(len(s) for s in all_soh)} cycles ({time.time()-t_s:.1f}s)")

    t_s = time.time()
    for cid in ["CS2_35", "CS2_36", "CS2_37"]:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        all_X.append(X); all_soh.append(soh)
        all_ds += ["CALCE"] * len(soh); all_bid += [cid] * len(soh); all_cyc += list(idxs)
    print(f"[phase2] CALCE: done ({time.time()-t_s:.1f}s)")

    for name, ids_fn, iter_fn in [("Oxford", oxford_cell_ids, iterate_oxford_cycles),
                                   ("HUST", hust_cell_ids, iterate_hust_cycles),
                                   ("XJTU", xjtu_cell_ids_soh_valid, iterate_xjtu_cycles)]:
        t_s = time.time()
        for cid in ids_fn():
            cycles = list(iter_fn(cid))
            if len(cycles) < 5:
                continue
            X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
            if X is None:
                continue
            all_X.append(X); all_soh.append(soh)
            all_ds += [name] * len(soh); all_bid += [cid] * len(soh); all_cyc += list(idxs)
        print(f"[phase2] {name}: done ({time.time()-t_s:.1f}s)")

    for source in BATTERYLIFE_LOWER:
        raw_name = BATTERYLIFE_RAW[source]
        ids = batterylife_cell_ids(raw_name)
        if not ids:
            print(f"[phase2] {source} ({raw_name}): no local files - skipped")
            continue
        t_s = time.time()
        n_cyc_source = 0
        for cid in ids:
            cycles = list(iterate_batterylife_cycles(raw_name, cid))
            if len(cycles) < 5:
                continue
            try:
                X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
            except Exception as e:
                print(f"[phase2]   {source}/{cid}: tensor build FAILED ({type(e).__name__}) - skipped")
                continue
            if X is None:
                continue
            all_X.append(X); all_soh.append(soh)
            all_ds += [source] * len(soh); all_bid += [f"{source}::{cid}"] * len(soh); all_cyc += list(idxs)
            n_cyc_source += len(soh)
        print(f"[phase2] {source}: {n_cyc_source} cycles ({time.time()-t_s:.1f}s)")

    X_all = np.concatenate(all_X).astype(np.float32)
    soh_all = np.concatenate(all_soh).astype(np.float32)
    ds_all = np.array(all_ds)
    bid_all = np.array(all_bid)
    cyc_all = np.array(all_cyc)
    print(f"\n[phase2] TOTAL: {len(soh_all)} cycles, {len(set(bid_all.tolist()))} batteries, "
          f"{len(set(ds_all.tolist()))} sources. Tensor build took {(time.time()-t0)/60:.1f} min so far.")

    # ---------- 2. battery-level stratified split ----------
    dataset_of = dict(zip(bid_all, ds_all))
    unique_bids = sorted(set(bid_all.tolist()))
    train_ids, test_ids = battery_level_split(unique_bids, test_every=TEST_EVERY, dataset_of=dataset_of)
    train_mask = np.isin(bid_all, train_ids)
    test_mask = ~train_mask
    print(f"[phase2] split: {len(train_ids)} train batteries / {len(test_ids)} test batteries")

    # ---------- 3. channel norm on TRAIN split ----------
    norm_stats = compute_channel_norm_stats(X_all[train_mask], clip_percentile=1.0)
    X_norm = apply_channel_norm(X_all, norm_stats)

    # ---------- 4. train ICA encoder ----------
    print("\n[phase2] === step 2: training the multi-source ICA encoder ===")
    train_bid_set = set(train_ids)
    n_val_batt = max(1, len(train_ids) // 5)
    val_ids_set = set(sorted(train_ids)[-n_val_batt:])
    fit_ids_set = train_bid_set - val_ids_set
    fit_mask = np.isin(bid_all, list(fit_ids_set))
    val_mask = np.isin(bid_all, list(val_ids_set))

    X_fit_ica = X_norm[fit_mask][:, :, ICA_CHANNEL_SLICE]
    y_fit = soh_all[fit_mask]
    X_val_ica = X_norm[val_mask][:, :, ICA_CHANNEL_SLICE]
    y_val = soh_all[val_mask]
    print(f"[phase2] encoder fit cycles={len(X_fit_ica)}, val cycles={len(X_val_ica)}")

    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder, hist = train_one_model("ICAEncoder-multisource", encoder, X_fit_ica, y_fit, X_val_ica, y_val,
                                     epochs=25, patience=6)
    torch.save(encoder.state_dict(), ROOT / "models" / "_candidate_ica_encoder.pt")
    print("[phase2] saved models/_candidate_ica_encoder.pt")

    encoder.eval()
    with torch.no_grad():
        embeddings = encoder.encode(torch.tensor(X_norm[:, :, ICA_CHANNEL_SLICE])).numpy()
    fusion_df = pd.DataFrame({"dataset": ds_all, "battery_id": bid_all, "cycle_idx": cyc_all})
    for i in range(EMBED_DIM):
        fusion_df[f"fusion_{i}"] = embeddings[:, i]
    fusion_df.to_csv(PROC_DIR / "fusion_embeddings_multisource.csv", index=False)
    print(f"[phase2] saved {len(fusion_df)} NEW embeddings to fusion_embeddings_multisource.csv")

    # ---------- 5. merge new embeddings onto each source's EXISTING HI columns ----------
    print("\n[phase2] === step 3: rebuilding the pool with NEW fusion embeddings, OLD (validated) HI features ===")
    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_full = add_reformulated_duration_features(hi_full)
    feature_cols_base = canonical_feature_cols(reformulated=True)
    hi_cols_needed = feature_cols_base + ["cycle_idx", "SOH", "battery_id", "dataset"]

    # OLD (currently-deployed-encoder) fusion embeddings, captured PER
    # SOURCE from each source's own already-existing merged file - NOT a
    # central fusion_embeddings.csv lookup, which only ever covers NASA+
    # MIT (train_fusion_encoder.py's own documented scope). Using a
    # single central-CSV lookup for every source would silently find
    # nothing for the other 14 and skip the gate's routed-model
    # comparison for them entirely - caught before running, not after.
    OLD_FUSION_COLS = [f"fusion_{i}" for i in range(16)]
    old_fusion_by_source = {}

    sources_hi = {}
    sources_hi["NASA"] = hi_full[hi_full["dataset"] == "NASA"][hi_cols_needed].copy()
    sources_hi["MIT"] = hi_full[hi_full["dataset"] == "MIT"][hi_cols_needed].copy()
    nasa_mit_old_fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    old_fusion_by_source["NASA"] = nasa_mit_old_fusion[nasa_mit_old_fusion["dataset"] == "NASA"][["battery_id", "cycle_idx"] + OLD_FUSION_COLS]
    old_fusion_by_source["MIT"] = nasa_mit_old_fusion[nasa_mit_old_fusion["dataset"] == "MIT"][["battery_id", "cycle_idx"] + OLD_FUSION_COLS]

    calce_merged = build_calce_merged(hi_full)
    sources_hi["CALCE"] = calce_merged[feature_cols_base + ["cycle_idx", "SOH", "battery_id"]].copy()
    sources_hi["CALCE"]["dataset"] = "CALCE"
    old_fusion_by_source["CALCE"] = calce_merged[["battery_id", "cycle_idx"] + OLD_FUSION_COLS]

    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"), ("HUST", "stage5_1_hust_merged.parquet"),
                        ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        df = pd.read_parquet(PROC_DIR / fname)
        sources_hi[name] = df[feature_cols_base + ["cycle_idx", "SOH", "battery_id"]].copy()
        sources_hi[name]["dataset"] = name
        old_fusion_by_source[name] = df[["battery_id", "cycle_idx"] + OLD_FUSION_COLS]

    for source in BATTERYLIFE_LOWER:
        if source == "tongji":
            path = PROC_DIR / "batterylife_tongji_merged.parquet"
        else:
            path = PROC_DIR / f"batterylife_{source}_merged.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        sub = df[feature_cols_base + ["cycle_idx", "SOH", "battery_id"]].copy()
        sub["battery_id"] = source + "::" + sub["battery_id"].astype(str)
        sub["dataset"] = source
        sources_hi[source] = sub
        old_fusion_sub = df[["battery_id", "cycle_idx"] + OLD_FUSION_COLS].copy()
        old_fusion_sub["battery_id"] = source + "::" + old_fusion_sub["battery_id"].astype(str)
        old_fusion_by_source[source] = old_fusion_sub

    fusion_cols_new = [f"fusion_{i}" for i in range(EMBED_DIM)]
    all_cols = feature_cols_base + ["cycle_idx"] + fusion_cols_new

    pooled_frames = []
    for name, df in sources_hi.items():
        fsub = fusion_df[fusion_df["dataset"] == name][["battery_id", "cycle_idx"] + fusion_cols_new]
        merged = pd.merge(df, fsub, on=["battery_id", "cycle_idx"], how="inner")
        merged["_source"] = name
        pooled_frames.append(merged)
        print(f"[phase2]   {name}: {len(df)} HI rows -> {len(merged)} rows after fusion merge")
    pool = pd.concat(pooled_frames, ignore_index=True)
    print(f"[phase2] pooled: {len(pool)} rows, {pool['battery_id'].nunique()} batteries")

    # ---------- 6. train candidate XGBoost-fusion ----------
    print("\n[phase2] === step 4: training candidate XGBoost-fusion ===")
    n_fusion = len(fusion_cols_new)
    monotone = tuple([0] * len(feature_cols_base) + [-1] + [0] * n_fusion)
    pool_train_mask = pool["battery_id"].isin(train_ids).to_numpy()
    pool_test_mask = ~pool_train_mask

    X_pool = pool[all_cols].to_numpy(dtype=float, copy=True)
    X_pool = np.where(np.isinf(X_pool), np.nan, X_pool)
    col_medians = np.nanmedian(X_pool[pool_train_mask], axis=0)
    inds = np.where(np.isnan(X_pool))
    X_pool[inds] = np.take(col_medians, inds[1])
    y_pool = pool["SOH"].to_numpy(dtype=float)

    xgb_model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED, n_jobs=-1, reg_lambda=1.0,
                              monotone_constraints=monotone)
    xgb_model.fit(X_pool[pool_train_mask], y_pool[pool_train_mask])
    xgb_model.save_model(str(ROOT / "models" / "_candidate_multisource.json"))
    print("[phase2] saved models/_candidate_multisource.json")

    pred_candidate = xgb_model.predict(X_pool[pool_test_mask])
    y_test = y_pool[pool_test_mask]
    bids_test = pool.loc[pool_test_mask, "battery_id"].to_numpy()
    src_test = pool.loc[pool_test_mask, "_source"].to_numpy()

    # ---------- 7. GATE: candidate vs CURRENT DEPLOYED ROUTED model, same held-out rows ----------
    print("\n[phase2] === step 5: gate table vs. current deployed routed model ===")
    base_cols_full = base_feature_cols()
    ext_cols_full = extended_feature_cols()
    base_model = load_base_model()
    ext_model = load_extended_model()
    from researchpass_partA_common import load_base_pool_and_split, load_extended_pool_and_split, fit_medians
    merged_b, hi_full_b, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_e, hi_full_e, hi_full_raw_e, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_medians = fit_medians(merged_b, train_mask_b, base_cols_full)
    ext_medians = fit_medians(merged_e, train_mask_e, ext_cols_full)
    from stage5_extended_reformulation import add_scv_matd_viect_reformulated

    test_rows = pool.loc[pool_test_mask].reset_index(drop=True)
    pred_routed = np.full(len(test_rows), np.nan)
    routed_which = np.empty(len(test_rows), dtype=object)
    for src in test_rows["_source"].unique():
        src_mask = (test_rows["_source"] == src).to_numpy()
        rows_src = test_rows.loc[src_mask].copy()
        old_fusion_match = old_fusion_by_source.get(src)
        if old_fusion_match is None or len(old_fusion_match) == 0:
            # old encoder's embeddings were never computed for this source at all - skip routed comparison for it
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
            ci_r = bootstrap_ci(y_s[valid_r], pred_r_s[valid_r], bids_s[valid_r])
        else:
            m_r = {"r2": np.nan, "rmse": np.nan, "mae": np.nan}
            ci_r = {"r2": (np.nan, np.nan)}

        which = routed_which[np.where(mask)[0][0]]
        print(f"[phase2] {name} (n={mask.sum()}, {len(set(bids_s))} batt): CANDIDATE R2={m_c['r2']:.4f} "
              f"[{ci_c['r2'][0]:.4f},{ci_c['r2'][1]:.4f}] RMSE={m_c['rmse']:.4f} MAE={m_c['mae']:.4f} | "
              f"ROUTED({which}) R2={m_r['r2']:.4f} RMSE={m_r.get('rmse',np.nan):.4f} MAE={m_r.get('mae',np.nan):.4f} -> "
              f"{'CANDIDATE WINS' if (np.isfinite(m_r['r2']) and m_c['r2']>m_r['r2']) else ('N/A' if not np.isfinite(m_r['r2']) else 'ROUTED WINS')}")
        gate_rows.append({"source": name, "n": int(mask.sum()), "n_batteries": len(set(bids_s)),
                          "candidate_r2": m_c["r2"], "candidate_r2_ci_lo": ci_c["r2"][0], "candidate_r2_ci_hi": ci_c["r2"][1],
                          "candidate_rmse": m_c["rmse"], "candidate_mae": m_c["mae"],
                          "routed_r2": m_r["r2"], "routed_rmse": m_r.get("rmse", np.nan), "routed_mae": m_r.get("mae", np.nan),
                          "routed_model_used": which,
                          "candidate_wins": bool(np.isfinite(m_r["r2"]) and m_c["r2"] > m_r["r2"])})

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(OUT_DIR / "toolkit_phase2_gate_table.csv", index=False)

    # ---------- 8. OC-SVM retrain ----------
    print("\n[phase2] === step 6: retraining OC-SVM on the multi-source pool ===")
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
    ocsvm_medians = np.nanmedian(Xo, axis=0)
    inds4 = np.where(np.isnan(Xo))
    Xo[inds4] = np.take(ocsvm_medians, inds4[1])
    ocsvm_scaler = StandardScaler().fit(Xo)
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(ocsvm_scaler.transform(Xo))
    with open(ROOT / "models" / "_candidate_ocsvm.pkl", "wb") as f:
        pickle.dump(ocsvm, f)
    with open(ROOT / "models" / "_candidate_ocsvm_scaler.pkl", "wb") as f:
        pickle.dump(ocsvm_scaler, f)
    print(f"[phase2] saved models/_candidate_ocsvm.pkl (trained on {len(fit_df_sampled)} cycles, "
          f"{len(fit_ids_set)} fit batteries)")

    # flag rate on EXTERNAL (non-NASA/MIT) held-out batteries, candidate vs
    # deployed OC-SVM - a genuine before/vs/after, both scored on the SAME
    # rows (the deployed OC-SVM uses base_cols_full + its OWN OLD fusion
    # embeddings, matching how it's actually loaded/scored in production).
    with open(ROOT / "models" / "ocsvm_model.pkl", "rb") as f:
        deployed_ocsvm = pickle.load(f)
    with open(ROOT / "models" / "ocsvm_scaler.pkl", "rb") as f:
        deployed_scaler = pickle.load(f)
    deployed_ocsvm_cols = json.loads((PROC_DIR / "ocsvm_feature_cols.json").read_text())

    ext_test = test_rows[~test_rows["_source"].isin({"NASA", "MIT"})].reset_index(drop=True)
    Xo_cand = ext_test[ocsvm_feature_cols].to_numpy(dtype=float, copy=True)
    Xo_cand = np.where(np.isinf(Xo_cand), np.nan, Xo_cand)
    inds5 = np.where(np.isnan(Xo_cand))
    Xo_cand[inds5] = np.take(ocsvm_medians, inds5[1])
    cand_flags = ocsvm.predict(ocsvm_scaler.transform(Xo_cand)) == -1
    cand_flag_rate = float(cand_flags.mean())
    print(f"[phase2] CANDIDATE OC-SVM flags {cand_flag_rate:.1%} of {len(ext_test)} external held-out cycles "
          f"({ext_test['battery_id'].nunique()} batteries) as anomalous")

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
            col_med_fallback = np.nanmedian(X_d, axis=0)
            col_med_fallback = np.where(np.isnan(col_med_fallback), 0.0, col_med_fallback)
            X_d[inds6] = np.take(col_med_fallback, inds6[1])
        flags = deployed_ocsvm.predict(deployed_scaler.transform(X_d)) == -1
        deployed_flags_per_src.append(flags)
    if deployed_flags_per_src:
        deployed_flags_all = np.concatenate(deployed_flags_per_src)
        deployed_flag_rate = float(deployed_flags_all.mean())
        print(f"[phase2] DEPLOYED OC-SVM (currently in production) flags {deployed_flag_rate:.1%} of the "
              f"SAME external held-out cycles as anomalous")
        print(f"[phase2] OC-SVM flag rate on external batteries: deployed={deployed_flag_rate:.1%} -> "
              f"candidate={cand_flag_rate:.1%} ({'MORE' if cand_flag_rate>deployed_flag_rate else 'FEWER'} "
              f"flagged after multi-source retraining)")
    else:
        print("[phase2] could not score the deployed OC-SVM on the same rows (no old fusion embeddings matched)")

    print(f"\n[phase2] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")
    print("[phase2] STOPPING here per instruction - candidates saved, NOT wired into app.py/live_inference.py.")
    print("\n=== FULL GATE TABLE ===")
    print(gate_df.to_string(index=False))
    n_wins = int(gate_df["candidate_wins"].sum())
    n_comparable = int(gate_df["routed_r2"].notna().sum())
    print(f"\n[phase2] GATE SUMMARY: candidate wins on {n_wins}/{n_comparable} comparable sources.")


if __name__ == "__main__":
    main()
