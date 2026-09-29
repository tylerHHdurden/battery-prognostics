"""
Toolkit Phase 2e: OC-SVM sanity check on the candidate detector.

(a) Flag rate on IN-DOMAIN held-out batteries (NASA+MIT test split of
    the 16-source pool) - should sit near nu=0.05, the detector's own
    contamination assumption.
(b) Flag rate on EACH external source individually (not the single
    aggregate Phase 2c already reported).
(c) Flag rate on 5 SYNTHETIC CORRUPTIONS of real in-domain cycles:
    Gaussian noise at 1x/5x a disclosed BMS-level assumption (voltage
    std=2mV/10mV, current std=20mA/100mA - typical consumer/EV BMS ADC
    noise floors, stated explicitly since no exact spec was given),
    voltage/current columns swapped, discharge_capacity scaled x10 (a
    plausible mAh/Ah unit-confusion error), and a truncated cycle (last
    60% of the discharge phase dropped).

DECISION (per instruction, applied and reported): if (c) is flagged
well above (a), KEEP the candidate OC-SVM as the in-app anomaly flag.
If not, the recommendation is to REPLACE the in-app anomaly flag with
the nearest-source trust report and move OC-SVM to the Research
section - stated as a recommendation here, not itself an app change
(Phase 3 wiring is separate).

DISCLOSED APPROXIMATION: corrupted cycles are re-encoded using the
OLD (currently-deployed) `channel_norm_stats.json`, not the candidate
encoder's own exact training-time stats (never saved to disk before
Phase 2's original run crashed, and recomputing them exactly would
require re-running the ~25-minute raw-tensor-build step just for this
sanity check). Channel normalization stats are per-channel scale/
center constants (median/IQR-style) - unlikely to differ by enough to
change the QUALITATIVE flag-rate comparison this check cares about,
but stated as an approximation, not hidden.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage1_common import canonical_feature_cols, add_reformulated_duration_features, build_calce_merged, OUT_DIR, PROC_DIR, ROOT
from split_utils import battery_level_split
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from health_indicators import compute_health_indicators
from sequence_features import get_cycle_tensor, apply_channel_norm
from models.ica_encoder import ICAEncoder
import pickle
import json

BATTERYLIFE_LOWER = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                     "stanford", "stanford_2", "isu_ilcc", "tongji"]
ALL_SOURCES = ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_LOWER
TEST_EVERY = 5
EMBED_DIM = 16
ICA_CHANNEL_SLICE = slice(3, 6)
DURATION_FEATURES = ["ICHV", "TEVD", "TEVI"]
BASELINE_CYCLE = 10

# Disclosed BMS-noise assumption (stated explicitly, not hidden)
V_NOISE_1X, I_NOISE_1X = 0.002, 0.02   # 2 mV, 20 mA
V_NOISE_5X, I_NOISE_5X = 0.010, 0.10   # 10 mV, 100 mA


def main():
    t0 = time.time()
    print("=== Toolkit Phase 2e: OC-SVM sanity check ===")

    with open(ROOT / "models" / "_candidate_ocsvm.pkl", "rb") as f:
        ocsvm = pickle.load(f)
    with open(ROOT / "models" / "_candidate_ocsvm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(PROC_DIR / "_candidate_ocsvm_medians.json") as f:
        med_info = json.load(f)
    ocsvm_cols, ocsvm_medians = med_info["cols"], np.array(med_info["medians"])

    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder.load_state_dict(torch.load(ROOT / "models" / "_candidate_ica_encoder.pt"))
    encoder.eval()
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())  # disclosed approximation, see docstring

    # ---------- rebuild the pool (same as Phase 2c) ----------
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

    dataset_of = dict(zip(pool["battery_id"], pool["_source"]))
    unique_bids = sorted(pool["battery_id"].unique().tolist())
    train_ids, test_ids = battery_level_split(unique_bids, test_every=TEST_EVERY, dataset_of=dataset_of)
    test_rows = pool[pool["battery_id"].isin(test_ids)].reset_index(drop=True)

    def score(df_rows):
        X = df_rows[ocsvm_cols].to_numpy(dtype=float, copy=True)
        X = np.where(np.isinf(X), np.nan, X)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(ocsvm_medians, inds[1])
        return ocsvm.predict(scaler.transform(X)) == -1

    # ---------- (a) in-domain held-out ----------
    print("\n=== (a) in-domain (NASA+MIT) held-out flag rate ===")
    indomain_rows = test_rows[test_rows["_source"].isin({"NASA", "MIT"})]
    indomain_flags = score(indomain_rows)
    print(f"[phase2e] NASA+MIT held-out: {indomain_flags.mean():.1%} flagged (n={len(indomain_rows)}, "
          f"target ~nu=5.0%)")

    # ---------- (b) each external source individually ----------
    print("\n=== (b) per-source external flag rate ===")
    rows_out = [{"group": "in-domain (NASA+MIT)", "flag_rate": float(indomain_flags.mean()), "n": len(indomain_rows)}]
    for name in ALL_SOURCES:
        if name in {"NASA", "MIT"}:
            continue
        sub = test_rows[test_rows["_source"] == name]
        if len(sub) == 0:
            continue
        flags = score(sub)
        print(f"[phase2e] {name}: {flags.mean():.1%} flagged (n={len(sub)}, {sub['battery_id'].nunique()} batteries)")
        rows_out.append({"group": name, "flag_rate": float(flags.mean()), "n": len(sub)})

    # ---------- (c) synthetic corruptions ----------
    print("\n=== (c) synthetic corruptions of real in-domain cycles ===")
    print(f"[phase2e] BMS-noise assumption (disclosed): 1x = {V_NOISE_1X*1000:.0f}mV/{I_NOISE_1X*1000:.0f}mA, "
          f"5x = {V_NOISE_5X*1000:.0f}mV/{I_NOISE_5X*1000:.0f}mA")

    rng = np.random.default_rng(42)
    real_cycles = []
    for cid in ["B0005", "B0006", "B0007", "B0018"]:
        cycles = list(iterate_nasa_cycles(cid))
        mid = cycles[len(cycles) // 2] if cycles else None
        if mid is not None:
            real_cycles.append((cid, mid))
    print(f"[phase2e] using {len(real_cycles)} real NASA mid-life cycles as corruption source material")

    def cycle_to_feature_row(cycle, bid, baseline_his):
        his = compute_health_indicators(cycle)
        row = {}
        for feat in feature_cols_base:
            base_feat = feat.replace("_rel", "")
            if feat.endswith("_rel") and base_feat in DURATION_FEATURES:
                base_val = baseline_his.get(base_feat, np.nan)
                row[feat] = his[base_feat] / base_val if (base_val and np.isfinite(base_val) and abs(base_val) > 1e-6) else np.nan
            else:
                row[feat] = his.get(feat, np.nan)
        row["cycle_idx"] = cycle["cycle_idx"]
        x_raw = get_cycle_tensor(cycle, n_bins=200)
        if x_raw is None:
            return None
        x_norm = apply_channel_norm(x_raw[None].astype(np.float32), norm_stats)[0]
        with torch.no_grad():
            emb = encoder.encode(torch.tensor(x_norm[None, :, ICA_CHANNEL_SLICE])).numpy()[0]
        for i in range(EMBED_DIM):
            row[f"fusion_{i}"] = emb[i]
        return row

    def corrupt_noise(cycle, v_std, i_std):
        c2 = {"cycle_idx": cycle["cycle_idx"], "discharge_capacity": cycle["discharge_capacity"]}
        for phase in ("charge", "discharge"):
            p = cycle[phase]
            c2[phase] = {"t": p["t"], "V": p["V"] + rng.normal(0, v_std, len(p["V"])),
                        "I": p["I"] + rng.normal(0, i_std, len(p["I"])), "T": p["T"]}
        return c2

    def corrupt_vi_swap(cycle):
        c2 = {"cycle_idx": cycle["cycle_idx"], "discharge_capacity": cycle["discharge_capacity"]}
        for phase in ("charge", "discharge"):
            p = cycle[phase]
            c2[phase] = {"t": p["t"], "V": p["I"].copy(), "I": p["V"].copy(), "T": p["T"]}
        return c2

    def corrupt_capacity_x10(cycle):
        c2 = dict(cycle)
        c2["discharge_capacity"] = cycle["discharge_capacity"] * 10
        return c2

    def corrupt_truncated(cycle):
        c2 = {"cycle_idx": cycle["cycle_idx"], "discharge_capacity": cycle["discharge_capacity"] * 0.4}
        c2["charge"] = cycle["charge"]
        p = cycle["discharge"]
        cut = max(2, int(len(p["V"]) * 0.4))
        c2["discharge"] = {"t": p["t"][:cut], "V": p["V"][:cut], "I": p["I"][:cut],
                           "T": p["T"][:cut] if p["T"] is not None else None}
        return c2

    corruption_defs = [
        ("gaussian_noise_1x_bms", lambda c: corrupt_noise(c, V_NOISE_1X, I_NOISE_1X)),
        ("gaussian_noise_5x_bms", lambda c: corrupt_noise(c, V_NOISE_5X, I_NOISE_5X)),
        ("voltage_current_swapped", corrupt_vi_swap),
        ("capacity_scaled_x10", corrupt_capacity_x10),
        ("truncated_cycle", corrupt_truncated),
    ]

    for corrupt_name, corrupt_fn in corruption_defs:
        flags = []
        for bid, cycle in real_cycles:
            try:
                baseline_cycles = list(iterate_nasa_cycles(bid))
                baseline_cycle = next((c for c in baseline_cycles if c["cycle_idx"] == BASELINE_CYCLE), baseline_cycles[0])
                baseline_his = compute_health_indicators(baseline_cycle)
                corrupted = corrupt_fn(cycle)
                row = cycle_to_feature_row(corrupted, bid, baseline_his)
                if row is None:
                    continue
                X = np.array([[row.get(c, np.nan) for c in ocsvm_cols]], dtype=float)
                X = np.where(np.isinf(X), np.nan, X)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(ocsvm_medians, inds[1])
                flag = ocsvm.predict(scaler.transform(X))[0] == -1
                flags.append(flag)
            except Exception as e:
                print(f"[phase2e]   {corrupt_name}/{bid}: FAILED ({type(e).__name__}: {e}) - skipped")
        if flags:
            rate = float(np.mean(flags))
            print(f"[phase2e] {corrupt_name}: {rate:.1%} flagged (n={len(flags)})")
            rows_out.append({"group": f"corruption_{corrupt_name}", "flag_rate": rate, "n": len(flags)})

    result_df = pd.DataFrame(rows_out)
    result_df.to_csv(OUT_DIR / "toolkit_phase2e_ocsvm_sanity_check.csv", index=False)

    print("\n=== SUMMARY ===")
    print(result_df.to_string(index=False))

    indomain_rate = result_df[result_df["group"] == "in-domain (NASA+MIT)"]["flag_rate"].iloc[0]
    corruption_rows = result_df[result_df["group"].str.startswith("corruption_")]
    if len(corruption_rows):
        min_corruption_rate = corruption_rows["flag_rate"].min()
        well_above = min_corruption_rate > indomain_rate + 0.15  # "well above" = at least 15pp higher, disclosed threshold
        print(f"\n[phase2e] in-domain baseline flag rate: {indomain_rate:.1%}")
        print(f"[phase2e] LOWEST corruption flag rate: {min_corruption_rate:.1%}")
        print(f"[phase2e] DECISION: corruptions flagged 'well above' baseline (>15pp higher, disclosed threshold)? {well_above}")
        print(f"[phase2e] RECOMMENDATION: {'KEEP the candidate OC-SVM as the in-app anomaly flag' if well_above else 'REPLACE the in-app anomaly flag with the nearest-source trust report; move OC-SVM to Research'}")

    print(f"\n[phase2e] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
