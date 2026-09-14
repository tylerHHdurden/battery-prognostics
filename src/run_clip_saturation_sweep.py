"""
Systematic follow-up to the B0053 clip-floor discovery: does the
training-pool-fit percentile clip (sequence_features.compute_channel_
norm_stats/apply_channel_norm) saturate any OTHER channel/battery, and
critically, does it saturate CALCE - a potential confound inside this
project's central domain-shift finding if so.

No retraining. Pure analysis over already-existing raw (pre-clip)
tensors and the already-fit channel_norm_stats{,_expanded}.json files.

Channels (fixed order, sequence_features.CHANNEL_NAMES):
    0 V_t, 1 I_t, 2 T_t, 3 dQdV, 4 dVdQ, 5 dIdV
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sequence_features import build_dataset_tensors, CHANNEL_NAMES
from data_adapters import iterate_calce_cycles

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]

def per_battery_channel_stats(X, lo, hi):
    """X: (n_cycles, n_bins) raw values for ONE channel of ONE battery.
    Returns (frac_cycles_fully_saturated, frac_values_clipped).

    IMPORTANT (caught by direct verification before trusting this
    script's output - see DEVELOPMENT_LOG.md): a first version used
    std() < 1e-6 on the (float32) clipped array to detect a fully-
    saturated cycle. For B0053's T_t channel specifically (raw values
    ~12-22, clip bounds ~28-41), every value clips to the exact same
    float32 constant (verified: clipped.min()==clipped.max() bit-for-
    bit), but float32 std() of a same-valued array is NOT always
    exactly 0.0 due to accumulation rounding - it returned ~1.9e-6,
    just above the 1e-6 threshold, silently producing a false "not
    saturated" negative. Fixed by (a) casting to float64 first and
    (b) using max-min RANGE rather than std - range is exact for a
    clipped-to-constant array regardless of float precision, and a
    1e-4 threshold leaves a wide margin above any realistic float64
    rounding noise at these value scales."""
    n_cycles = X.shape[0]
    clipped = np.clip(X.astype(np.float64), lo, hi)
    per_cycle_range = clipped.max(axis=1) - clipped.min(axis=1)
    frac_fully_sat = float((per_cycle_range < 1e-4).mean())
    frac_clipped_values = float(((X <= lo) | (X >= hi)).mean())
    return frac_fully_sat, frac_clipped_values


def sweep_pool(battery_data: dict, stats: list, pool_name: str):
    rows = []
    for bid, entry in battery_data.items():
        X = entry[0] if isinstance(entry, tuple) else entry
        for c, cname in enumerate(CHANNEL_NAMES):
            lo, hi = stats[c]["lo"], stats[c]["hi"]
            frac_sat, frac_clip = per_battery_channel_stats(X[:, :, c], lo, hi)
            rows.append({"pool": pool_name, "battery_id": bid, "channel": cname,
                         "frac_cycles_fully_saturated": frac_sat,
                         "frac_values_clipped": frac_clip, "n_cycles": X.shape[0]})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()

    # ---------------------------------------------------------------
    # Item 1a: EXPANDED pool sweep (the pool B0053's own saturation was
    # found in) against channel_norm_stats_expanded.json
    # ---------------------------------------------------------------
    print("[clip-sweep] === loading EXPANDED pool raw tensors (disk cache) ===")
    cache_path = PROC_DIR / "_expanded_battery_tensors_cache.pkl"
    with open(cache_path, "rb") as f:
        battery_data_expanded = pickle.load(f)
    print(f"[clip-sweep] loaded {len(battery_data_expanded)} batteries in {time.time()-t0:.1f}s")

    stats_expanded = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    df_expanded = sweep_pool(battery_data_expanded, stats_expanded, "expanded_204")

    print("\n[clip-sweep] === EXPANDED pool: any battery >5% fully-saturated cycles on any channel ===")
    hits_expanded = df_expanded[df_expanded["frac_cycles_fully_saturated"] > 0.05].sort_values(
        "frac_cycles_fully_saturated", ascending=False)
    print(hits_expanded.to_string(index=False) if len(hits_expanded) else "  (none)")

    print("\n[clip-sweep] === EXPANDED pool: any battery >10% of VALUES clipped on any channel "
          "(partial saturation, even if not whole-cycle) ===")
    partial_expanded = df_expanded[df_expanded["frac_values_clipped"] > 0.10].sort_values(
        "frac_values_clipped", ascending=False)
    print(partial_expanded.to_string(index=False) if len(partial_expanded) else "  (none)")

    # ---------------------------------------------------------------
    # Item 1b: ORIGINAL 32-battery pool sweep against channel_norm_stats.json
    # (the canonical pool most directly relevant to near-term Stage 4)
    # ---------------------------------------------------------------
    print("\n[clip-sweep] === loading ORIGINAL 32-battery pool raw tensors ===")
    from train_deep_models import load_all_battery_tensors
    battery_data_orig = load_all_battery_tensors()
    stats_orig = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    df_orig = sweep_pool(battery_data_orig, stats_orig, "original_32")

    print("\n[clip-sweep] === ORIGINAL 32-pool: any battery >5% fully-saturated cycles on any channel ===")
    hits_orig = df_orig[df_orig["frac_cycles_fully_saturated"] > 0.05].sort_values(
        "frac_cycles_fully_saturated", ascending=False)
    print(hits_orig.to_string(index=False) if len(hits_orig) else "  (none)")

    print("\n[clip-sweep] === ORIGINAL 32-pool: any battery >10% of VALUES clipped on any channel ===")
    partial_orig = df_orig[df_orig["frac_values_clipped"] > 0.10].sort_values(
        "frac_values_clipped", ascending=False)
    print(partial_orig.to_string(index=False) if len(partial_orig) else "  (none)")

    # ---------------------------------------------------------------
    # Item 2: CALCE per-cell, per-channel saturation, against
    # channel_norm_stats.json (the stats the CANONICAL Stage 1-3
    # CALCE-evaluation pipeline actually applies - stage1_common.
    # build_calce_tensors calls apply_channel_norm with THIS exact file)
    # ---------------------------------------------------------------
    print("\n[clip-sweep] === CALCE: building raw (pre-clip) tensors per cell ===")
    calce_rows = []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        print(f"[clip-sweep] CALCE/{cid}: {X.shape[0]} cycles")
        for c, cname in enumerate(CHANNEL_NAMES):
            lo, hi = stats_orig[c]["lo"], stats_orig[c]["hi"]
            frac_sat, frac_clip = per_battery_channel_stats(X[:, :, c], lo, hi)
            raw_vals = X[:, :, c]
            calce_rows.append({
                "cell": cid, "channel": cname, "clip_lo": lo, "clip_hi": hi,
                "raw_min": float(raw_vals.min()), "raw_max": float(raw_vals.max()),
                "raw_mean": float(raw_vals.mean()),
                "frac_cycles_fully_saturated": frac_sat,
                "frac_values_clipped": frac_clip, "n_cycles": X.shape[0],
            })
    df_calce = pd.DataFrame(calce_rows)
    print("\n[clip-sweep] === CALCE per-cell, per-channel saturation (full detail, not pooled) ===")
    print(df_calce.to_string(index=False))

    # ---------------------------------------------------------------
    # Save everything
    # ---------------------------------------------------------------
    df_expanded.to_csv(OUT_DIR / "clip_saturation_sweep_expanded_pool.csv", index=False)
    df_orig.to_csv(OUT_DIR / "clip_saturation_sweep_original_pool.csv", index=False)
    df_calce.to_csv(OUT_DIR / "clip_saturation_sweep_calce.csv", index=False)
    print(f"\n[clip-sweep] saved outputs/clip_saturation_sweep_{{expanded_pool,original_pool,calce}}.csv")
    print(f"[clip-sweep] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
