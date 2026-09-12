"""
Targeted improvement pass, Part 1, item 1: battery-level outlier
detection on the EARLY-CYCLE capacity trace, run before any retraining.

Root cause already diagnosed (Dataset Expansion session): NASA/B0053's
raw capacity trace starts at 0.000Ah (visible in nasa_validation.txt) -
the same flavor of low-rate-characterization-phase artifact that got 14
OTHER batteries excluded outright (>110% SOH ceiling), except B0053's
overall SOH never crossed that ceiling, so it correctly stayed in the
pool - and then went on to single-handedly wreck PiFormer's pooled
RMSE. This script checks whether a much cheaper, EARLIER check (on raw
early-cycle capacity, before any SOH/RUL computation) would have
flagged it anyway, and whether it flags anything else worth knowing
about, honestly reported either way.

Method: for every battery in the 204-battery expanded pool (PLUS the 14
already-excluded batteries and B0052, as a sanity check that a
sensible metric scores them at least as extreme, not as something
extra to re-exclude - they're excluded already, on stronger evidence),
take the discharge capacity of its first 5 logged cycles. The candidate
statistic is min(first-5-cycle capacity) - directly captures "does this
battery's trace touch near-zero early on" (B0053's specific pattern),
computed PER DATASET (NASA and MIT have different nominal capacities,
so pooling them would bias the z-score) and z-scored against the
REMAINING in-pool batteries of that same dataset (leave-one-out mean/
std, so a single extreme battery can't inflate its own denominator and
mask itself as within +-3 sigma).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

K = 5  # first-K-cycles window


def early_cycle_min_capacity_nasa(cell_id: str) -> float | None:
    caps = []
    for i, c in enumerate(iterate_nasa_cycles(cell_id)):
        caps.append(c["discharge_capacity"])
        if i + 1 >= K:
            break
    return min(caps) if caps else None


def early_cycle_min_capacity_mit(batch_file: str, cell_index: int) -> float | None:
    caps = []
    for c in iterate_mit_cycles(batch_file, cell_index, max_cycles=K):
        caps.append(c["discharge_capacity"])
    return min(caps) if caps else None


def main():
    # Same NASA raw list as train_deep_models_expanded.py, but WITHOUT
    # filtering out EXCLUDED_BATTERIES/B0052 - this script deliberately
    # checks all 34, in-pool and already-excluded alike, as a sanity
    # check on the metric itself.
    all_nasa = [
        "B0005", "B0006", "B0007", "B0018",
        "B0025", "B0026", "B0027", "B0028", "B0029", "B0030", "B0031", "B0032",
        "B0033", "B0034", "B0036", "B0038", "B0039", "B0040", "B0041", "B0042",
        "B0043", "B0044", "B0045", "B0046", "B0047", "B0048", "B0049", "B0050",
        "B0051", "B0052", "B0053", "B0054", "B0055", "B0056",
    ]
    from expanded_pool_exclusions import EXCLUDED_BATTERIES
    excluded_nasa = set(EXCLUDED_BATTERIES) | {"B0052"}

    print(f"[outlier-detect] Scanning {len(all_nasa)} NASA batteries' first {K} cycles...")
    rows = []
    for cid in all_nasa:
        try:
            m = early_cycle_min_capacity_nasa(cid)
        except Exception as e:
            print(f"[outlier-detect] NASA/{cid}: FAILED to read ({e}), skipping")
            continue
        rows.append({"dataset": "NASA", "battery_id": cid, "min_early_cap": m,
                      "in_pool": cid not in excluded_nasa})
        print(f"[outlier-detect] NASA/{cid}: min(first {K} cycles' capacity)={m:.4f}Ah "
              f"({'in-pool' if cid not in excluded_nasa else 'EXCLUDED (data-quality)'})")

    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full_raw = json.load(f)
    print(f"\n[outlier-detect] Scanning {len(mit_full_raw)} MIT cells' first {K} cycles "
          f"(this is the slow part - one h5py open per cell)...")
    for i, entry in enumerate(mit_full_raw):
        try:
            m = early_cycle_min_capacity_mit(entry["batch_file"], entry["cell_index"])
        except Exception as e:
            print(f"[outlier-detect] MIT/{entry['global_id']}: FAILED to read ({e}), skipping")
            continue
        rows.append({"dataset": "MIT", "battery_id": entry["global_id"], "min_early_cap": m,
                      "in_pool": entry["global_id"] not in EXCLUDED_BATTERIES})
        if (i + 1) % 40 == 0:
            print(f"[outlier-detect] ...{i+1}/{len(mit_full_raw)} MIT cells scanned")

    df = pd.DataFrame(rows).dropna(subset=["min_early_cap"])
    print(f"\n[outlier-detect] {len(df)} batteries scanned successfully.")

    flagged = []
    for ds in ["NASA", "MIT"]:
        pool = df[(df.dataset == ds) & (df.in_pool)].copy()
        print(f"\n[outlier-detect] === {ds}: z-scoring {len(pool)} IN-POOL batteries "
              f"(leave-one-out mean/std) ===")
        for idx, row in pool.iterrows():
            others = pool.drop(idx)["min_early_cap"]
            mu, sigma = others.mean(), others.std()
            z = (row["min_early_cap"] - mu) / sigma if sigma > 0 else 0.0
            pool.loc[idx, "z_score"] = z
        pool = pool.sort_values("z_score")
        print(pool[["battery_id", "min_early_cap", "z_score"]].to_string(index=False))
        hits = pool[pool["z_score"].abs() > 3]
        for _, r in hits.iterrows():
            print(f"[outlier-detect] *** FLAGGED (|z|>3): {ds}/{r.battery_id} "
                  f"min_early_cap={r.min_early_cap:.4f}Ah z={r.z_score:.2f} ***")
            flagged.append(r.battery_id)

        # Sanity check: do the ALREADY-EXCLUDED batteries of this dataset
        # score at least as extreme on this metric, using the IN-POOL
        # distribution as reference (not their own, so they can't hide
        # behind each other)?
        excl = df[(df.dataset == ds) & (~df.in_pool)].copy()
        if len(excl):
            mu_pool, sigma_pool = pool["min_early_cap"].mean(), pool["min_early_cap"].std()
            excl["z_vs_pool"] = (excl["min_early_cap"] - mu_pool) / sigma_pool
            print(f"\n[outlier-detect] Sanity check - {ds} ALREADY-EXCLUDED batteries "
                  f"scored against the in-pool distribution (expect these to look at "
                  f"least as extreme, confirming the metric is sensible):")
            print(excl[["battery_id", "min_early_cap", "z_vs_pool"]].sort_values("z_vs_pool").to_string(index=False))

    print(f"\n[outlier-detect] === RESULT ===")
    print(f"[outlier-detect] Batteries flagged by a PURELY early-cycle-capacity, "
          f"pre-SOH check (|z|>3, leave-one-out, per-dataset): {flagged}")
    if "B0053" in flagged:
        print("[outlier-detect] B0053 IS flagged by this check - a cheap, purely-"
              "structural early-cycle check would have caught it BEFORE any model "
              "was ever trained on it, without needing PiFormer's RMSE to blow up "
              "first.")
    else:
        print("[outlier-detect] B0053 is NOT flagged by this |z|>3 threshold - "
              "reported honestly: this specific check does not catch it, meaning "
              "B0053's problem is real but more subtle than a gross early-capacity "
              "outlier the rest of the pool doesn't also exhibit to some degree.")

    df.to_csv(OUT_DIR / "early_cycle_outlier_detection.csv", index=False)
    print(f"[outlier-detect] Saved outputs/early_cycle_outlier_detection.csv "
          f"({len(df)} rows)")
    print("[outlier-detect] DONE")


if __name__ == "__main__":
    main()
