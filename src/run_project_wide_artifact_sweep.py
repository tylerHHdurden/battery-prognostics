"""
Stage 1 final closeout, Part A: project-wide single-cycle artifact
sweep. Three separate, unrelated investigations independently found
the SAME artifact signature by accident in three different batteries -
one isolated cycle collapsing to near-zero SOH while both immediate
neighbors are ordinary (B0053 cycle 55, B0044 cycle 6, B0045 cycle 19).
No systematic sweep for this signature has ever been run.

Detector: for each battery's own SOH-vs-cycle_idx sequence (sorted),
at each interior cycle i, compute the local median of the ~5 cycles on
EACH side (window=5, excluding i itself). Flag i if:
  1. SOH[i] < (1 - DROP_FRAC) * local_median  (a large, isolated drop)
  2. BOTH immediate neighbors (i-1, i+1) are within NEIGHBOR_TOL of the
     local median (i.e. NOT part of a genuine multi-cycle collapse or
     real end-of-life decline - specifically isolated).
Threshold/window tuned empirically against the 3 known cases before
being trusted on the rest of the pool (step required by the task).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

WINDOW = 5
DROP_FRAC = 0.80  # flag if the cycle's SOH is <20% of local median
NEIGHBOR_TOL = 0.30  # each immediate neighbor must be within 30% of local median (i.e. NOT also collapsed)

KNOWN_CASES = {("NASA", "B0053"): 55, ("NASA", "B0044"): 6, ("NASA", "B0045"): 19}


def detect_isolated_drops(cyc: np.ndarray, soh: np.ndarray, window=WINDOW,
                           drop_frac=DROP_FRAC, neighbor_tol=NEIGHBOR_TOL):
    """Interior cycles need BOTH neighbors to confirm "isolated" (not a
    genuine multi-cycle collapse). The FIRST/LAST cycle structurally
    have only one neighbor - checked separately below (real bug found
    and fixed here: B0053's own known artifact IS at its last cycle,
    n=55 of 55 total, and was MISSED entirely by an interior-only
    version of this detector, caught only by explicitly checking this
    edge case rather than assuming interior logic covers everything)."""
    n = len(soh)
    flags = []
    for i in range(1, n - 1):  # interior cycles: need both neighbors to confirm isolation
        lo, hi = max(0, i - window), min(n, i + window + 1)
        surrounding = np.concatenate([soh[lo:i], soh[i + 1:hi]])
        if len(surrounding) < 3:
            continue
        local_median = np.median(surrounding)
        if local_median <= 0:
            continue
        is_drop = soh[i] < (1 - drop_frac) * local_median
        if not is_drop:
            continue
        left_ok = abs(soh[i - 1] - local_median) < neighbor_tol * local_median
        right_ok = abs(soh[i + 1] - local_median) < neighbor_tol * local_median
        if left_ok and right_ok:
            flags.append({"cycle_idx": int(cyc[i]), "soh_value": float(soh[i]),
                          "local_median": float(local_median),
                          "neighbor_before": float(soh[i - 1]), "neighbor_after": float(soh[i + 1]),
                          "position": "interior"})

    # endpoint cases: only one neighbor exists, so only that one is checked
    if n >= window + 1:
        # last cycle
        i = n - 1
        lo = max(0, i - window)
        surrounding = soh[lo:i]
        if len(surrounding) >= 3:
            local_median = np.median(surrounding)
            if local_median > 0 and soh[i] < (1 - drop_frac) * local_median:
                left_ok = abs(soh[i - 1] - local_median) < neighbor_tol * local_median
                if left_ok:
                    flags.append({"cycle_idx": int(cyc[i]), "soh_value": float(soh[i]),
                                  "local_median": float(local_median),
                                  "neighbor_before": float(soh[i - 1]), "neighbor_after": None,
                                  "position": "last_cycle (no after-neighbor to check)"})
        # first cycle
        i = 0
        hi = min(n, i + window + 1)
        surrounding = soh[i + 1:hi]
        if len(surrounding) >= 3:
            local_median = np.median(surrounding)
            if local_median > 0 and soh[i] < (1 - drop_frac) * local_median:
                right_ok = abs(soh[i + 1] - local_median) < neighbor_tol * local_median
                if right_ok:
                    flags.append({"cycle_idx": int(cyc[i]), "soh_value": float(soh[i]),
                                  "local_median": float(local_median),
                                  "neighbor_before": None, "neighbor_after": float(soh[i + 1]),
                                  "position": "first_cycle (no before-neighbor to check)"})
    return flags


def tune_and_verify():
    print("[artifact-sweep] === TUNING: verifying the detector catches all 3 known cases first ===")
    hi_exp = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    all_ok = True
    for (ds, bid), known_cycle in KNOWN_CASES.items():
        sub = hi_exp[hi_exp.battery_id == bid].sort_values("cycle_idx")
        cyc, soh = sub["cycle_idx"].to_numpy(), sub["SOH"].to_numpy()
        flags = detect_isolated_drops(cyc, soh)
        hit = any(f["cycle_idx"] == known_cycle for f in flags)
        print(f"[artifact-sweep] {ds}/{bid} cycle {known_cycle}: {'CAUGHT' if hit else 'MISSED'} "
              f"(all flags for this battery: {[f['cycle_idx'] for f in flags]})")
        all_ok = all_ok and hit
    print(f"[artifact-sweep] detector catches all 3 known cases: {all_ok}")
    if not all_ok:
        print("[artifact-sweep] WARNING: detector does not catch all known cases - "
              "thresholds need adjustment before trusting the full sweep below.")
    return all_ok


def main():
    ok = tune_and_verify()
    if not ok:
        print("[artifact-sweep] STOPPING before full sweep - fix detector first.")
        return

    for pool_name, path in [("EXPANDED (204-battery)", "hi_table_expanded.parquet"),
                             ("ORIGINAL (32-battery)", "hi_table.parquet")]:
        print(f"\n[artifact-sweep] === FULL SWEEP: {pool_name} pool ({path}) ===")
        hi = pd.read_parquet(PROC_DIR / path)
        all_flags = []
        for (ds, bid), g in hi.groupby(["dataset", "battery_id"]):
            g = g.sort_values("cycle_idx")
            cyc, soh = g["cycle_idx"].to_numpy(), g["SOH"].to_numpy()
            flags = detect_isolated_drops(cyc, soh)
            for f in flags:
                all_flags.append({"dataset": ds, "battery_id": bid, **f})

        flags_df = pd.DataFrame(all_flags)
        print(f"[artifact-sweep] {len(flags_df)} isolated-drop cycles flagged across "
              f"{hi.groupby(['dataset','battery_id']).ngroups} batteries")
        if len(flags_df):
            print(flags_df.to_string(index=False))
            n_new_batteries = flags_df["battery_id"].nunique()
            known_bids = {b for _, b in KNOWN_CASES}
            new_bids = set(flags_df["battery_id"].unique()) - known_bids
            print(f"\n[artifact-sweep] batteries flagged: {n_new_batteries} total, "
                  f"{len(new_bids)} NOT among the 3 already-known cases: {sorted(new_bids)}")

        out_path = OUT_DIR / f"artifact_sweep_{'expanded' if 'expanded' in path else 'original'}.csv"
        flags_df.to_csv(out_path, index=False)
        print(f"[artifact-sweep] saved {out_path}")

    print("\n[artifact-sweep] DONE")


if __name__ == "__main__":
    main()
