"""
Research pass Group C, item 7: reference-anchored peak-tracking,
replacing the current sequential (last-cycle-to-last-cycle) tracker.

DISCLOSED SCOPE, stated plainly rather than overclaimed: the task asks
for "half-cell OCV-model-anchored" tracking - genuine half-cell
open-circuit-voltage anchoring needs real half-cell (isolated
electrode) reference curves. run_degradation_mode_analysis.py's own
docstring already discloses this project's datasets (NASA/CALCE/MIT)
have NO half-cell reference data at all - that has not changed, and
fabricating approximate literature-typical half-cell voltages for
THESE SPECIFIC cells' chemistries, without being able to verify they
actually apply, would be exactly the kind of unverified claim this
project's own standing practice avoids. What IS implemented instead is
a genuine, verifiable mechanism change addressing the SAME underlying
problem real anchoring solves: the current tracker only ever compares
each cycle to the PREVIOUS cycle's own tracked position, so once a
peak is briefly lost (weak prominence for a few cycles), the stale
last-known position is all that's left to search around, compounding
future losses. This version ALSO keeps a FIXED reference (the first
usable cycle's own peak - a real, always-available anchor, unlike a
half-cell curve this project doesn't have) and retries against that
fixed anchor with a wider window whenever the sequential search fails,
before giving up.

Re-runs the SAME 3 batteries (NASA/B0018, MIT/b3c0, MIT/b1c4) the
original evaluation used, so results are directly, exactly comparable
- not a new evaluation set.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from ica_dv_dc import compute_ica_dv_dc

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
BATTERIES = [("NASA", "B0018"), ("MIT", "b3c0"), ("MIT", "b1c4")]
MAX_SEARCH_WINDOW_V = 0.15
REF_RETRY_WINDOW_V = 0.30  # wider window when retrying against the fixed anchor - disclosed, untuned, chosen as 2x the original window


def find_reference_peak(V_grid, dvdq_abs):
    peaks, props = find_peaks(dvdq_abs, prominence=0.1 * dvdq_abs.max())
    if len(peaks) == 0:
        return None, None
    best = np.argmax(props["prominences"])
    return V_grid[peaks[best]], props["prominences"][best]


def find_peak_near(V_grid, dvdq_abs, target_v, window):
    peaks, props = find_peaks(dvdq_abs, prominence=0.05 * dvdq_abs.max())
    if len(peaks) == 0:
        return None, None
    peak_vs = V_grid[peaks]
    dist = np.abs(peak_vs - target_v)
    within = dist <= window
    if not within.any():
        return None, None
    candidates = np.where(within)[0]
    best = candidates[np.argmin(np.abs(peak_vs[candidates] - target_v))]
    return peak_vs[best], props["prominences"][best]


def track_peak_anchored(V_grid, dvdq_abs, last_v, ref_v):
    """Sequential search first (matches the original behavior exactly
    when tracking isn't lost); on failure, retries against the FIXED
    reference peak with a wider window before giving up - the one
    mechanism change."""
    v, h = find_peak_near(V_grid, dvdq_abs, last_v, MAX_SEARCH_WINDOW_V)
    if v is not None:
        return v, h, "sequential"
    if ref_v is not None:
        v, h = find_peak_near(V_grid, dvdq_abs, ref_v, REF_RETRY_WINDOW_V)
        if v is not None:
            return v, h, "reference-anchor-recovery"
    return None, None, "lost"


def analyze_battery(dataset, battery_id, cycles):
    rows = []
    last_v, ref_v = None, None
    for c in cycles:
        r = compute_ica_dv_dc(c)
        if r is None:
            continue
        dvdq_abs = np.abs(r["dVdQ"])
        if last_v is None:
            v, h = find_reference_peak(r["V_grid"], dvdq_abs)
            ref_v = v
            source = "initial" if v is not None else "lost"
        else:
            v, h, source = track_peak_anchored(r["V_grid"], dvdq_abs, last_v, ref_v)
        if v is not None:
            last_v = v
        rows.append({"dataset": dataset, "battery_id": battery_id, "cycle_idx": c["cycle_idx"],
                     "peak_V": v, "peak_H": h, "source": source})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("=== Research pass Group C, item 7: reference-anchored peak tracking ===")
    results = []
    import json
    for dataset, bid in BATTERIES:
        if dataset == "NASA":
            cycles = list(iterate_nasa_cycles(bid))
        else:
            mit_subset = json.loads((ROOT / "data" / "processed" / "mit_subset.json").read_text())
            entry = next(e for e in mit_subset if e["global_id"] == bid)
            cycles = list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))

        peak_df = analyze_battery(dataset, bid, cycles)
        n_tracked = peak_df["peak_V"].notna().sum()
        n_lost = peak_df["peak_V"].isna().sum()
        n_recovered = (peak_df["source"] == "reference-anchor-recovery").sum()
        print(f"[peak-v2] {dataset}/{bid}: {n_tracked}/{len(peak_df)} tracked, {n_lost} lost "
              f"({n_recovered} recovered via reference-anchor retry that would otherwise be lost)")
        results.append({"dataset": dataset, "battery_id": bid, "n_cycles": len(peak_df),
                        "n_tracked": int(n_tracked), "n_lost": int(n_lost), "n_recovered_via_anchor": int(n_recovered)})
        peak_df.to_csv(OUT_DIR / f"researchpass_groupC7_peak_tracks_{dataset}_{bid}.csv", index=False)

    results_df = pd.DataFrame(results)
    baseline = pd.DataFrame([
        {"dataset": "NASA", "battery_id": "B0018", "n_cycles": 132, "n_tracked_ORIGINAL": 131, "n_lost_ORIGINAL": 1},
        {"dataset": "MIT", "battery_id": "b3c0", "n_cycles": 1007, "n_tracked_ORIGINAL": 930, "n_lost_ORIGINAL": 77},
        {"dataset": "MIT", "battery_id": "b1c4", "n_cycles": 1225, "n_tracked_ORIGINAL": 548, "n_lost_ORIGINAL": 677},
    ])
    merged = results_df.merge(baseline, on=["dataset", "battery_id", "n_cycles"])
    merged["improvement_in_lost_cycles"] = merged["n_lost_ORIGINAL"] - merged["n_lost"]
    merged.to_csv(OUT_DIR / "researchpass_groupC7_peak_tracking_v2.csv", index=False)
    print("\n=== SUMMARY (n_lost: lower is better) ===")
    print(merged[["dataset", "battery_id", "n_cycles", "n_lost_ORIGINAL", "n_lost", "improvement_in_lost_cycles"]].to_string(index=False))
    for _, row in merged.iterrows():
        verdict = "WIN (fewer lost cycles)" if row["improvement_in_lost_cycles"] > 0 else (
            "TIE" if row["improvement_in_lost_cycles"] == 0 else "REGRESSION (more lost cycles)")
        print(f"[peak-v2] {row['dataset']}/{row['battery_id']}: {verdict}")
    print(f"\n[peak-v2] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
