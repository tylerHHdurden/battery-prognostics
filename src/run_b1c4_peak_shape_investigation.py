"""
Follow-up Part A.2: does b1c4's peak-tracking coverage collapse (44.7%,
session 23) come from a genuinely flatter dV/dQ curve (session 23's
unconfirmed hypothesis: MIT fast-charge cells may have less-pronounced
phase-transition features), or from something else (voltage range,
sampling density)?

Directly computes dV/dQ peak prominence (session 23's own method:
find_peaks(prominence=0.05*max), same threshold track_peak() uses) for
b1c4 vs. 3 comparison batteries with much higher tracking coverage:
MIT/b3c0 (92.3%), MIT/b3c35, MIT/b4c38 (both from the same original
32-battery test set, never itself coverage-checked, included as
additional MIT reference points), and NASA/B0018 (99.2%).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from ica_dv_dc import compute_ica_dv_dc

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

N_CYCLES_TO_CHECK = 10  # first 10 valid cycles per battery, for a stable summary rather than 1 cycle's noise


def load_cycles(dataset, gid, mit_full):
    if dataset == "NASA":
        return list(iterate_nasa_cycles(gid))
    entry = next(e for e in mit_full if e["global_id"] == gid)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def summarize_battery(dataset, gid, cycles):
    rows = []
    n_checked = 0
    for c in cycles:
        r = compute_ica_dv_dc(c)
        if r is None:
            continue
        dvdq_abs = np.abs(r["dVdQ"])
        v_range = float(r["V_grid"].max() - r["V_grid"].min())
        peaks, props = find_peaks(dvdq_abs, prominence=0.05 * dvdq_abs.max())
        n_peaks = len(peaks)
        max_prom = float(props["prominences"].max()) if n_peaks else 0.0
        # relative prominence: how much the dominant peak rises above the
        # curve's own typical (median) level - a scale-free "sharpness" measure
        rel_prom = max_prom / (np.median(dvdq_abs) + 1e-9)
        rows.append({"dataset": dataset, "battery_id": gid, "cycle_idx": c["cycle_idx"],
                      "v_range": v_range, "n_samples": len(c["discharge"]["V"]),
                      "n_peaks_found": n_peaks, "max_prominence": max_prom,
                      "relative_prominence": rel_prom, "curve_median": float(np.median(dvdq_abs))})
        n_checked += 1
        if n_checked >= N_CYCLES_TO_CHECK:
            break
    return pd.DataFrame(rows)


def main():
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)

    targets = [("NASA", "B0018"), ("MIT", "b1c4"), ("MIT", "b3c0"), ("MIT", "b3c35"), ("MIT", "b4c38")]
    all_df = []
    for dataset, gid in targets:
        cycles = load_cycles(dataset, gid, mit_subset)
        df = summarize_battery(dataset, gid, cycles)
        all_df.append(df)
        print(f"\n[b1c4-shape] === {dataset}/{gid} (n cycles checked={len(df)}) ===")
        print(f"[b1c4-shape] mean n_peaks_found={df.n_peaks_found.mean():.2f}, "
              f"mean max_prominence={df.max_prominence.mean():.4f}, "
              f"mean relative_prominence={df.relative_prominence.mean():.2f}, "
              f"mean V_range={df.v_range.mean():.4f}V, mean n_samples/cycle={df.n_samples.mean():.0f}")

    full = pd.concat(all_df, ignore_index=True)
    summary = full.groupby(["dataset", "battery_id"]).agg(
        mean_n_peaks=("n_peaks_found", "mean"),
        mean_max_prominence=("max_prominence", "mean"),
        mean_relative_prominence=("relative_prominence", "mean"),
        mean_v_range=("v_range", "mean"),
        mean_n_samples=("n_samples", "mean"),
    ).reset_index()
    print("\n[b1c4-shape] === SUMMARY, all 5 batteries side by side ===")
    print(summary.to_string(index=False))

    b1c4_row = summary[summary.battery_id == "b1c4"].iloc[0]
    others = summary[summary.battery_id != "b1c4"]
    print(f"\n[b1c4-shape] b1c4's relative_prominence ({b1c4_row.mean_relative_prominence:.2f}) vs. "
          f"others' range [{others.mean_relative_prominence.min():.2f}, {others.mean_relative_prominence.max():.2f}]")
    print(f"[b1c4-shape] b1c4's V_range ({b1c4_row.mean_v_range:.4f}V) vs. "
          f"others' range [{others.mean_v_range.min():.4f}, {others.mean_v_range.max():.4f}]")
    print(f"[b1c4-shape] b1c4's n_samples/cycle ({b1c4_row.mean_n_samples:.0f}) vs. "
          f"others' range [{others.mean_n_samples.min():.0f}, {others.mean_n_samples.max():.0f}]")

    flatter = b1c4_row.mean_relative_prominence < others.mean_relative_prominence.min()
    print(f"\n[b1c4-shape] Is b1c4's peak MEANINGFULLY flatter/less prominent than every comparison "
          f"battery: {flatter}")

    full.to_csv(ROOT / "outputs" / "b1c4_peak_shape_investigation.csv", index=False)
    print("\n[b1c4-shape] saved outputs/b1c4_peak_shape_investigation.csv")
    print("[b1c4-shape] DONE")


if __name__ == "__main__":
    main()
