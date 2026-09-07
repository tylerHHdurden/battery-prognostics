"""
Session 23: degradation-mode analysis via dV/dQ peak-tracking, built
directly on top of this project's EXISTING differential-voltage
computation (`ica_dv_dc.compute_ica_dv_dc`, unchanged, imported not
duplicated) - the same function that already produces the dV/dQ channel
feeding the CNN-LSTM/PiFormer/CNN-BiGRU deep models. No new model, no
retraining: this re-invokes that existing function per cycle (its
output is not persisted with the voltage grid in the saved
`differential_tensors/*.npy` files - see note below) and adds a
peak-finding/tracking layer on top.

**Scoping note, stated plainly per instruction**: this is an
APPROXIMATE, voltage-curve-based indicator inspired by established
differential-voltage-analysis (DVA) degradation-mode literature (Bloom
et al. 2005, "Differential voltage analyses of high-power lithium-ion
cells"; Dubarry, Truchot & Liaw 2012, "Synthesize battery degradation
modes via a diagnostic and prognostic model") - it is NOT a validated
loss-of-lithium-inventory (LLI) / loss-of-active-material (LAM)
decomposition. That literature's actual diagnostic power comes from
comparing a full-cell DVA/ICA curve against HALF-CELL reference curves
(isolated anode/cathode electrodes) to fit how much of each electrode's
capacity window has shifted or shrunk. **This project's datasets
(NASA/CALCE/MIT) have no half-cell reference data at all** - so what
follows is a simplified, single-signal heuristic (peak position shift
vs. peak height shift on the full-cell dV/dQ curve alone), reported as
a qualitative LEANING, not a quantified LLI%/LAM% split. Treat the
labels below as "which established signature this curve's shift most
resembles," not a lab-validated diagnosis.

**Why V_grid is recomputed rather than read from the saved tensor**:
`ica_dv_dc.build_differential_tensor` stacks [dQdV, dVdQ, dIdV] as 3
channels over a per-cycle voltage grid, but does NOT persist that
grid itself (`V_grid` is per-cycle, since `np.linspace(V_u.min(),
V_u.max(), n_bins)` depends on each cycle's own observed voltage
range) - so peak POSITION (in real Volts, not bin index 0-199, which
would conflate genuine voltage shift with each cycle's own window
rescaling) requires the actual V_grid, recomputed here via the exact
same unmodified `compute_ica_dv_dc` call Phase 1 already used, not a
new computation method.
"""

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
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# "a few representative test batteries across NASA/MIT" (per instruction,
# not all 6 test batteries): NASA's only test battery, plus 2 MIT test
# batteries spanning short-vs-long life (b3c0 also revisits session 16's
# knee-point analysis, which flagged an early-life curve anomaly there -
# a useful cross-check for this session's peaks too).
BATTERIES = [("NASA", "B0018"), ("MIT", "b3c0"), ("MIT", "b1c4")]

MAX_SEARCH_WINDOW_V = 0.15  # how far (in Volts) the tracked peak is allowed to drift cycle-to-cycle before it's considered "lost", not silently re-anchored to an unrelated peak
LLI_V_FRAC_THRESHOLD = 0.05   # >5% of the discharge voltage window = "substantial" position shift
LAM_H_FRAC_THRESHOLD = 0.30   # >30% relative height loss = "substantial" amplitude loss


def load_cycles(dataset: str, battery_id: str) -> list[dict]:
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    with open(PROC_DIR / "mit_subset.json") as f:
        import json
        mit_subset = json.load(f)
    entry = next(e for e in mit_subset if e["global_id"] == battery_id)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def find_reference_peak(V_grid: np.ndarray, dvdq_abs: np.ndarray):
    """Most prominent peak in the FIRST usable cycle - the anchor every
    later cycle's tracked peak is compared against."""
    peaks, props = find_peaks(dvdq_abs, prominence=0.1 * dvdq_abs.max())
    if len(peaks) == 0:
        return None, None
    best = np.argmax(props["prominences"])
    return V_grid[peaks[best]], props["prominences"][best]


def track_peak(V_grid: np.ndarray, dvdq_abs: np.ndarray, last_v: float):
    """Finds the peak in THIS cycle's curve closest to last_v (within
    MAX_SEARCH_WINDOW_V) - standard peak-tracking practice (follow the
    same physical feature's drift, don't re-pick 'whatever is biggest'
    each cycle, which risks jumping between unrelated peaks as their
    relative heights cross over during fade).

    Height is reported as PEAK PROMINENCE (how far the peak rises above
    its surrounding local baseline), not the raw curve value at the
    peak - switched after inspecting the first version of this script's
    output: raw dV/dQ values swung by 50-60x between cycle-to-cycle
    reacquisitions of a position-stable peak (e.g. one MIT battery's
    tracked peak: 367502 -> 6318 -> 9119 over its first 3 cycles, with
    the peak's VOLTAGE position barely moving) - dV/dQ is well known in
    the DVA literature to be far noisier/spikier than dQ/dV at
    comparable smoothing (differentiating twice amplifies noise, and a
    Savitzky-Golay pass isn't enough to fully tame near-singularities
    where a discharge curve is locally very steep), so a handful of
    raw-value spikes were dominating what should be a real degradation
    signal. Prominence is comparatively robust to this because it's a
    RELATIVE measure (height above local surrounding baseline) rather
    than an absolute curve value, and visibly stabilized the height
    track on inspection - logged here as a genuine mid-analysis
    correction, not hidden."""
    peaks, props = find_peaks(dvdq_abs, prominence=0.05 * dvdq_abs.max())
    if len(peaks) == 0:
        return None, None
    peak_vs = V_grid[peaks]
    dist = np.abs(peak_vs - last_v)
    within = dist <= MAX_SEARCH_WINDOW_V
    if not within.any():
        return None, None  # peak drifted out of the search window or vanished - reported as lost, not guessed
    candidates = np.where(within)[0]
    best = candidates[np.argmin(np.abs(peak_vs[candidates] - last_v))]
    return peak_vs[best], props["prominences"][best]


def analyze_battery(dataset: str, battery_id: str, cycles: list[dict]) -> pd.DataFrame:
    rows = []
    last_v = None
    for c in cycles:
        r = compute_ica_dv_dc(c)
        if r is None:
            continue
        dvdq_abs = np.abs(r["dVdQ"])
        if last_v is None:
            v, h = find_reference_peak(r["V_grid"], dvdq_abs)
        else:
            v, h = track_peak(r["V_grid"], dvdq_abs, last_v)
        if v is not None:
            last_v = v
        rows.append({"dataset": dataset, "battery_id": battery_id,
                      "cycle_idx": c["cycle_idx"], "peak_V": v, "peak_H": h})
    return pd.DataFrame(rows)


REF_WINDOW = 5  # cycles averaged for the start/end height baseline


def classify_degradation_mode(peak_df: pd.DataFrame, V_window: float) -> dict:
    valid = peak_df.dropna(subset=["peak_V", "peak_H"])
    if len(valid) < 2:
        return {"n_tracked": len(valid), "delta_V_frac": np.nan, "delta_H_frac": np.nan,
                "signature": "insufficient tracked cycles to classify"}
    first, last = valid.iloc[0], valid.iloc[-1]
    delta_V = last["peak_V"] - first["peak_V"]
    delta_V_frac = abs(delta_V) / V_window

    # Height baseline uses the MEDIAN of the first/last REF_WINDOW tracked
    # cycles, not a single endpoint cycle - found necessary after
    # inspecting the data: cycle 1's prominence is a reproducible
    # first-cycle OUTLIER on every battery checked here (e.g. NASA
    # B0018's cycle-1 prominence ~101618 vs. ~1000-6000 for cycles 2-11;
    # MIT b1c4's cycle-1 ~248658 vs. ~5000-8000 for cycles 2-10) - most
    # plausibly a numerical edge effect (Savitzky-Golay/np.gradient
    # boundary behavior on the very first interpolated curve, or a
    # genuine formation-cycle shape difference, per session 16's
    # unrelated finding of a formation-cycle SOH anomaly) rather than a
    # real degradation-relevant value. A single-cycle endpoint made
    # delta_H_frac almost entirely an artifact of this one anomalous
    # cycle rather than a genuine start-of-life baseline - logged here
    # as a mid-analysis correction, not hidden.
    first_h = valid["peak_H"].iloc[:REF_WINDOW].median()
    last_h = valid["peak_H"].iloc[-REF_WINDOW:].median()
    delta_H_frac = (last_h - first_h) / first_h

    shifted = delta_V_frac > LLI_V_FRAC_THRESHOLD
    shrunk = abs(delta_H_frac) > LAM_H_FRAC_THRESHOLD and delta_H_frac < 0
    if shifted and shrunk:
        signature = "mixed LLI+LAM-leaning signature"
    elif shifted:
        signature = "LLI-leaning signature (peak position shifted, height relatively preserved)"
    elif shrunk:
        signature = "LAM-leaning signature (peak height collapsed, position relatively stable)"
    else:
        signature = "minimal peak-shape change over the observed test-set window"

    return {"n_tracked": len(valid), "n_lost": len(peak_df) - len(valid),
            "delta_V_volts": float(delta_V), "delta_V_frac_of_window": float(delta_V_frac),
            "delta_H_frac": float(delta_H_frac), "signature": signature}


def main():
    soh_df = pd.read_csv(PRED_DIR / "ensemble_fusion_test_preds.csv")

    summary_rows = []
    all_peak_dfs = []
    for dataset, battery_id in BATTERIES:
        print(f"\n[degmode] === {dataset}/{battery_id} ===")
        cycles = load_cycles(dataset, battery_id)
        # V_window from the first usable cycle's own discharge range -
        # the natural reference scale for "how much of this battery's
        # OWN discharge voltage window did the peak move across"
        first_r = None
        for c in cycles:
            first_r = compute_ica_dv_dc(c)
            if first_r is not None:
                break
        V_window = float(first_r["V_grid"].max() - first_r["V_grid"].min()) if first_r is not None else np.nan

        peak_df = analyze_battery(dataset, battery_id, cycles)
        all_peak_dfs.append(peak_df)
        result = classify_degradation_mode(peak_df, V_window)
        print(f"[degmode] {dataset}/{battery_id}: tracked {result.get('n_tracked')}/{len(peak_df)} cycles "
              f"({result.get('n_lost', 0)} lost - peak drifted beyond the {MAX_SEARCH_WINDOW_V}V search window)")
        print(f"[degmode] delta_V={result.get('delta_V_volts', float('nan')):+.4f}V "
              f"({result.get('delta_V_frac_of_window', float('nan')) * 100:.1f}% of this battery's "
              f"{V_window:.3f}V discharge window), delta_H_frac={result.get('delta_H_frac', float('nan')):+.3f}")
        print(f"[degmode] SIGNATURE: {result['signature']}")

        battery_soh = soh_df[(soh_df["dataset"] == dataset) & (soh_df["battery_id"] == battery_id)]
        soh_first = float(battery_soh["SOH"].iloc[0]) if len(battery_soh) else float("nan")
        soh_last = float(battery_soh["SOH"].iloc[-1]) if len(battery_soh) else float("nan")
        pred_last = float(battery_soh["pred_Stacking_Ridge_fusion"].iloc[-1]) if len(battery_soh) else float("nan")
        print(f"[degmode] SOH (true): {soh_first:.2f}% -> {soh_last:.2f}%  "
              f"(ensemble pred at last test cycle: {pred_last:.2f}%)")

        summary_rows.append({
            "dataset": dataset, "battery_id": battery_id,
            "n_cycles": len(cycles), "V_window": V_window,
            "SOH_first": soh_first, "SOH_last": soh_last, "SOH_pred_last": pred_last,
            **result,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "degradation_mode_summary.csv", index=False)
    pd.concat(all_peak_dfs, ignore_index=True).to_csv(
        PRED_DIR / "degradation_mode_peak_tracks.csv", index=False
    )

    print("\n[degmode] === SUMMARY: degradation-mode signature alongside SOH ===")
    print(summary_df[["dataset", "battery_id", "SOH_first", "SOH_last", "SOH_pred_last",
                       "delta_V_frac_of_window", "delta_H_frac", "signature"]].to_string(index=False))
    print("\n[degmode] DONE")


if __name__ == "__main__":
    main()
