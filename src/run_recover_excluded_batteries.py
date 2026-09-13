"""
Stage 2, Item 2.1: recover the 14 batteries excluded in session 33
(Dataset Expansion) for degenerate SOH baselines up to 2,177%.

ROOT-CAUSE FINDING, made by inspecting every one of the 14 batteries'
RAW capacity traces directly before writing any correction code (not
assumed from the task's framing): session 33's own writeup documented
ONE clean example (NASA/B0041: a genuine low-rate CHARACTERIZATION
PHASE for its first 41 of 66 cycles, then an abrupt sustained jump to
its real aging trend) and implied the other 13 shared this same
pattern. Direct inspection shows this is NOT true for most of them -
two structurally DIFFERENT pathologies are present, and forcing all 14
through a single "drop the leading block, rebaseline" procedure would
be wrong for most of them:

  GROUP 1 - genuine characterization phase (matches B0041's documented
  pattern: a sustained LOW-capacity block at the very start, then a
  single clean transition to normal aging): B0038, B0039, B0040, B0041
  (4 of 14). Fix: detect the transition cycle, drop everything before
  it, re-baseline SOH to the first post-transition cycle (per this
  item's own instruction).

  GROUP 2 - an ISOLATED single-cycle artifact (a spurious spike OR
  drop at ONE specific cycle, both immediate neighbors completely
  normal, baseline/cycle-1-3 unaffected) - the EXACT SAME signature
  class the prior session's project-wide sweep already found in
  B0053/B0044/B0045/etc., just not previously connected to these 14
  exclusions: B0033, B0034, B0036, B0049, B0051, MIT/b1c18, MIT/b2c44,
  MIT/b1c0, MIT/b2c12 (9 of 14). Fix: remove ONLY the flagged cycle(s) -
  no baseline change needed, since in every one of these 9 cases the
  spike/drop occurs well after cycle 3, so the existing median(first-3)
  baseline was never actually corrupted to begin with.

  GROUP 3 - NOT cleanly recoverable: NASA/B0050 (1 of 14). Its 20-cycle
  trace shows 8+ scattered anomalous readings (near-zero drops AND
  spikes interleaved throughout, not confined to 1-2 isolated cycles) -
  reported honestly as a different, more pervasive data-quality problem
  this correction method does not fix, not forced into either group
  above.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from expanded_pool_exclusions import EXCLUDED_NASA_BATTERIES, EXCLUDED_MIT_BATTERIES

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

CHAR_PHASE_BATTERIES = {"B0038", "B0039", "B0040", "B0041"}
ISOLATED_ARTIFACT_BATTERIES = {
    "B0033", "B0034", "B0036", "B0049", "B0051",
    "b1c18", "b2c44", "b1c0", "b2c12",
}
UNRECOVERABLE_BATTERIES = {"B0050"}

DROP_FRAC = 0.80     # isolated-artifact detector: flag if a cycle's value is <20% OR >180%... see spike/drop logic below
SPIKE_FRAC = 1.30    # flag a spike if a cycle exceeds 130% of local median
NEIGHBOR_TOL = 0.30
WINDOW = 5

# Follow-up verification session (post-Stage-2 closeout): B0036 landed at
# EXACTLY 110.0% max SOH under the global SPIKE_FRAC=1.30 default - a
# residual SECOND, milder isolated spike (cycle 45, ~13.2% above its
# local median) sits just below that threshold. Verified directly before
# accepting: relaxing SPIKE_FRAC to 1.12 catches it and brings B0036 to
# exactly 100.0% max SOH, WITH a false-positive check against 15 other
# batteries (4 normal NASA, 11 normal MIT, 7 of 8 other Group-2
# batteries) showing zero new spurious flags - EXCEPT B0033/B0034 (the
# genuine gradual-early-life-ramp batteries), which each pick up several
# new, unjustified flags at that relaxed threshold. So SPIKE_FRAC=1.12
# is NOT safe as a new global default - it is applied here ONLY to
# B0036 specifically, as a documented, individually-verified override,
# exactly as reported in DEVELOPMENT_LOG.md's closeout entry.
PER_BATTERY_SPIKE_FRAC_OVERRIDE = {"B0036": 1.12}


def get_cycles(ds, bid, mit_full):
    if ds == "NASA":
        return list(iterate_nasa_cycles(bid))
    entry = next(e for e in mit_full if e["global_id"] == bid)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def detect_isolated_artifacts(caps: np.ndarray, spike_frac: float = None):
    """Bidirectional extension of the prior session's isolated-drop
    detector: flags a cycle if it is EITHER a large isolated DROP
    (<20% of local median) OR an isolated SPIKE (>spike_frac of local
    median, default SPIKE_FRAC=1.30), with both immediate neighbors
    within tolerance of the local median (i.e. genuinely isolated, not
    a real trend). `spike_frac` is overridable per-call - see
    PER_BATTERY_SPIKE_FRAC_OVERRIDE above for the one documented,
    individually-verified case (B0036) this is used for."""
    spike_frac = SPIKE_FRAC if spike_frac is None else spike_frac
    n = len(caps)
    flagged = []
    for i in range(1, n - 1):
        lo, hi = max(0, i - WINDOW), min(n, i + WINDOW + 1)
        surrounding = np.concatenate([caps[lo:i], caps[i + 1:hi]])
        if len(surrounding) < 3:
            continue
        local_median = np.median(surrounding)
        if local_median <= 0:
            continue
        is_drop = caps[i] < (1 - DROP_FRAC) * local_median
        is_spike = caps[i] > spike_frac * local_median
        if not (is_drop or is_spike):
            continue
        left_ok = abs(caps[i - 1] - local_median) < NEIGHBOR_TOL * local_median
        right_ok = abs(caps[i + 1] - local_median) < NEIGHBOR_TOL * local_median
        if left_ok and right_ok:
            flagged.append(i)
    # endpoint checks (first AND last cycle - matching the prior session's
    # project-wide sweep script's completeness fix, applied here too)
    if n >= WINDOW + 1:
        i = n - 1
        surrounding = caps[max(0, i - WINDOW):i]
        if len(surrounding) >= 3:
            local_median = np.median(surrounding)
            if local_median > 0:
                is_drop = caps[i] < (1 - DROP_FRAC) * local_median
                is_spike = caps[i] > spike_frac * local_median
                if (is_drop or is_spike) and abs(caps[i - 1] - local_median) < NEIGHBOR_TOL * local_median:
                    flagged.append(i)
        i = 0
        surrounding = caps[1:min(n, WINDOW + 1)]
        if len(surrounding) >= 3:
            local_median = np.median(surrounding)
            if local_median > 0:
                is_drop = caps[i] < (1 - DROP_FRAC) * local_median
                is_spike = caps[i] > spike_frac * local_median
                if (is_drop or is_spike) and abs(caps[i + 1] - local_median) < NEIGHBOR_TOL * local_median:
                    flagged.append(i)
    return sorted(set(flagged))


def detect_characterization_transition(caps: np.ndarray, jump_ratio=1.3, local_window=5):
    """Finds the first SUSTAINED local level-shift: a candidate
    transition point t where the local window right after t sits at
    least `jump_ratio`x above the local window right before t, held
    for at least local_window more cycles (not a single-cycle blip).

    ROOT-CAUSED FIX #1 (not the first version of this function): a
    fixed "50% of the whole trace's 75th percentile" threshold
    correctly found B0041's transition (a ~95% capacity drop during
    characterization) but MISSED B0038's (a smaller, ~40% relative
    drop, 1.0-1.1Ah characterization vs. 1.78Ah real capacity) -
    confirmed directly by inspecting B0038's raw trace before rewriting
    this function. A LOCAL before/after comparison (not one global
    threshold) catches both magnitudes of relative jump.

    ROOT-CAUSED FIX #2 (found testing fix #1 against ALL 4 candidate
    batteries, not just the one it was written for): a pure local
    ratio check latched onto NOISE WITHIN B0039's own characterization
    phase itself (its early cycles are highly erratic, 0.16-0.48Ah, not
    smoothly flat like B0038/B0040/B0041's) - the first "1.3x local
    jump" it found was between two noisy characterization-phase points,
    not the real transition to B0039's ~1.77Ah stable range. Added a
    second, independent condition: the after-window's median must ALSO
    reach at least 60% of the whole trace's 75th percentile (i.e.
    genuinely approaching the real stable level, not just locally
    higher than an even-noisier neighbor) - re-verified against all 4
    candidates below before trusting this version."""
    n = len(caps)
    global_ref = np.percentile(caps, 75)
    for t in range(1, n - local_window):
        before = caps[max(0, t - local_window):t]
        after = caps[t:t + local_window]
        if len(before) < 2 or len(after) < local_window:
            continue
        before_med, after_med = np.median(before), np.median(after)
        if before_med <= 0:
            continue
        local_jump_ok = after_med >= jump_ratio * before_med and after.min() >= jump_ratio * before_med * 0.85
        approaches_real_level = after_med >= 0.6 * global_ref
        if local_jump_ok and approaches_real_level:
            return t
    return 0  # no sustained transition found


def main():
    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full = json.load(f)

    all_targets = [("NASA", b) for b in sorted(EXCLUDED_NASA_BATTERIES)] + \
                  [("MIT", b) for b in sorted(EXCLUDED_MIT_BATTERIES)]
    print(f"[recover] {len(all_targets)} excluded batteries to process")

    results = []
    recovered_data = {}  # (ds, bid) -> list of (cycle_idx, discharge_capacity, new_soh)

    for ds, bid in all_targets:
        cycles = get_cycles(ds, bid, mit_full)
        caps = np.array([c["discharge_capacity"] for c in cycles])
        idxs = np.array([c["cycle_idx"] for c in cycles])
        n = len(caps)

        if bid in UNRECOVERABLE_BATTERIES:
            n_anom = sum(1 for i in detect_isolated_artifacts(caps) if True)
            print(f"\n[recover] {ds}/{bid}: GROUP 3 (not cleanly recoverable) - "
                  f"n={n} cycles, isolated-artifact detector alone flags "
                  f"{len(detect_isolated_artifacts(caps))} of them (pervasive, not confined to 1-2 cycles)")
            results.append({"dataset": ds, "battery_id": bid, "group": "unrecoverable",
                            "n_cycles_original": n, "n_cycles_kept": 0, "recovered": False,
                            "max_soh_after": None, "note": "too many scattered anomalies for isolated-artifact removal"})
            continue

        if bid in CHAR_PHASE_BATTERIES:
            trans = detect_characterization_transition(caps)
            kept_idx = np.arange(trans, n)
            baseline = caps[trans]  # first POST-characterization cycle, per instruction
            new_soh = caps[kept_idx] / baseline * 100.0
            max_soh = float(new_soh.max())
            print(f"\n[recover] {ds}/{bid}: GROUP 1 (characterization phase) - "
                  f"dropped {trans} leading cycles (cycles {int(idxs[0])}-{int(idxs[trans-1]) if trans>0 else 'none'}), "
                  f"new baseline=cycle {int(idxs[trans])} ({baseline:.4f}Ah), "
                  f"{n - trans} cycles kept, new max SOH={max_soh:.1f}%")
            recovered = max_soh <= 110.0
            recovered_data[(ds, bid)] = list(zip(idxs[kept_idx].tolist(), caps[kept_idx].tolist(), new_soh.tolist()))
            results.append({"dataset": ds, "battery_id": bid, "group": "characterization_phase",
                            "n_cycles_original": n, "n_dropped": int(trans), "n_cycles_kept": int(n - trans),
                            "recovered": recovered, "max_soh_after": max_soh,
                            "note": f"dropped cycles up to idx{trans-1}, rebaselined to cycle {int(idxs[trans])}"})

        elif bid in ISOLATED_ARTIFACT_BATTERIES:
            spike_frac_used = PER_BATTERY_SPIKE_FRAC_OVERRIDE.get(bid)
            if spike_frac_used is not None:
                print(f"[recover] {ds}/{bid}: using verified per-battery override "
                      f"SPIKE_FRAC={spike_frac_used} (see PER_BATTERY_SPIKE_FRAC_OVERRIDE docstring)")
            flagged = detect_isolated_artifacts(caps, spike_frac=spike_frac_used)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[flagged] = False
            kept_idx = np.where(keep_mask)[0]
            baseline = np.median(caps[kept_idx][:3])  # first-3 baseline, UNCHANGED convention (never corrupted for this group)
            new_soh = caps[kept_idx] / baseline * 100.0
            max_soh = float(new_soh.max())
            print(f"\n[recover] {ds}/{bid}: GROUP 2 (isolated artifact) - "
                  f"removed {len(flagged)} flagged cycle(s) at index(es) {[int(idxs[f]) for f in flagged]} "
                  f"(values {[round(float(caps[f]),4) for f in flagged]}), baseline UNCHANGED "
                  f"(median first-3 kept cycles={baseline:.4f}), new max SOH={max_soh:.1f}%")
            recovered = max_soh <= 110.0
            recovered_data[(ds, bid)] = list(zip(idxs[kept_idx].tolist(), caps[kept_idx].tolist(), new_soh.tolist()))
            results.append({"dataset": ds, "battery_id": bid, "group": "isolated_artifact",
                            "n_cycles_original": n, "n_dropped": len(flagged), "n_cycles_kept": int(n - len(flagged)),
                            "recovered": recovered, "max_soh_after": max_soh,
                            "note": f"removed cycles {[int(idxs[f]) for f in flagged]}"})

    results_df = pd.DataFrame(results)
    print(f"\n[recover] === SUMMARY ===")
    print(results_df.to_string(index=False))

    n_recovered = int(results_df["recovered"].sum())
    print(f"\n[recover] {n_recovered} of {len(all_targets)} batteries successfully recovered "
          f"(new max SOH <= 110% after correction)")
    not_recovered = results_df[~results_df["recovered"].fillna(False)]
    if len(not_recovered):
        print(f"[recover] NOT recovered: {not_recovered[['dataset','battery_id','group','max_soh_after']].to_string(index=False)}")

    total_recovered_cycles = sum(len(v) for (ds, bid), v in recovered_data.items()
                                  if results_df[(results_df.battery_id == bid)]["recovered"].iloc[0])
    print(f"\n[recover] total NEW cycles made available by this recovery: {total_recovered_cycles}")
    print(f"[recover] new pool total would be: 204 (current expanded) + "
          f"{n_recovered} batteries + {total_recovered_cycles} cycles "
          f"(exact new total to be computed when this pool is actually rebuilt for training)")

    # save recovered data for future use
    # VERIFICATION-SWEEP FIX (found here, not silently left as-is): the
    # first version of this file included EVERY Group 1/2 battery's
    # correction ATTEMPT, successful or not, with no column to tell them
    # apart - a future consumer filtering this file by battery_id alone
    # would silently pull in B0033/B0034/B0049's still-corrupted data
    # (max SOH >110% even after correction) alongside the genuinely
    # recovered ones. Added an explicit `recovered` boolean column so
    # this file is safe to consume by battery_id + `recovered==True`
    # filtering, without deleting the not-recovered attempts (kept for
    # the audit trail, same reasoning as the original 14 not being
    # silently dropped from analysis).
    recovered_by_bid = dict(zip(results_df["battery_id"], results_df["recovered"]))
    out_rows = []
    for (ds, bid), rows in recovered_data.items():
        for cyc, cap, soh in rows:
            out_rows.append({"dataset": ds, "battery_id": bid, "cycle_idx": cyc,
                              "discharge_capacity": cap, "SOH_recovered": soh,
                              "recovered": bool(recovered_by_bid.get(bid, False))})
    pd.DataFrame(out_rows).to_csv(PROC_DIR / "recovered_battery_cycles.csv", index=False)
    results_df.to_csv(OUT_DIR / "battery_recovery_summary.csv", index=False)
    n_true_recovered_rows = sum(1 for r in out_rows if r["recovered"])
    print(f"\n[recover] saved data/processed/recovered_battery_cycles.csv "
          f"({len(out_rows)} rows total, {n_true_recovered_rows} with recovered=True) "
          f"and outputs/battery_recovery_summary.csv")
    print("[recover] DONE")


if __name__ == "__main__":
    main()
