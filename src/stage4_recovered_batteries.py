"""
Stage 4: shared, single-source-of-truth loader for Stage 2.1's
recovered batteries (10 as of the current recovered_battery_cycles.csv -
6 NASA + 4 MIT; see run_stage4_feature_regen.py's own docstring for the
"9 vs 10" discrepancy this project's prior framing had, found and
reconciled there), used identically by BOTH the HI-table pipeline
(run_stage4_feature_regen.py) and the sequence-tensor pipeline
(sequence-model/encoder retraining) - avoids two divergent
implementations of the same correction logic.

`recovered_battery_cycles.csv` (Stage 2.1) gives, per (battery_id,
cycle_idx) with recovered==True: which ORIGINAL cycle_idx values to
KEEP and a corrected SOH_recovered value (baseline-corrected per the
battery's own recovery group - see run_recover_excluded_batteries.py's
module docstring for the exact group-1/group-2 correction methodology,
reused here rather than re-derived). This module does NOT recompute
SOH from scratch (that would risk reproducing the Group-1 baseline
mismatch bug - see below) - it uses the recovery script's own
already-verified SOH_recovered values directly as ground truth.

EOL/RUL, NOT provided by the recovery file, computed here as:
  - MIT b1c*/b3c* cells (Severson-aware override applies): EOL from
    Severson's own published cycle_life (identical mechanism/offset as
    rul_labels.compute_eol_and_rul_severson_aware) - independent of the
    local capacity correction, since Severson's number doesn't derive
    from this project's own capacity trace at all.
  - everything else: first cycle (in kept, original cycle_idx order)
    where SOH_recovered <= 80.0, else censored - the same 80%-of-
    initial-capacity EOL convention as rul_labels.compute_eol_and_rul,
    just applied to the corrected SOH trace instead of re-deriving a
    baseline from (possibly still-corrupted) raw capacity.

IMPORTANT, stated explicitly: NOT using soh_per_cycle/compute_eol_and_rul
directly on a naively-filtered cycle list here is a deliberate choice,
not an oversight - for the 4 GROUP-1 (characterization-phase) batteries
specifically, the correct baseline is the SINGLE first post-transition
cycle (per the recovery script), not median(first 3 kept cycles) which
soh_per_cycle would silently compute instead - a real, subtle mismatch
that reusing SOH_recovered directly avoids entirely.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from rul_labels import compute_eol_and_rul_severson_aware

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

RECOVERED_NASA = ["B0036", "B0038", "B0039", "B0040", "B0041", "B0051"]
RECOVERED_MIT = ["b1c0", "b1c18", "b2c12", "b2c44"]
ALL_RECOVERED = [("NASA", b) for b in RECOVERED_NASA] + [("MIT", b) for b in RECOVERED_MIT]

_cache = {}


def _load_recovery_df():
    if "df" not in _cache:
        df = pd.read_csv(PROC_DIR / "recovered_battery_cycles.csv")
        _cache["df"] = df[df["recovered"] == True]  # noqa: E712 - explicit, matches the CSV's own bool column
    return _cache["df"]


def _mit_full_lookup():
    if "mit_full" not in _cache:
        with open(PROC_DIR / "mit_full_cells.json") as f:
            _cache["mit_full"] = json.load(f)
    return _cache["mit_full"]


def get_recovered_battery(dataset: str, bid: str):
    """Returns (kept_cycles, soh_map, rul_map, eol_cycle, censored) for
    ONE recovered battery - kept_cycles is the RAW cycle dict list
    (data_adapters convention), filtered to only the recovery-verified
    kept cycle_idx values, in original chronological order."""
    if dataset == "NASA":
        cycles = list(iterate_nasa_cycles(bid))
    else:
        entry = next(e for e in _mit_full_lookup() if e["global_id"] == bid)
        cycles = list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))

    sub = _load_recovery_df()
    sub = sub[sub["battery_id"] == bid]
    if len(sub) == 0:
        raise ValueError(f"{dataset}/{bid}: no recovered=True rows found in recovered_battery_cycles.csv")

    kept_idx_set = set(int(i) for i in sub["cycle_idx"])
    soh_map = {int(i): float(s) for i, s in zip(sub["cycle_idx"], sub["SOH_recovered"])}
    kept_cycles = [c for c in cycles if c["cycle_idx"] in kept_idx_set]
    if not kept_cycles:
        raise ValueError(f"{dataset}/{bid}: zero cycles survived filtering - mismatch with recovery file?")

    if dataset == "MIT" and (bid.startswith("b1c") or bid.startswith("b3c")):
        eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(kept_cycles, global_id=bid)
    else:
        idxs_sorted = sorted(kept_idx_set)
        soh_trace = [soh_map[i] for i in idxs_sorted]
        below = [i for i, s in zip(idxs_sorted, soh_trace) if s <= 80.0]
        if below:
            eol_cycle, censored = below[0], False
        else:
            eol_cycle, censored = idxs_sorted[-1] + 1, True
        rul_map = {i: max(eol_cycle - i, 0) for i in idxs_sorted}

    return kept_cycles, soh_map, rul_map, eol_cycle, censored
