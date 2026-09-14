"""
Stage 4, Step 1: regenerate hi_table.parquet from scratch over the
CONFIRMED Stage 4 pool - original 32 batteries (4 NASA + 3 CALCE + 28
MIT) + Stage 2.1's recovered batteries.

DISCREPANCY FOUND AND FLAGGED, per standing instruction to state such
things explicitly: the task text (and most of this project's own prior
framing) says "9 recovered batteries, 2,993 cycles." Checked directly
against `recovered_battery_cycles.csv` (the single source of truth):
it currently holds **10** recovered==True batteries and **3,187**
cycles - 6 NASA (B0036/38/39/40/41/51) + 4 MIT (b1c0/b1c18/b2c12/b2c44).
The discrepancy is explained, not just noted: the LATER closeout
session ("B0036 recovery" - see DEVELOPMENT_LOG.md) individually
verified and added B0036 as a 10th recovered battery via a
per-battery-overridden spike threshold, AFTER the original "9 of 14"
Stage 2.1 count was set. `recovered_battery_cycles.csv` already
reflects this later addition (confirmed directly: B0036 appears with
recovered=True). This script reads the recovered-battery set DYNAMICALLY
from that file (not a hardcoded "9"), so it correctly includes all 10
already-verified recoveries - using a stale "9" would mean arbitrarily
excluding B0036 despite it being legitimately recovered and already
part of the canonical recovery artifact.

Mandatory per instruction: the CURRENT hi_table.parquet was confirmed
last session to still carry PRE-Severson-aware RUL labels for MIT
batch-1/3 cells (b1c20's cycle-1 RUL matched the old convention, not
the new one) - this regeneration is what actually applies the
Severson-aware default that's been live in code (but unused on disk)
since session 43.

Original 32 batteries: IDENTICAL code path to run_phase1_features.py
(same functions, same Severson-aware default, unchanged) - not
reimplemented, imported and reused directly.

9 recovered batteries: uses stage4_recovered_batteries.py's shared,
single-source-of-truth loader (filtered kept-cycle sets + corrected
SOH_recovered + freshly-computed EOL/RUL) - see that module's own
docstring for why SOH_recovered is used directly rather than
recomputed via soh_per_cycle.

Existing hi_table.parquet/rul_summary.csv/battery_split.json are
BACKED UP (not silently overwritten) before this script writes the new
pool's versions, preserving full Stage 1-3 reproducibility/audit trail.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_calce_cycles, iterate_mit_cycles
from health_indicators import compute_health_indicators, HI_NAMES
from rul_labels import compute_eol_and_rul_severson_aware, soh_per_cycle
from ica_dv_dc import build_differential_tensor
from stage4_recovered_batteries import get_recovered_battery

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
TENSOR_DIR = PROC_DIR / "differential_tensors"
TENSOR_DIR.mkdir(parents=True, exist_ok=True)

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]


def process_battery(dataset: str, battery_id: str, cycle_iter):
    """Unchanged from run_phase1_features.py - reused, not reimplemented."""
    t0 = time.time()
    cycles = list(cycle_iter)
    if len(cycles) < 5:
        print(f"[stage4-feat] SKIP {dataset}/{battery_id}: only {len(cycles)} usable cycles")
        return None, None

    eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=battery_id)
    soh_map = soh_per_cycle(cycles)

    rows = []
    for c in cycles:
        his = compute_health_indicators(c)
        row = {
            "dataset": dataset, "battery_id": battery_id,
            "cycle_idx": c["cycle_idx"],
            "discharge_capacity": c["discharge_capacity"],
            "SOH": soh_map[c["cycle_idx"]],
            "RUL": rul_map[c["cycle_idx"]],
        }
        row.update(his)
        rows.append(row)
    df = pd.DataFrame(rows)

    tensor = build_differential_tensor(cycles)
    np.save(TENSOR_DIR / f"{dataset}_{battery_id}.npy", tensor)

    dt = time.time() - t0
    print(f"[stage4-feat] DONE {dataset}/{battery_id}: {len(cycles)} cycles, "
          f"EOL={eol_cycle} censored={censored} tensor_shape={tensor.shape} ({dt:.1f}s)")
    return df, {"dataset": dataset, "battery_id": battery_id, "n_cycles": len(cycles),
                "eol_cycle": eol_cycle, "censored": censored}


def process_recovered_battery(dataset: str, battery_id: str):
    t0 = time.time()
    kept_cycles, soh_map, rul_map, eol_cycle, censored = get_recovered_battery(dataset, battery_id)

    rows = []
    for c in kept_cycles:
        his = compute_health_indicators(c)
        row = {
            "dataset": dataset, "battery_id": battery_id,
            "cycle_idx": c["cycle_idx"],
            "discharge_capacity": c["discharge_capacity"],
            "SOH": soh_map[c["cycle_idx"]],   # CORRECTED (Stage 2.1), not recomputed
            "RUL": rul_map[c["cycle_idx"]],
        }
        row.update(his)
        rows.append(row)
    df = pd.DataFrame(rows)

    tensor = build_differential_tensor(kept_cycles)
    np.save(TENSOR_DIR / f"{dataset}_{battery_id}.npy", tensor)

    dt = time.time() - t0
    print(f"[stage4-feat] DONE (RECOVERED) {dataset}/{battery_id}: {len(kept_cycles)} cycles "
          f"(of {len(kept_cycles)} kept), EOL={eol_cycle} censored={censored} "
          f"tensor_shape={tensor.shape} ({dt:.1f}s)")
    return df, {"dataset": dataset, "battery_id": battery_id, "n_cycles": len(kept_cycles),
                "eol_cycle": eol_cycle, "censored": censored}


def main():
    t_start = time.time()

    # --- backup existing artifacts before overwriting anything ---
    backup_dir = PROC_DIR / "_pre_stage4_backup"
    backup_dir.mkdir(exist_ok=True)
    for fname in ["hi_table.parquet", "rul_summary.csv", "battery_split.json"]:
        src = PROC_DIR / fname
        if src.exists():
            dst = backup_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"[stage4-feat] backed up {fname} -> _pre_stage4_backup/{fname}")
            else:
                print(f"[stage4-feat] backup already exists for {fname}, not overwriting the backup")

    all_dfs = []
    all_summaries = []

    # --- determine the recovered-battery set FIRST, so the "original"
    # loop below can skip any battery_id that also appears there - found
    # and fixed BEFORE trusting any output, not after: b2c44 is a real
    # member of the ORIGINAL 32-battery mit_subset.json (verified
    # directly) that ALSO appears in recovered_battery_cycles.csv
    # (it was independently flagged during the 204-pool expansion's
    # separate exclusion sweep and individually corrected there - see
    # run_recover_excluded_batteries.py's GROUP 2 list). Processing it
    # via BOTH paths would duplicate ~477 of its 478 cycles in hi_table
    # (confirmed by direct inspection of a first, buggy run of this
    # script - caught and fixed here rather than propagated downstream).
    # The recovered version is STRICTLY the better one (identical data
    # minus one confirmed isolated-artifact cycle), so it wins.
    recovery_df = pd.read_csv(PROC_DIR / "recovered_battery_cycles.csv")
    recovered_battery_ids = recovery_df[recovery_df["recovered"] == True][  # noqa: E712
        ["dataset", "battery_id"]].drop_duplicates()
    recovered_id_set = set(recovered_battery_ids["battery_id"])
    print(f"[stage4-feat] {len(recovered_battery_ids)} recovered batteries found in "
          f"recovered_battery_cycles.csv: {recovered_battery_ids.to_dict('records')}")

    print("\n[stage4-feat] === original 32-battery pool (unchanged processing, "
          "MINUS any battery also in the recovered set - see note above) ===")
    for cid in NASA_CELLS:
        if cid in recovered_id_set:
            print(f"[stage4-feat] SKIP original-path NASA/{cid}: also in recovered set, "
                  f"will be processed via the corrected path only")
            continue
        df, summ = process_battery("NASA", cid, iterate_nasa_cycles(cid))
        if df is not None:
            all_dfs.append(df)
            all_summaries.append(summ)

    for cid in CALCE_CELLS:
        df, summ = process_battery("CALCE", cid, iterate_calce_cycles(cid))
        if df is not None:
            all_dfs.append(df)
            all_summaries.append(summ)

    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    for entry in mit_subset:
        gid = entry["global_id"]
        if gid in recovered_id_set:
            print(f"[stage4-feat] SKIP original-path MIT/{gid}: also in recovered set, "
                  f"will be processed via the corrected path only")
            continue
        df, summ = process_battery(
            "MIT", gid,
            iterate_mit_cycles(entry["batch_file"], entry["cell_index"]),
        )
        if df is not None:
            all_dfs.append(df)
            all_summaries.append(summ)

    print(f"\n[stage4-feat] === recovered batteries (Stage 2.1, filtered+corrected) ===")

    recovered_train_ids = []
    for _, row in recovered_battery_ids.iterrows():
        ds, bid = row["dataset"], row["battery_id"]
        df, summ = process_recovered_battery(ds, bid)
        if df is not None:
            all_dfs.append(df)
            all_summaries.append(summ)
            recovered_train_ids.append(bid)

    hi_table = pd.concat(all_dfs, ignore_index=True)
    hi_table.to_parquet(PROC_DIR / "hi_table.parquet", index=False)
    pd.DataFrame(all_summaries).to_csv(PROC_DIR / "rul_summary.csv", index=False)

    print(f"\n[stage4-feat] ALL DONE. {len(all_summaries)} batteries, {len(hi_table)} total cycles "
          f"({time.time()-t_start:.1f}s). Saved hi_table.parquet + rul_summary.csv")
    print(f"[stage4-feat] pool breakdown by dataset:")
    print(hi_table.groupby("dataset")["battery_id"].nunique().to_string())

    # --- extend battery_split.json: add the 9 recovered batteries to TRAIN
    # (the existing 6-battery TEST set stays FIXED for continuity with every
    # prior Stage 1-3 in-domain number) ---
    with open(backup_dir / "battery_split.json") as f:
        old_split = json.load(f)
    new_split = {
        "train_ids": sorted(set(old_split["train_ids"]) | set(recovered_train_ids)),
        "test_ids": old_split["test_ids"],  # UNCHANGED
    }
    with open(PROC_DIR / "battery_split.json", "w") as f:
        json.dump(new_split, f, indent=2)
    print(f"\n[stage4-feat] battery_split.json extended: train {len(old_split['train_ids'])} -> "
          f"{len(new_split['train_ids'])} batteries (+{len(recovered_train_ids)} recovered), "
          f"test unchanged at {len(new_split['test_ids'])} batteries")

    # --- MANDATORY verification: Severson-aware convention now actually live on disk ---
    print(f"\n[stage4-feat] === MANDATORY VERIFICATION: Severson-aware RUL now live? ===")
    b1c20 = hi_table[(hi_table.battery_id == "b1c20") & (hi_table.cycle_idx == 1)]
    if len(b1c20):
        rul_val = float(b1c20["RUL"].iloc[0])
        print(f"[stage4-feat] b1c20 cycle-1 RUL = {rul_val} "
              f"(OLD convention would give 532.0; NEW Severson-aware convention should give 531.0)")
        is_new = abs(rul_val - 531.0) < 0.5
        is_old = abs(rul_val - 532.0) < 0.5
        print(f"[stage4-feat] matches NEW (Severson-aware) convention: {is_new}")
        print(f"[stage4-feat] matches OLD (pre-Severson) convention: {is_old}")
        if not is_new:
            print("[stage4-feat] *** WARNING: Severson-aware convention NOT confirmed live - "
                  "STOP and investigate before proceeding to Step 2 ***")
        else:
            print("[stage4-feat] CONFIRMED: Severson-aware EOL/RUL convention is now live in hi_table.parquet")
    else:
        print("[stage4-feat] *** WARNING: b1c20 cycle 1 not found in new hi_table - cannot verify ***")

    print("[stage4-feat] DONE")


if __name__ == "__main__":
    main()
