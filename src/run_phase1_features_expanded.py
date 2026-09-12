"""
Dataset Expansion Phase 1 (additive companion to run_phase1_features.py,
which is left completely untouched - its outputs, hi_table.parquet and
rul_summary.csv, are NOT overwritten by this script).

Same per-cycle HI/SOH/RUL + ICA/DV/DC tensor computation as the original
Phase 1, but over the FULL available NASA+MIT pool instead of the
original 4 NASA + 28-cell MIT subset:
  - NASA: all 34 batteries now extracted (data/raw/nasa/.../*.mat),
    vs. the original 4 (B0005/6/7/18).
  - MIT: all 185 cells across the 4 MATR_batch_*.mat files
    (data_adapters.mit_cell_ids()), vs. the original 28-cell subset in
    mit_subset.json.
CALCE is intentionally left OUT of this expansion (same 3 cells as
before) - it is the held-out domain-shift target this whole project
evaluates against, never part of the train/test pool itself.

Per-battery try/except added here (NOT present in the original script) -
justified explicitly, not a silent change in standard: at 6x more
batteries than the original run, the probability of hitting a real
per-file data quirk (already found and fixed once this session - see
data_adapters.py's B0050/B0052 empty-Capacity fix) is much higher, and a
single bad battery crashing the entire ~3-4 hour feature-extraction run
would be a severe, avoidable waste. A battery that raises here is
recorded in the failure log, not silently dropped - and the original
4/28-cell run already proved every one of those batteries is clean, so
this is purely defensive for the 187 newly-added ones.

Outputs (all new files, originals untouched):
  data/processed/hi_table_expanded.parquet
  data/processed/rul_summary_expanded.csv
  data/processed/mit_full_cells.json          (all 185 MIT cell descriptors)
  data/processed/phase1_expanded_failures.csv (any battery that failed, empty if none)
  data/processed/differential_tensors/{dataset}_{battery_id}.npy
      (written into the SAME tensor directory as the original run - safe,
      since every original battery_id's tensor is byte-identical to
      before: this script recomputes them with the exact same
      build_differential_tensor call on the exact same raw cycles, so
      re-writing them is a no-op in content, not a divergence. New
      battery_ids simply add new files alongside, nothing is deleted.)
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_calce_cycles, iterate_mit_cycles, mit_cell_ids
from health_indicators import compute_health_indicators, HI_NAMES
from rul_labels import compute_eol_and_rul, soh_per_cycle
from ica_dv_dc import build_differential_tensor

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
TENSOR_DIR = PROC_DIR / "differential_tensors"
TENSOR_DIR.mkdir(parents=True, exist_ok=True)

ALL_NASA_CELLS = [
    "B0005", "B0006", "B0007", "B0018",
    "B0025", "B0026", "B0027", "B0028", "B0029", "B0030", "B0031", "B0032",
    "B0033", "B0034", "B0036", "B0038", "B0039", "B0040", "B0041", "B0042",
    "B0043", "B0044", "B0045", "B0046", "B0047", "B0048", "B0049", "B0050",
    "B0051", "B0052", "B0053", "B0054", "B0055", "B0056",
]
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]  # unchanged, held out as before


def process_battery(dataset: str, battery_id: str, cycle_iter):
    t0 = time.time()
    try:
        cycles = list(cycle_iter)
    except Exception as e:
        print(f"[phase1-exp] ERROR {dataset}/{battery_id} while reading raw cycles: "
              f"{type(e).__name__}: {e}")
        return None, None, {"dataset": dataset, "battery_id": battery_id,
                             "stage": "read_cycles", "error": f"{type(e).__name__}: {e}"}

    if len(cycles) < 5:
        print(f"[phase1-exp] SKIP {dataset}/{battery_id}: only {len(cycles)} usable cycles")
        return None, None, {"dataset": dataset, "battery_id": battery_id,
                             "stage": "too_few_cycles", "error": f"only {len(cycles)} cycles"}

    try:
        eol_cycle, censored, rul_map = compute_eol_and_rul(cycles)
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
    except Exception as e:
        print(f"[phase1-exp] ERROR {dataset}/{battery_id} during HI/tensor computation: "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None, {"dataset": dataset, "battery_id": battery_id,
                             "stage": "hi_or_tensor", "error": f"{type(e).__name__}: {e}"}

    dt = time.time() - t0
    print(f"[phase1-exp] DONE {dataset}/{battery_id}: {len(cycles)} cycles, "
          f"EOL={eol_cycle} censored={censored} tensor_shape={tensor.shape} "
          f"({dt:.1f}s)")
    return df, {"dataset": dataset, "battery_id": battery_id, "n_cycles": len(cycles),
                "eol_cycle": eol_cycle, "censored": censored}, None


def main():
    t_start = time.time()
    all_dfs = []
    all_summaries = []
    failures = []

    print(f"[phase1-exp] NASA: {len(ALL_NASA_CELLS)} batteries")
    for cid in ALL_NASA_CELLS:
        df, summ, fail = process_battery("NASA", cid, iterate_nasa_cycles(cid))
        if df is not None:
            all_dfs.append(df)
            all_summaries.append(summ)
        if fail is not None:
            failures.append(fail)

    print(f"[phase1-exp] CALCE: {len(CALCE_CELLS)} cells (unchanged, held-out target)")
    for cid in CALCE_CELLS:
        df, summ, fail = process_battery("CALCE", cid, iterate_calce_cycles(cid))
        if df is not None:
            all_dfs.append(df)
            all_summaries.append(summ)
        if fail is not None:
            failures.append(fail)

    all_mit = mit_cell_ids()  # (batch_file, cell_index, global_id) x 185
    with open(PROC_DIR / "mit_full_cells.json", "w") as f:
        json.dump([{"batch_file": bf, "cell_index": idx, "global_id": gid}
                    for bf, idx, gid in all_mit], f, indent=2)
    print(f"[phase1-exp] MIT: {len(all_mit)} cells (all batches, vs. original 28-cell subset)")

    for i, (bf, idx, gid) in enumerate(all_mit):
        df, summ, fail = process_battery("MIT", gid, iterate_mit_cycles(bf, idx))
        if df is not None:
            all_dfs.append(df)
            all_summaries.append(summ)
        if fail is not None:
            failures.append(fail)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"[phase1-exp] MIT progress: {i+1}/{len(all_mit)} "
                  f"(elapsed {elapsed:.0f}s, {elapsed/(i+1):.2f}s/cell avg)")

    hi_table = pd.concat(all_dfs, ignore_index=True)
    hi_table.to_parquet(PROC_DIR / "hi_table_expanded.parquet", index=False)
    pd.DataFrame(all_summaries).to_csv(PROC_DIR / "rul_summary_expanded.csv", index=False)
    pd.DataFrame(failures).to_csv(PROC_DIR / "phase1_expanded_failures.csv", index=False)

    dt_total = time.time() - t_start
    print(f"[phase1-exp] ALL DONE in {dt_total:.1f}s ({dt_total/60:.1f} min). "
          f"{len(all_summaries)} batteries succeeded, {len(failures)} failed. "
          f"{len(hi_table)} total cycles. Saved hi_table_expanded.parquet + "
          f"rul_summary_expanded.csv + phase1_expanded_failures.csv")


if __name__ == "__main__":
    main()
