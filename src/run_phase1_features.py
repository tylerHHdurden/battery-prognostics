"""
Phase 1 orchestrator: for every battery in NASA (4), CALCE (3), and the
MIT subset (28), compute:
  - per-cycle HIs (16) + SOH
  - per-battery RUL labels (EOL cycle, censored flag) -> per-cycle RUL
  - per-cycle ICA/DV/DC differential arrays -> saved as one .npy tensor
    per battery

Outputs:
  data/processed/hi_table.parquet      (per-cycle HI+SOH+RUL, all batteries)
  data/processed/rul_summary.csv       (per-battery EOL/censored summary)
  data/processed/differential_tensors/{dataset}_{battery_id}.npy

Designed to run standalone (so it can be launched in the background and
tailed): every battery prints a progress line as it finishes.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_calce_cycles, iterate_mit_cycles
from health_indicators import compute_health_indicators, HI_NAMES
from rul_labels import compute_eol_and_rul, soh_per_cycle
from ica_dv_dc import build_differential_tensor

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
TENSOR_DIR = PROC_DIR / "differential_tensors"
TENSOR_DIR.mkdir(parents=True, exist_ok=True)

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]


def process_battery(dataset: str, battery_id: str, cycle_iter):
    t0 = time.time()
    cycles = list(cycle_iter)
    if len(cycles) < 5:
        print(f"[phase1] SKIP {dataset}/{battery_id}: only {len(cycles)} usable cycles")
        return None, None

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

    dt = time.time() - t0
    print(f"[phase1] DONE {dataset}/{battery_id}: {len(cycles)} cycles, "
          f"EOL={eol_cycle} censored={censored} tensor_shape={tensor.shape} "
          f"({dt:.1f}s)")
    return df, {"dataset": dataset, "battery_id": battery_id, "n_cycles": len(cycles),
                "eol_cycle": eol_cycle, "censored": censored}


def main():
    all_dfs = []
    all_summaries = []

    for cid in NASA_CELLS:
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
        df, summ = process_battery(
            "MIT", entry["global_id"],
            iterate_mit_cycles(entry["batch_file"], entry["cell_index"]),
        )
        if df is not None:
            all_dfs.append(df)
            all_summaries.append(summ)

    hi_table = pd.concat(all_dfs, ignore_index=True)
    hi_table.to_parquet(PROC_DIR / "hi_table.parquet", index=False)
    pd.DataFrame(all_summaries).to_csv(PROC_DIR / "rul_summary.csv", index=False)

    print(f"[phase1] ALL DONE. {len(all_summaries)} batteries, "
          f"{len(hi_table)} total cycles. Saved hi_table.parquet + rul_summary.csv")


if __name__ == "__main__":
    main()
