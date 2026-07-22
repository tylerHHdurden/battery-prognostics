"""
Load MIT/Stanford/TRI "Data-driven prediction of battery cycle life before
capacity degradation" dataset (Severson et al., Nature Energy 2019) using the
`beep` package, following the official quickstart pattern from
https://tri-amdd.github.io/beep/Python%20tutorials/1%20-%20quickstart/

Two tiers of data are provided in data/raw/mit/:

1. Severson-et-al/  - the single documented quickstart example:
     2017-05-12_6C-50per_3_6C_CH36.csv           (raw Arbin cycler export)
     2017-05-12_6C-50per_3_6C_CH36_Metadata.csv

2. MATR_batch_*.mat - four whole-batch MATLAB files covering the ENTIRE
   published dataset (batches from 2017-05-12, 2017-06-30, 2018-04-12,
   2019-01-24), each holding dozens of individual cells. These file IDs are
   not in the BEEP docs themselves; they were recovered from
   microsoft/BatteryML's download manifest
   (batteryml/preprocess/download.py), which references the same
   data.matr.io file-id API that BEEP's quickstart uses. See
   NASA/CALCE/MIT status notes for details on how these were found.

Requires: pip install beep
"""

from pathlib import Path

import h5py
import numpy as np
from beep.structure.cli import auto_load

MIT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "mit"
BATCH_FILES = [
    "MATR_batch_20170512.mat",
    "MATR_batch_20170630.mat",
    "MATR_batch_20180412.mat",
    "MATR_batch_20190124.mat",
]


def load_quickstart_example():
    """Load the single documented BEEP quickstart cell (A123 LFP, 6C fast charge)."""
    cycler_file = MIT_DIR / "Severson-et-al" / "2017-05-12_6C-50per_3_6C_CH36.csv"
    datapath = auto_load(str(cycler_file.resolve()))
    is_valid, msg = datapath.validate()
    return datapath, is_valid, msg


def describe_quickstart_example():
    datapath, is_valid, msg = load_quickstart_example()
    df = datapath.raw_data
    print("=== MIT quickstart example: 2017-05-12_6C-50per_3_6C_CH36.csv ===")
    print(f"Valid: {is_valid} {msg}")
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Cycle count: {df['cycle_index'].max()}")
    print(df.head())
    return datapath


def batch_file_status():
    """Report presence/size of the four whole-dataset batch .mat files."""
    print("\n=== MIT full-dataset batch files (v7.3/HDF5 MATLAB structs) ===")
    for b in BATCH_FILES:
        p = MIT_DIR / b
        if p.exists():
            print(f"  {b}: {p.stat().st_size / 1e9:.2f} GB")
        else:
            print(f"  {b}: NOT YET DOWNLOADED")


def describe_batch_structure(batch_file: str, n_cells_preview: int = 2) -> None:
    """
    Open one MATR batch .mat (HDF5/v7.3) and show its structure: how many
    cells it contains, and for a couple of example cells, the summary
    per-cycle arrays and per-cycle time-series fields.

    Schema reference: microsoft/BatteryML batteryml/preprocess/preprocess_MATR.py
    Top level: batch/{summary, cycle_life, policy_readable, cycles}, each an
    HDF5 object-reference array with one entry per cell.
    """
    path = MIT_DIR / batch_file
    print(f"\n=== {batch_file} ===")
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        num_cells = batch["summary"].shape[0]
        print(f"Cells in this batch: {num_cells}")
        print(f"Top-level batch fields: {list(batch.keys())}")

        for i in range(min(n_cells_preview, num_cells)):
            cycle_life = f[batch["cycle_life"][i, 0]][:]
            policy = f[batch["policy_readable"][i, 0]][:].tobytes()[::2].decode()
            summary_group = f[batch["summary"][i, 0]]
            summary_fields = list(summary_group.keys())
            # index 0 is a low-rate diagnostic cycle, not part of the aging
            # trend, so start the fade summary from cycle 1
            qd = np.hstack(summary_group["QDischarge"][0, :].tolist())[1:]

            cycles = f[batch["cycles"][i, 0]]
            n_cycles_logged = cycles["I"].shape[0]

            print(f"\n  -- Cell b_c{i} --")
            print(f"     charge_policy: {policy}")
            print(f"     cycle_life: {cycle_life.flatten()[0]:.0f}")
            print(f"     summary fields: {summary_fields}")
            print(f"     discharge capacity (QDischarge) fade: "
                  f"{qd[0]:.4f} Ah -> {qd[-1]:.4f} Ah over {len(qd)} summarized cycles")
            print(f"     per-cycle time-series fields: {list(cycles.keys())}")
            print(f"     cycles logged (raw, incl. cycle 0 diagnostic): {n_cycles_logged}")


if __name__ == "__main__":
    describe_quickstart_example()
    batch_file_status()
    for bf in BATCH_FILES:
        if (MIT_DIR / bf).exists():
            describe_batch_structure(bf)
