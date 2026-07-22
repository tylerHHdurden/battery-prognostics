"""
Normalizes NASA / CALCE / MIT raw data into a common per-cycle record so
downstream HI/BFA/RUL/ICA code doesn't need to know which dataset it's
looking at.

A "cycle record" is a dict:
    {
        "cycle_idx": int,               # position in this battery's life
        "discharge_capacity": float,    # Ah, used for RUL/SOH labeling
        "charge":    {"t": arr, "V": arr, "I": arr, "T": arr or None},
        "discharge": {"t": arr, "V": arr, "I": arr, "T": arr or None},
    }

Current sign convention (ASSUMPTION, applied uniformly): I > 0 during
charge, I < 0 during discharge. This matches both the NASA field names
(Current_charge/Current_load) and the standard Arbin/MIT convention, so no
sign-flipping is needed for any of the three datasets.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
NASA_DIR = ROOT / "data" / "raw" / "nasa" / "B0005_B0006_B0007_B0018"
CALCE_DIR = ROOT / "data" / "raw" / "calce"
MIT_DIR = ROOT / "data" / "raw" / "mit"


# --------------------------------------------------------------------------
# NASA
# --------------------------------------------------------------------------

def iterate_nasa_cycles(cell_id: str):
    mat = sio.loadmat(NASA_DIR / f"{cell_id}.mat", simplify_cells=True)
    cycles = mat[cell_id]["cycle"]

    pending_charge = None
    cycle_idx = 0
    for c in cycles:
        if c["type"] == "charge":
            d = c["data"]
            pending_charge = {
                "t": np.atleast_1d(d["Time"]).astype(float),
                "V": np.atleast_1d(d["Voltage_measured"]).astype(float),
                "I": np.atleast_1d(d["Current_measured"]).astype(float),
                "T": np.atleast_1d(d["Temperature_measured"]).astype(float),
            }
        elif c["type"] == "discharge":
            d = c["data"]
            discharge = {
                "t": np.atleast_1d(d["Time"]).astype(float),
                "V": np.atleast_1d(d["Voltage_measured"]).astype(float),
                # NASA discharge current is logged positive (load current);
                # flip sign so I<0 during discharge, matching our
                # dataset-wide convention.
                "I": -np.atleast_1d(d["Current_measured"]).astype(float),
                "T": np.atleast_1d(d["Temperature_measured"]).astype(float),
            }
            cap = float(np.atleast_1d(d["Capacity"])[0]) if "Capacity" in d else np.nan
            if pending_charge is not None and not np.isnan(cap):
                cycle_idx += 1
                yield {
                    "cycle_idx": cycle_idx,
                    "discharge_capacity": cap,
                    "charge": pending_charge,
                    "discharge": discharge,
                }
        # impedance entries are ignored here (used elsewhere if needed)


# --------------------------------------------------------------------------
# CALCE
# --------------------------------------------------------------------------

def iterate_calce_cycles(cell_id: str):
    """
    CALCE CS2 xlsx logs have no Temperature column, so charge/discharge
    'T' is always None for this dataset (documented limitation, not a bug).
    Cycle_Index resets per session file (see src/load_calce.py docstring),
    so we group by (source_file, Cycle_Index) in date order, same as the
    loader's capacity-fade fix.
    """
    from load_calce import load_cell  # local import to avoid path issues

    df = load_cell(cell_id)
    grouped = df.groupby(["source_file", "Cycle_Index"], sort=False)
    # preserve chronological order
    order = (
        df.groupby(["source_file", "Cycle_Index"], sort=False)["Date_Time"]
        .min()
        .sort_values()
        .index
    )

    cycle_idx = 0
    for key in order:
        g = grouped.get_group(key).sort_values("Test_Time(s)")
        charge_mask = g["Current(A)"] > 0
        discharge_mask = g["Current(A)"] < 0
        if charge_mask.sum() < 2 or discharge_mask.sum() < 2:
            continue

        gc = g[charge_mask]
        gd = g[discharge_mask]

        # true (non-cumulative) discharge capacity for this cycle:
        # cumulative-within-file column, so take (max - min) within this
        # cycle's own rows.
        cap = float(gd["Discharge_Capacity(Ah)"].max() - gd["Discharge_Capacity(Ah)"].min())
        if cap <= 0:
            continue

        cycle_idx += 1
        yield {
            "cycle_idx": cycle_idx,
            "discharge_capacity": cap,
            "charge": {
                "t": gc["Test_Time(s)"].to_numpy(float),
                "V": gc["Voltage(V)"].to_numpy(float),
                "I": gc["Current(A)"].to_numpy(float),
                "T": None,
            },
            "discharge": {
                "t": gd["Test_Time(s)"].to_numpy(float),
                "V": gd["Voltage(V)"].to_numpy(float),
                "I": gd["Current(A)"].to_numpy(float),
                "T": None,
            },
        }


# --------------------------------------------------------------------------
# MIT (batch .mat / HDF5)
# --------------------------------------------------------------------------

MIT_BATCH_FILES = [
    "MATR_batch_20170512.mat",
    "MATR_batch_20170630.mat",
    "MATR_batch_20180412.mat",
    "MATR_batch_20190124.mat",
]


def mit_cell_ids():
    """List (batch_file, cell_index, global_id) for every cell in every batch."""
    ids = []
    for bidx, bf in enumerate(MIT_BATCH_FILES, start=1):
        path = MIT_DIR / bf
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            n = f["batch"]["summary"].shape[0]
        for i in range(n):
            ids.append((bf, i, f"b{bidx}c{i}"))
    return ids


def iterate_mit_cycles(batch_file: str, cell_index: int, max_cycles: int | None = None):
    """
    Skips cycle 0 (low-rate diagnostic cycle, not part of the aging trend —
    same assumption as src/load_mit.py's fade summary).
    Charge/discharge split by sign of I, consistent with NASA/CALCE.
    """
    path = MIT_DIR / batch_file
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        cycles = f[batch["cycles"][cell_index, 0]]
        n_cycles = cycles["I"].shape[0]
        upper = n_cycles if max_cycles is None else min(n_cycles, max_cycles + 1)

        cycle_idx = 0
        for j in range(1, upper):  # skip cycle 0
            I = np.hstack(f[cycles["I"][j, 0]][:]).astype(float)
            V = np.hstack(f[cycles["V"][j, 0]][:]).astype(float)
            t = np.hstack(f[cycles["t"][j, 0]][:]).astype(float)
            T = np.hstack(f[cycles["T"][j, 0]][:]).astype(float)
            Qd = np.hstack(f[cycles["Qd"][j, 0]][:]).astype(float)

            charge_mask = I > 0
            discharge_mask = I < 0
            if charge_mask.sum() < 2 or discharge_mask.sum() < 2:
                continue

            cap = float(Qd.max())
            if cap <= 0:
                continue

            cycle_idx += 1
            yield {
                "cycle_idx": cycle_idx,
                "discharge_capacity": cap,
                "charge": {
                    "t": t[charge_mask], "V": V[charge_mask],
                    "I": I[charge_mask], "T": T[charge_mask],
                },
                "discharge": {
                    "t": t[discharge_mask], "V": V[discharge_mask],
                    "I": I[discharge_mask], "T": T[discharge_mask],
                },
            }
