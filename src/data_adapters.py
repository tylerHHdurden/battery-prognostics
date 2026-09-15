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
OXFORD_DIR = ROOT / "data" / "raw" / "oxford"
HUST_DIR = ROOT / "data" / "raw" / "hust" / "extracted" / "our_data"
XJTU_DIR = ROOT / "data" / "raw" / "xjtu" / "extracted"
NASA_RANDOMIZED_DIR = ROOT / "data" / "raw" / "nasa_randomized" / "extracted2"


# --------------------------------------------------------------------------
# Raw-data availability checks
#
# data/raw/ is entirely gitignored (third-party research datasets - size +
# redistribution licensing concerns), so a fresh clone (e.g. Streamlit
# Community Cloud) has none of these files. These let a caller (the
# dashboard) check BEFORE trying to load, instead of hitting a raw
# FileNotFoundError/OSError deep inside loadmat/h5py/openpyxl.
# --------------------------------------------------------------------------

def nasa_data_available() -> bool:
    return NASA_DIR.is_dir() and any(NASA_DIR.glob("*.mat"))


def calce_data_available() -> bool:
    return CALCE_DIR.is_dir() and any(CALCE_DIR.glob("*.zip"))


def mit_data_available() -> bool:
    return MIT_DIR.is_dir() and any((MIT_DIR / bf).exists() for bf in MIT_BATCH_FILES)


def oxford_data_available() -> bool:
    return (OXFORD_DIR / "Oxford_Battery_Degradation_Dataset_1.mat").exists()


def hust_data_available() -> bool:
    return HUST_DIR.is_dir() and any(HUST_DIR.glob("*.pkl"))


def xjtu_data_available() -> bool:
    return XJTU_DIR.is_dir() and any(XJTU_DIR.rglob("*.mat"))


def nasa_randomized_data_available() -> bool:
    return NASA_RANDOMIZED_DIR.is_dir() and any(NASA_RANDOMIZED_DIR.rglob("RW*.mat"))


# --------------------------------------------------------------------------
# XJTU (Wang et al., Zenodo 10963339) (Stage 5.1)
#
# Source: https://zenodo.org/records/10963339 (2.44 GB zip, "Battery
# Dataset/Batch-{1..6}/{policy}_battery-{n}.mat", 55 LISHEN NCM 18650
# cells, 2000 mAh nominal, 6 charge/discharge protocols, 1 Hz). One
# extra top-level file (Temperature_Compensation_Data.mat) is not a
# battery cell and is excluded. Each cell's .mat has `data` (list of
# per-cycle dicts with combined charge+discharge current/voltage/
# capacity/temperature/time - already in our sign convention,
# verified directly: I>0 charge, I<0 discharge, capacity_Ah resets to
# 0 at each charge/discharge phase transition) and `summary` (per-
# cycle aggregates, not used here - recomputed from `data` instead for
# consistency with every other adapter in this project).
# --------------------------------------------------------------------------

def xjtu_cell_ids():
    ids = []
    for batch_dir in sorted(XJTU_DIR.glob("Battery Dataset/Batch-*")):
        for f in sorted(batch_dir.glob("*.mat")):
            ids.append(f"{batch_dir.name}/{f.stem}")
    return ids


def iterate_xjtu_cycles(cell_id: str, test_capacity_only: bool = False):
    """
    test_capacity_only=True (Stage 5 follow-on, recovering Batch-6/
    Sim_satellite rather than excluding it): XJTU's per-cycle
    `description` field distinguishes periodic full-capacity-check
    cycles (e.g. "0.5C charge and 0.2C discharge [test capacity]",
    confirmed present in EVERY batch, ~1 in 6 cycles) from the batch's
    own regular cycling protocol. For batches 1-5, the regular cycles
    are themselves consistent-depth full discharges, so both cycle
    types are usable and this flag changes nothing meaningful if set.
    For Batch-6 (Sim_satellite), the regular cycles are a genuinely
    variable, PARTIAL depth-of-discharge (simulating real satellite
    load - verified directly: cell battery-1's regular-cycle discharge
    deltas swing 1.991/0.111/0.445 Ah in its first 3 cycles alone),
    incompatible with this project's constant-depth SOH convention
    (rul_labels.soh_per_cycle) - only the "[test capacity]" checkpoints
    represent genuine, comparable full-discharge tests for that batch.
    Restricting to those checkpoints recovers physically sensible SOH
    (82-104%, smooth degradation) where using every cycle gave up to
    457% (see DEVELOPMENT_LOG.md for the root-cause and the recovery
    verification). Cycle_idx keeps the SAME sequential-usable-cycle
    convention as every other cycle_idx in this project (increments
    only for cycles that pass every filter, not a raw file position).
    """
    path = XJTU_DIR / "Battery Dataset" / f"{cell_id}.mat"
    mat = sio.loadmat(path, simplify_cells=True)
    cycles = mat["data"]

    cycle_idx = 0
    for c in cycles:
        if test_capacity_only and "test capacity" not in str(c.get("description", "")).lower():
            continue

        t = np.atleast_1d(c["relative_time_min"]).astype(float) * 60.0  # min -> s
        V = np.atleast_1d(c["voltage_V"]).astype(float)
        I = np.atleast_1d(c["current_A"]).astype(float)
        cap = np.atleast_1d(c["capacity_Ah"]).astype(float)
        T = np.atleast_1d(c["temperature_C"]).astype(float)

        charge_mask = I > 0.01
        discharge_mask = I < -0.01
        if charge_mask.sum() < 2 or discharge_mask.sum() < 2:
            continue

        dis_cap = cap[discharge_mask]
        discharge_capacity = float(dis_cap.max() - dis_cap.min())
        if discharge_capacity <= 0:
            continue

        cycle_idx += 1
        yield {
            "cycle_idx": cycle_idx,
            "discharge_capacity": discharge_capacity,
            "charge": {
                "t": t[charge_mask], "V": V[charge_mask],
                "I": I[charge_mask], "T": T[charge_mask],
            },
            "discharge": {
                "t": t[discharge_mask], "V": V[discharge_mask],
                "I": I[discharge_mask], "T": T[discharge_mask],
            },
        }


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
            # Dataset-expansion session (Phase 1): "Capacity" in d" alone
            # isn't sufficient - B0050/B0052 (2 of the 30 newly-extracted
            # NASA batteries) have discharge entries where the field is
            # PRESENT but an EMPTY array (size 0), which raised
            # `IndexError: index 0 is out of bounds for axis 0 with size 0`
            # on the un-guarded `[0]` below. Confirmed a genuine NASA data
            # quirk (4/25 and 21/25 discharge cycles respectively), not an
            # extraction artifact - the other 32/34 batteries (incl. all 4
            # originally-used B0005/6/7/18, verified to have zero
            # empty-Capacity cycles) are completely unaffected by this fix.
            cap_field = np.atleast_1d(d["Capacity"]) if "Capacity" in d else np.array([])
            cap = float(cap_field[0]) if cap_field.size else np.nan
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


# --------------------------------------------------------------------------
# Oxford Battery Degradation Dataset 1 (Stage 5.1)
#
# Source: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac
# (254 MB single .mat). 8 Kokam 740 mAh pouch cells, thermal chamber at
# 40degC, Artemis urban drive-cycle aging with a full C1 constant-current
# charge/discharge characterization test saved every ~100 real cycles
# (cyc0000, cyc0100, ...) - NOT every cycle. cycle_idx below is the actual
# aging-protocol cycle number at each checkpoint (parsed from the key), not
# an ordinal reindex - this keeps RUL/EOL cycle-count units comparable to
# every other dataset in this project, at the cost of a much coarser
# per-battery time resolution than NASA/MIT/CALCE (tens of checkpoints per
# cell life, not hundreds-to-thousands of cycles).
#
# No current channel is logged directly - only cumulative charge/discharge
# capacity q (mAh) vs. time t (MATLAB datenum, days). Current is
# reconstructed via I = dq/dt (converted to A), the same numerical-
# differentiation approach already used elsewhere in this project whenever
# a dataset logs cumulative capacity instead of raw current (e.g. CALCE's
# per-cycle capacity, MIT's Qd). Verified sign convention needs no flip:
# C1ch's q rises (I>0 already); C1dc's q is already negative (I<0 already).
# --------------------------------------------------------------------------

def oxford_cell_ids():
    return [f"Cell{i}" for i in range(1, 9)]


def iterate_oxford_cycles(cell_id: str):
    mat = sio.loadmat(OXFORD_DIR / "Oxford_Battery_Degradation_Dataset_1.mat", simplify_cells=True)
    cell = mat[cell_id]
    checkpoint_keys = sorted(
        (k for k in cell.keys() if k.startswith("cyc")),
        key=lambda k: int(k[3:]),
    )

    for key in checkpoint_keys:
        cyc_num = int(key[3:])
        checkpoint = cell[key]
        if "C1ch" not in checkpoint or "C1dc" not in checkpoint:
            continue
        ch_raw, dc_raw = checkpoint["C1ch"], checkpoint["C1dc"]

        def _build(raw):
            t_days = np.atleast_1d(raw["t"]).astype(float)
            v = np.atleast_1d(raw["v"]).astype(float)
            q_mah = np.atleast_1d(raw["q"]).astype(float)
            T = np.atleast_1d(raw["T"]).astype(float)
            if t_days.size < 3:
                return None
            t_s = (t_days - t_days[0]) * 86400.0  # datenum days -> seconds, zeroed
            t_hours = t_days * 24.0
            i_ma = np.gradient(q_mah, t_hours)  # dq/dt, mA (q in mAh, t in hours)
            i_a = i_ma / 1000.0
            return {"t": t_s, "V": v, "I": i_a, "T": T}, q_mah

        charge, q_ch = _build(ch_raw)
        discharge, q_dc = _build(dc_raw)
        if charge is None or discharge is None:
            continue

        cap = float(abs(q_dc[-1] - q_dc[0])) / 1000.0  # mAh -> Ah
        if cap <= 0:
            continue

        yield {
            "cycle_idx": cyc_num if cyc_num > 0 else 1,  # cyc0000 -> treat as cycle 1
            "discharge_capacity": cap,
            "charge": charge,
            "discharge": discharge,
        }


# --------------------------------------------------------------------------
# HUST (Ma et al. 2022, Mendeley Data nsc7hnsg4s) (Stage 5.1)
#
# Source: https://data.mendeley.com/datasets/nsc7hnsg4s/2 (1.19 GB zip,
# 77 pickled dicts, one per A123 APR18650M1A LFP cell, 30degC). Each pickle
# is {cell_id: {"rul": {cycle: rul_at_cycle}, "dq": ..., "data": {cycle:
# DataFrame}}}; every cycle's DataFrame has clean columns Status/
# `Current (mA)`/`Voltage (V)`/`Capacity (mAh)`/`Time (s)`, charge/discharge
# split by the Status string (discharge is itself multi-stage - "Constant
# current discharge_0..3" - hence the .str.contains("discharge") match
# rather than an exact string). Current sign verified already matches this
# project's convention (charge>0, discharge<0) - no flip needed. Discharge
# `Capacity (mAh)` counts DOWN from the cycle's full discharged capacity to
# ~0 (remaining-to-discharge, not cumulative-discharged) - true per-cycle
# discharge capacity is (max-min) within the discharge rows, same pattern
# already used for CALCE's cumulative-within-file capacity column.
# --------------------------------------------------------------------------

def hust_cell_ids():
    return sorted(p.stem for p in HUST_DIR.glob("*.pkl"))


def iterate_hust_cycles(cell_id: str):
    import pickle

    with open(HUST_DIR / f"{cell_id}.pkl", "rb") as f:
        raw = pickle.load(f)[cell_id]
    data = raw["data"]

    for cyc_num in sorted(data.keys()):
        df = data[cyc_num]
        status = df["Status"].astype(str)
        charge_mask = status.str.contains("charge") & ~status.str.contains("discharge")
        discharge_mask = status.str.contains("discharge")
        if charge_mask.sum() < 2 or discharge_mask.sum() < 2:
            continue

        gc = df[charge_mask]
        gd = df[discharge_mask]
        cap_series = gd["Capacity (mAh)"].to_numpy(float)
        cap = float(abs(cap_series.max() - cap_series.min())) / 1000.0  # mAh -> Ah
        if cap <= 0:
            continue

        yield {
            "cycle_idx": int(cyc_num),
            "discharge_capacity": cap,
            "charge": {
                "t": gc["Time (s)"].to_numpy(float),
                "V": gc["Voltage (V)"].to_numpy(float),
                "I": gc["Current (mA)"].to_numpy(float) / 1000.0,  # mA -> A
                "T": None,  # not logged in this dataset
            },
            "discharge": {
                "t": gd["Time (s)"].to_numpy(float),
                "V": gd["Voltage (V)"].to_numpy(float),
                "I": gd["Current (mA)"].to_numpy(float) / 1000.0,
                "T": None,
            },
        }


# --------------------------------------------------------------------------
# NASA PCoE Randomized Battery Usage Data Set (Part B, data-expansion pass)
#
# Source: NASA's own S3-hosted repository, https://phm-datasets.s3.
# amazonaws.com/NASA/11.+Randomized+Battery+Usage+Data+Set.zip (1.07 GB,
# verified currently downloadable directly - NASA's PCoE listing page
# was checked live before assuming otherwise). 28 LG Chem 18650 cells
# (RW1-RW28, 2.1 Ah nominal), 7 sub-experiments (uniform-random-walk
# discharge, variable recharge, skewed-high/low load at room temp and
# 40degC) - a genuinely different NASA collection from the B00XX cells
# already in this project (different chemistry/vendor, randomized-load
# protocol vs. B00XX's fixed CC/CV cycling).
#
# Each cell's .mat is NOT organized into discrete numbered cycles like
# every other adapter in this project - it's one long stream of `step`
# records (rest/charge/discharge segments of a continuous randomized-
# current profile), with periodic REFERENCE charge/discharge steps
# (exact `comment` fields "reference charge"/"reference discharge",
# confirmed by direct inspection - NOT the same string as "rest post
# reference charge/discharge", which are separate rest-type steps
# filtered out here) interspersed roughly every ~900-1000 steps to
# provide comparable capacity-fade checkpoints - the SAME structural
# pattern as XJTU's Sim_satellite "[test capacity]" checkpoints (Stage
# 5 follow-on), handled the same way: only the reference charge/
# discharge PAIRS are yielded as cycle records, not the randomized-
# load steps in between (which have no fixed, comparable depth and
# would break this project's SOH convention the same way Sim_
# satellite's regular cycles did).
#
# Sign convention VERIFIED, not assumed, and found to need a flip: this
# dataset logs reference-charge current NEGATIVE and reference-discharge
# current POSITIVE (checked directly on RW1: charge I in [-2.008,-0.01],
# discharge I in [0.999,1.005]) - the OPPOSITE of this project's
# convention (charge>0, discharge<0) and of the OTHER NASA dataset's own
# convention. Flipped here (negated) to match every other adapter.
#
# No direct capacity field on a `step` - discharge_capacity computed by
# trapezoidal integration of |I| dt over the reference-discharge step,
# the same convention already documented in ica_dv_dc.py's own
# docstring for computing capacity uniformly from raw current.
# --------------------------------------------------------------------------

def nasa_randomized_cell_ids():
    return [f"RW{i}" for i in range(1, 29)]


def iterate_nasa_randomized_cycles(cell_id: str):
    matches = list(NASA_RANDOMIZED_DIR.rglob(f"{cell_id}.mat"))
    if not matches:
        raise FileNotFoundError(f"{cell_id}.mat not found under {NASA_RANDOMIZED_DIR}")
    mat = sio.loadmat(matches[0], simplify_cells=True)
    steps = mat["data"]["step"]

    cycle_idx = 0
    i = 0
    n = len(steps)
    while i < n:
        s = steps[i]
        if s.get("comment") != "reference charge":
            i += 1
            continue
        # look forward (skipping any REST step whose own comment also
        # says "reference" - e.g. "rest post reference charge", "rest
        # prior reference discharge" - both forms seen across the 7
        # sub-datasets, confirmed by direct inspection of RW1 and RW25)
        # for the matching discharge. The 4 "Skewed" sub-datasets
        # (RW13/17/21/25 and neighbors) use "reference power discharge"
        # (constant-POWER, not constant-current) instead of "reference
        # discharge" as their own capacity-check step - confirmed by
        # direct inspection (RW25 has zero "reference discharge" steps
        # at all, only "reference power discharge") before accepting
        # both as valid closing matches. Either is a legitimate, full,
        # comparable reference test - the capacity computation below
        # (trapz of |I|dt) is agnostic to whether the step was current-
        # or power-controlled.
        DISCHARGE_MATCH = ("reference discharge", "reference power discharge")
        j = i + 1
        while j < n and steps[j].get("comment") not in DISCHARGE_MATCH:
            if "reference" not in str(steps[j].get("comment", "")).lower():
                break  # not a reference block - abandon this pairing attempt
            j += 1
        if j >= n or steps[j].get("comment") not in DISCHARGE_MATCH:
            i += 1
            continue

        ch, dc = s, steps[j]
        tc = np.atleast_1d(ch["relativeTime"]).astype(float)
        Vc = np.atleast_1d(ch["voltage"]).astype(float)
        Ic = -np.atleast_1d(ch["current"]).astype(float)  # sign flip, see module docstring
        Tc = np.atleast_1d(ch["temperature"]).astype(float)

        td = np.atleast_1d(dc["relativeTime"]).astype(float)
        Vd = np.atleast_1d(dc["voltage"]).astype(float)
        Id = -np.atleast_1d(dc["current"]).astype(float)
        Td = np.atleast_1d(dc["temperature"]).astype(float)

        # sensor-failure sentinel found and handled, not silently
        # passed through: RW2 (every cycle) logs temperature around
        # -4093.9degC, a physically-impossible value confirmed by
        # direct inspection, not a rare one-off; RW2/RW3/RW18 also
        # have a handful of isolated transient glitch points per
        # affected cycle (a few points of several hundred, values like
        # -54.9/-98.7/-79.2degC - scattered, not one fixed sentinel,
        # but equally impossible for a battery test with no genuinely
        # sub-freezing protocol anywhere else in this dataset - checked
        # directly: every other cell's T stays within [18,60]degC).
        # NaN'd out at a -50degC threshold (matching this project's
        # existing convention for a missing/unusable T channel, e.g.
        # CALCE's/HUST's T=None) rather than feeding a garbage value
        # into MATC/MATD/etc.
        Tc = np.where(Tc < -50, np.nan, Tc)
        Td = np.where(Td < -50, np.nan, Td)

        if len(tc) < 2 or len(td) < 2:
            i = j + 1
            continue

        discharge_capacity = float(np.trapezoid(np.abs(Id), td)) / 3600.0
        if discharge_capacity <= 0:
            i = j + 1
            continue

        cycle_idx += 1
        yield {
            "cycle_idx": cycle_idx,
            "discharge_capacity": discharge_capacity,
            "charge": {"t": tc, "V": Vc, "I": Ic, "T": Tc},
            "discharge": {"t": td, "V": Vd, "I": Id, "T": Td},
        }
        i = j + 1
