"""
Session 22: NASA PCoE EIS (Electrochemical Impedance Spectroscopy)
feature extraction. NASA's raw .mat files include 'impedance'-type cycle
entries interleaved with charge/discharge in the same top-level struct
array. Each impedance entry contains a swept-frequency measurement
(NASA's documented protocol: 0.1Hz-5kHz) AND two ALREADY-FITTED
equivalent-circuit parameters: `Re` (electrolyte/ohmic resistance) and
`Rct` (charge-transfer resistance) - both present in every impedance
entry checked (278/278 for B0005/B0006/B0007, 53/53 for B0018). Using
these avoids any new curve-fitting dependency (no lmfit/impedance.py
Randles-circuit fit needed) - NASA already provides the fitted values
directly in the raw file; extracting them is a read, not a fit.

**Limitation reported explicitly, not worked around silently**: no
frequency vector is stored anywhere in the .mat structure - only the
complex impedance ARRAY itself (e.g. `Battery_impedance`, shape (48,)
complex128). NASA's own documentation states the sweep spans
0.1Hz-5kHz, but without an explicit per-sample frequency array in the
FILE ITSELF, indexing "impedance at a SPECIFIC frequency" (e.g. "Z at
1kHz") cannot be done from this data alone - it would require importing
a hardcoded frequency-point assumption from external documentation that
this script has no way to verify against the actual file. So this
script extracts Re/Rct (frequency-index-free, NASA's own fitted
parameters) plus one frequency-agnostic magnitude summary
(mean |Battery_impedance| across the whole sweep) - 3 new candidate HIs
total, not a per-frequency breakdown.

**Alignment limitation, also reported explicitly**: impedance
measurements are NOT one-to-one with discharge cycles - NASA ran EIS
sweeps on its own schedule, not once per discharge (e.g. B0005: 168
discharges vs. 278 impedance tests). Every cycle entry (charge/
discharge/impedance alike) carries its own `time` field
([year,month,day,hour,minute,second]), so each discharge cycle here is
matched to whichever impedance test's timestamp is CLOSEST in time -
a standard nearest-neighbor alignment, NOT an exact per-cycle EIS
reading, and the resulting time gap is reported per match so this
approximation's quality is visible rather than assumed.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
NASA_DIR = ROOT / "data" / "raw" / "nasa" / "B0005_B0006_B0007_B0018"
PROC_DIR = ROOT / "data" / "processed"
NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]


def _to_datetime(time_field) -> datetime:
    y, mo, d, h, mi, s = time_field
    return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))


def extract_eis_for_cell(cell_id: str) -> pd.DataFrame:
    """One row per DISCHARGE cycle_idx, numbered with the EXACT SAME
    charge-then-discharge-with-capacity logic as
    data_adapters.iterate_nasa_cycles (so cycle_idx here lines up 1:1
    with the cycle_idx already used throughout hi_table.parquet),
    carrying the nearest-in-time impedance test's Re/Rct/mean|Z|."""
    mat = sio.loadmat(NASA_DIR / f"{cell_id}.mat", simplify_cells=True)
    cycles = mat[cell_id]["cycle"]

    imp_times, imp_re, imp_rct, imp_zmag = [], [], [], []
    for c in cycles:
        if c["type"] == "impedance" and "Re" in c["data"] and "Rct" in c["data"]:
            imp_times.append(_to_datetime(c["time"]))
            imp_re.append(float(np.atleast_1d(c["data"]["Re"])[0]))
            imp_rct.append(float(np.atleast_1d(c["data"]["Rct"])[0]))
            z = np.atleast_1d(c["data"].get("Battery_impedance", np.array([np.nan])))
            imp_zmag.append(float(np.abs(z).mean()))
    if not imp_times:
        return pd.DataFrame(columns=["battery_id", "cycle_idx", "EIS_Re", "EIS_Rct",
                                      "EIS_Zmag_mean", "eis_time_gap_hours"])

    rows = []
    pending_charge_seen = False
    cycle_idx = 0
    for c in cycles:
        if c["type"] == "charge":
            pending_charge_seen = True
        elif c["type"] == "discharge":
            d = c["data"]
            cap = float(np.atleast_1d(d["Capacity"])[0]) if "Capacity" in d else np.nan
            if pending_charge_seen and not np.isnan(cap):
                cycle_idx += 1
                dt = _to_datetime(c["time"])
                deltas = np.array([abs((dt - it).total_seconds()) for it in imp_times])
                nearest = int(np.argmin(deltas))
                rows.append({
                    "battery_id": cell_id, "cycle_idx": cycle_idx,
                    "EIS_Re": imp_re[nearest], "EIS_Rct": imp_rct[nearest],
                    "EIS_Zmag_mean": imp_zmag[nearest],
                    "eis_time_gap_hours": deltas[nearest] / 3600.0,
                })
    return pd.DataFrame(rows)


def main():
    all_df = []
    for cid in NASA_CELLS:
        df = extract_eis_for_cell(cid)
        print(f"[eis] {cid}: {len(df)} discharge cycles matched to nearest impedance test "
              f"(time gap median={df['eis_time_gap_hours'].median():.2f}h, "
              f"max={df['eis_time_gap_hours'].max():.2f}h)")
        all_df.append(df)
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(PROC_DIR / "eis_features_nasa.csv", index=False)
    print(f"\n[eis] saved {len(out)} rows (NASA-only - MIT/CALCE have no EIS data at all, "
          f"see DEVELOPMENT_LOG.md) to data/processed/eis_features_nasa.csv")
    print(out[["EIS_Re", "EIS_Rct", "EIS_Zmag_mean"]].describe().to_string())


if __name__ == "__main__":
    main()
