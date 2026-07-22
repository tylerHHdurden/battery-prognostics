"""
Load NASA PCoE Li-ion Battery Aging Dataset (.mat files) and inspect structure.

Source: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
Cells: B0005, B0006, B0007, B0018

Each .mat file contains a top-level struct named after the battery id
(e.g. B0005) with a `cycle` field: a 1xN struct array where each entry is
one experiment cycle (charge / discharge / impedance) with fields:
    type       - 'charge' | 'discharge' | 'impedance'
    ambient_temperature
    time       - [year month day hour minute second] of start
    data       - struct of measurement arrays (varies by type), e.g.
                 Voltage_measured, Current_measured, Temperature_measured,
                 Current_charge, Voltage_charge, Time, Capacity (discharge only)
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

NASA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "nasa" / "B0005_B0006_B0007_B0018"
CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]


def load_cell(cell_id: str) -> np.ndarray:
    """Load a NASA .mat file and return the cycle struct array."""
    path = NASA_DIR / f"{cell_id}.mat"
    mat = sio.loadmat(path, simplify_cells=True)
    return mat[cell_id]["cycle"]


def describe_cell(cell_id: str) -> None:
    cycles = load_cell(cell_id)
    n_cycles = len(cycles)
    type_counts: dict[str, int] = {}
    for c in cycles:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1

    print(f"\n=== {cell_id} ===")
    print(f"Total logged cycles/operations: {n_cycles}")
    print(f"Operation type counts: {type_counts}")

    # Show detailed structure of the first charge, discharge, and impedance entries
    seen_types = set()
    for c in cycles:
        t = c["type"]
        if t in seen_types:
            continue
        seen_types.add(t)
        data = c["data"]
        print(f"\n  -- Example '{t}' cycle --")
        print(f"     ambient_temperature: {c.get('ambient_temperature')}")
        print(f"     time: {c.get('time')}")
        print(f"     data fields: {list(data.keys())}")
        for k, v in data.items():
            v_arr = np.atleast_1d(v)
            print(f"       {k}: shape={np.shape(v_arr)} dtype={v_arr.dtype}")
        if len(seen_types) == 3:
            break

    # Discharge capacity fade summary (the key prognostics signal)
    discharge_caps = [
        c["data"]["Capacity"]
        for c in cycles
        if c["type"] == "discharge" and "Capacity" in c["data"]
    ]
    if discharge_caps:
        caps = np.array(discharge_caps, dtype=float).flatten()
        print(f"\n  Discharge cycles with capacity readings: {len(caps)}")
        print(f"  Capacity range: {caps.min():.4f} Ah -> {caps.max():.4f} Ah "
              f"(first: {caps[0]:.4f} Ah, last: {caps[-1]:.4f} Ah)")


def main() -> None:
    for cell_id in CELL_IDS:
        describe_cell(cell_id)


if __name__ == "__main__":
    main()
