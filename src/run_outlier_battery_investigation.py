"""
Follow-up Part A.1: direct raw-trace inspection of the 3 outlier
batteries session 35 Part 1's early-cycle capacity check flagged
(NASA/B0045, MIT/b2c15, MIT/b2c16) but never investigated further -
same method as B0053's correction in session 35 Part 1 (read the
actual per-cycle values directly, don't infer from a summary stat).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

N_CYCLES_TO_SHOW = 20


def inspect_nasa(cid):
    print(f"\n[outlier-invest] === NASA/{cid} - first {N_CYCLES_TO_SHOW} cycles' discharge capacity ===")
    caps = []
    for i, c in enumerate(iterate_nasa_cycles(cid)):
        caps.append(c["discharge_capacity"])
        if i + 1 >= N_CYCLES_TO_SHOW:
            break
    caps = np.array(caps)
    print(f"[outlier-invest] {cid}: {np.round(caps, 4).tolist()}")
    print(f"[outlier-invest] {cid}: min={caps.min():.4f} (cycle {int(np.argmin(caps))+1}), "
          f"cycle-1={caps[0]:.4f}, mean(first 20)={caps.mean():.4f}, std={caps.std():.4f}")
    return caps


def inspect_mit(global_id, batch_file, cell_index):
    print(f"\n[outlier-invest] === MIT/{global_id} - first {N_CYCLES_TO_SHOW} cycles' discharge capacity ===")
    caps = []
    for c in iterate_mit_cycles(batch_file, cell_index, max_cycles=N_CYCLES_TO_SHOW):
        caps.append(c["discharge_capacity"])
    caps = np.array(caps)
    print(f"[outlier-invest] {global_id}: {np.round(caps, 4).tolist()}")
    print(f"[outlier-invest] {global_id}: min={caps.min():.4f} (cycle {int(np.argmin(caps))+1}), "
          f"cycle-1={caps[0]:.4f}, mean(first 20)={caps.mean():.4f}, std={caps.std():.4f}")
    return caps


def main():
    b0045_caps = inspect_nasa("B0045")

    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full = json.load(f)
    b2c15_entry = next(e for e in mit_full if e["global_id"] == "b2c15")
    b2c16_entry = next(e for e in mit_full if e["global_id"] == "b2c16")
    b2c15_caps = inspect_mit("b2c15", b2c15_entry["batch_file"], b2c15_entry["cell_index"])
    b2c16_caps = inspect_mit("b2c16", b2c16_entry["batch_file"], b2c16_entry["cell_index"])

    # for context: a few normal NASA/MIT batteries' first-cycle capacity, for scale comparison
    print("\n[outlier-invest] === reference: a few normal batteries' cycle-1 capacity, for scale ===")
    for cid in ["B0005", "B0025", "B0030"]:
        c1 = next(iter(iterate_nasa_cycles(cid)))["discharge_capacity"]
        print(f"[outlier-invest] NASA/{cid} cycle-1 capacity: {c1:.4f}")
    for gid in ["b1c4", "b3c0"]:
        entry = next(e for e in mit_full if e["global_id"] == gid)
        c1 = next(iter(iterate_mit_cycles(entry["batch_file"], entry["cell_index"], max_cycles=1)))["discharge_capacity"]
        print(f"[outlier-invest] MIT/{gid} cycle-1 capacity: {c1:.4f}")

    print("\n[outlier-invest] DONE")


if __name__ == "__main__":
    main()
