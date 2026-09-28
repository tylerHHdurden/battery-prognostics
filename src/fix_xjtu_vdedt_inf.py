"""
Surgical fix for the XJTU VDEDT=inf bug (see DEVELOPMENT_LOG.md):
health_indicators.py's VDEDT formula is now fixed (explicit near-zero
time-delta guard), but the FIX doesn't retroactively correct the
already-persisted data/processed/stage5_1_xjtu_merged.parquet, which
was built once by run_stage5_1_new_datasets_eval.py calling the OLD,
buggy compute_health_indicators.

Recomputes VDEDT for EVERY XJTU row (not just the 194 known-bad ones -
consistent, not cherry-picked, in case the fix's threshold also
quietly changes any other row that was finite-but-noisy rather than
outright inf) via the SAME raw-cycle-iteration this parquet was
originally built from, and overwrites ONLY the VDEDT column in place -
fusion embeddings, other HI columns, SOH, everything else is untouched,
not regenerated, to keep this a minimal, low-risk, targeted fix.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health_indicators import compute_health_indicators
from run_stage5_1_new_datasets_eval import xjtu_cell_ids_soh_valid, iterate_xjtu_cycles

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def main():
    t0 = time.time()
    print("=== Fixing XJTU's VDEDT=inf bug: recomputing VDEDT for every row, in place ===")
    path = PROC_DIR / "stage5_1_xjtu_merged.parquet"
    df = pd.read_parquet(path)
    n_inf_before = int(np.isinf(df["VDEDT"]).sum())
    print(f"[fix-vdedt] loaded {len(df)} rows, {n_inf_before} with VDEDT=inf before the fix")

    vdedt_map = {}
    cell_ids = xjtu_cell_ids_soh_valid()
    for i, cid in enumerate(cell_ids):
        cycles = list(iterate_xjtu_cycles(cid))
        for c in cycles:
            his = compute_health_indicators(c)
            vdedt_map[(cid, c["cycle_idx"])] = his["VDEDT"]
        if (i + 1) % 10 == 0:
            print(f"[fix-vdedt] recomputed {i+1}/{len(cell_ids)} cells ({time.time()-t0:.0f}s elapsed)")

    new_vdedt = df.apply(lambda r: vdedt_map.get((r["battery_id"], r["cycle_idx"]), r["VDEDT"]), axis=1)
    n_matched = int((new_vdedt != df["VDEDT"]).sum())
    n_unmatched = int(sum(1 for k in zip(df["battery_id"], df["cycle_idx"]) if k not in vdedt_map))
    df["VDEDT"] = new_vdedt

    n_inf_after = int(np.isinf(df["VDEDT"]).sum())
    print(f"[fix-vdedt] {n_matched} rows had a changed VDEDT value, {n_unmatched} rows had no "
          f"matching recomputed cycle (kept their existing value)")
    print(f"[fix-vdedt] VDEDT=inf count: {n_inf_before} -> {n_inf_after}")
    assert n_inf_after == 0, f"STOPPING: {n_inf_after} rows still have VDEDT=inf after the fix - not saving"

    df.to_parquet(path)
    print(f"[fix-vdedt] saved {path}")
    print(f"\n[fix-vdedt] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
