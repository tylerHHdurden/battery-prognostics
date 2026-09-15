"""
Data-expansion pass, Part A, Step 1: regenerate hi_table for the full
204-battery pool (session 33's "Dataset Expansion Phase 1" - 23 NASA +
181 MIT, all from already-downloaded raw data, zero new downloads)
through the CURRENT pipeline (Severson-aware EOL/RUL, the same
compute_health_indicators/soh_per_cycle used everywhere else) -
regenerating from scratch rather than trusting the existing
hi_table_expanded.parquet, which was VERIFIED STALE before writing
this: b1c20 cycle-1 RUL=532 there (the OLD, pre-Severson-aware value),
not 531 (Severson-aware) - the exact same staleness Stage 4 found and
fixed for the 42-battery pool, now confirmed to also affect the larger
204-battery build.

SCOPE DECISION, disclosed: "the full 204-battery pool" is interpreted
literally as the 204 IDs in battery_split_expanded_b0018pinned.json
(23 NASA + 181 MIT) - NOT combined with Stage 2.1's 10 separately-
recovered batteries (confirmed directly: none of the 10 recovered IDs
appear in the 204-list; they were excluded from a DIFFERENT,
independent sweep). Not silently expanded to 214 - if a combined pool
is wanted, that is a separate, explicit follow-up.

Does NOT touch the deployed data/processed/hi_table.parquet,
battery_split.json, or any file live_inference.py loads - writes to
NEW, separate files (hi_table_pool204.parquet, etc.) so the deployed
42-battery pool and model are completely unaffected regardless of how
this comparison turns out.

CALCE is included in the output hi_table (same structural convention
as every other hi_table in this project) but is NEVER part of any
training split - held out exactly as always.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_calce_cycles, iterate_mit_cycles
from health_indicators import compute_health_indicators
from rul_labels import compute_eol_and_rul_severson_aware, soh_per_cycle

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]


def process_battery(dataset: str, battery_id: str, cycle_iter):
    t0 = time.time()
    cycles = list(cycle_iter)
    if len(cycles) < 5:
        print(f"[pool204-feat] SKIP {dataset}/{battery_id}: only {len(cycles)} usable cycles")
        return None, None

    eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=battery_id)
    soh_map = soh_per_cycle(cycles)

    rows = []
    for c in cycles:
        his = compute_health_indicators(c)
        row = {"dataset": dataset, "battery_id": battery_id, "cycle_idx": c["cycle_idx"],
               "discharge_capacity": c["discharge_capacity"], "SOH": soh_map[c["cycle_idx"]],
               "RUL": rul_map[c["cycle_idx"]]}
        row.update(his)
        rows.append(row)
    df = pd.DataFrame(rows)

    dt = time.time() - t0
    print(f"[pool204-feat] DONE {dataset}/{battery_id}: {len(cycles)} cycles, "
          f"EOL={eol_cycle} censored={censored} ({dt:.1f}s)")
    return df, {"dataset": dataset, "battery_id": battery_id, "n_cycles": len(cycles),
                "eol_cycle": eol_cycle, "censored": censored}


def main():
    t_start = time.time()

    split = json.loads((PROC_DIR / "battery_split_expanded_b0018pinned.json").read_text())
    all_204_ids = set(split["train_ids"] + split["test_ids"])
    nasa_ids = sorted(b for b in all_204_ids if b.startswith("B0"))
    mit_ids = sorted(b for b in all_204_ids if not b.startswith("B0"))
    print(f"[pool204-feat] 204-battery pool: {len(nasa_ids)} NASA + {len(mit_ids)} MIT "
          f"(train={len(split['train_ids'])}, test={len(split['test_ids'])})")

    mit_full = json.loads((PROC_DIR / "mit_full_cells.json").read_text())
    mit_lookup = {e["global_id"]: (e["batch_file"], e["cell_index"]) for e in mit_full}
    missing = [m for m in mit_ids if m not in mit_lookup]
    if missing:
        raise RuntimeError(f"{len(missing)} MIT IDs in the 204-pool have no batch_file/cell_index "
                            f"mapping in mit_full_cells.json: {missing[:10]}")

    all_dfs, all_summaries = [], []

    print("\n[pool204-feat] === NASA (23 batteries) ===")
    for cid in nasa_ids:
        df, summ = process_battery("NASA", cid, iterate_nasa_cycles(cid))
        if df is not None:
            all_dfs.append(df); all_summaries.append(summ)

    print("\n[pool204-feat] === CALCE (3, structural only - never trained on) ===")
    for cid in CALCE_CELLS:
        df, summ = process_battery("CALCE", cid, iterate_calce_cycles(cid))
        if df is not None:
            all_dfs.append(df); all_summaries.append(summ)

    print("\n[pool204-feat] === MIT (181 batteries) ===")
    for cid in mit_ids:
        batch_file, cell_index = mit_lookup[cid]
        df, summ = process_battery("MIT", cid, iterate_mit_cycles(batch_file, cell_index))
        if df is not None:
            all_dfs.append(df); all_summaries.append(summ)

    full_df = pd.concat(all_dfs, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    # duplicate sanity check before trusting anything, same discipline
    # as Stage 4's own feature regen (which caught a real b2c44
    # duplication bug this exact way)
    dup = full_df.duplicated(subset=["dataset", "battery_id", "cycle_idx"]).sum()
    print(f"\n[pool204-feat] duplicate (dataset,battery_id,cycle_idx) rows: {dup}")
    if dup > 0:
        raise RuntimeError(f"{dup} duplicate rows found - not trusting this pool, fix before proceeding")

    n_batteries = full_df.groupby("dataset")["battery_id"].nunique()
    print(f"[pool204-feat] pool composition: {n_batteries.to_dict()}, {len(full_df)} total cycles")

    full_df.to_parquet(PROC_DIR / "hi_table_pool204.parquet")
    summary_df.to_csv(OUT_DIR / "pool204_rul_summary.csv", index=False)
    print(f"\n[pool204-feat] saved data/processed/hi_table_pool204.parquet "
          f"({len(full_df)} rows) and outputs/pool204_rul_summary.csv")

    # mandatory verification, same check Stage 4 used to confirm the
    # Severson-aware convention is actually live (not just in code)
    b1c20_row = full_df[(full_df["battery_id"] == "b1c20") & (full_df["cycle_idx"] == 1)]
    if len(b1c20_row) > 0:
        rul_val = b1c20_row["RUL"].iloc[0]
        print(f"\n[pool204-feat] MANDATORY VERIFICATION: b1c20 cycle-1 RUL={rul_val} "
              f"(expect 531.0 Severson-aware, NOT 532.0 old-convention)")
        assert abs(rul_val - 531.0) < 0.5, f"Severson-aware convention NOT live! Got RUL={rul_val}"
        print("[pool204-feat] VERIFIED: Severson-aware convention is live in this regenerated pool.")
    else:
        print("[pool204-feat] NOTE: b1c20 not in this pool (unexpected - check membership)")

    print(f"\n[pool204-feat] TOTAL TIME: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
