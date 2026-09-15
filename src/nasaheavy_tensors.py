"""
Data-expansion pass, item 1: shared sequence-tensor + HI-row loader for
the NASA-heavy pool - the designed test of whether the "more MIT-heavy
training data specializes the model toward MIT-adjacent domains"
mechanism (found correlating cleanly with domain-classifier AUC in the
204-battery pool result) is genuinely controllable, not just a
one-time correlation.

Composition, reasoned and disclosed:
- 23 "clean" NASA B00XX batteries (the 204-pool's own curated list,
  reused unchanged - not re-deriving a new curation).
- 6 "recovered" NASA B00XX batteries (RECOVERED_NASA, via stage4_
  recovered_batteries.get_recovered_battery - the SAME corrected
  loader already established and trusted elsewhere in this project,
  not a new, unverified inclusion of known-artifact-affected data).
- 28 NASA Randomized Battery Usage cells (RW1-RW28, Part B's newly-
  acquired dataset, verified in that same pass).
- = 57 NASA-family batteries total.
- 28 MIT batteries - held EXACTLY at the original 42-battery pool's
  own MIT subset (mit_subset.json), UNCHANGED, deliberately NOT
  expanded or reduced further - isolates "more/different NASA data"
  as the ONLY compositional change relative to the 42-battery pool,
  for the cleanest possible controlled comparison. No recovered MIT
  batteries added either (RECOVERED_MIT exists but is excluded here -
  adding it would grow MIT too, working against this pool's whole
  point).

Total: 85 batteries, NASA:MIT = 57:28 = 67.1%:32.9% NASA share -
a large, deliberate swing from the 204-pool's 11.3% and the 42-pool's
23.8%, in the OPPOSITE direction.

Same {battery_id: (X, soh, rul, dataset)} tensor contract as
pool204_tensors.py/stage4_pool.py, for full methodological parity.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles, iterate_nasa_randomized_cycles
from sequence_features import get_cycle_tensor
from rul_labels import compute_eol_and_rul_severson_aware, soh_per_cycle
from stage4_recovered_batteries import get_recovered_battery, RECOVERED_NASA
from health_indicators import compute_health_indicators

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

CLEAN_NASA_204 = [
    "B0005", "B0006", "B0007", "B0018", "B0025", "B0026", "B0027", "B0028",
    "B0029", "B0030", "B0031", "B0032", "B0042", "B0043", "B0044", "B0045",
    "B0046", "B0047", "B0048", "B0053", "B0054", "B0055", "B0056",
]


def _tensors_from_cycles(cycles, soh_map, rul_map):
    Xs, sohs, ruls, idxs = [], [], [], []
    for c in cycles:
        x = get_cycle_tensor(c, n_bins=200)
        if x is None:
            continue
        Xs.append(x)
        sohs.append(soh_map[c["cycle_idx"]])
        ruls.append(rul_map[c["cycle_idx"]])
        idxs.append(c["cycle_idx"])
    if not Xs:
        return None, None, None, None
    return np.stack(Xs), np.array(sohs, dtype=np.float32), np.array(ruls, dtype=np.float32), idxs


def nasaheavy_battery_lists():
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    mit_ids = [e["global_id"] for e in mit_subset]
    return CLEAN_NASA_204, RECOVERED_NASA, [f"RW{i}" for i in range(1, 29)], mit_ids


def load_battery_tensors_nasaheavy():
    """Returns {battery_id: (X, soh, rul, dataset)}, dataset in
    {'NASA','NASA_RANDOMIZED','MIT'} (NASA_RANDOMIZED kept distinct
    from NASA for reporting - it's a genuinely different chemistry/
    format collection, not folded silently into 'NASA')."""
    clean_nasa, recovered_nasa, rw_ids, mit_ids = nasaheavy_battery_lists()
    out = {}

    for cid in clean_nasa:
        cycles = list(iterate_nasa_cycles(cid))
        eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=cid)
        soh_map = soh_per_cycle(cycles)
        X, soh, rul, idxs = _tensors_from_cycles(cycles, soh_map, rul_map)
        if X is not None:
            out[cid] = (X.astype(np.float32), soh, rul, "NASA")
        print(f"[nasaheavy-tensors] loaded NASA/{cid}: {X.shape if X is not None else None}")

    for cid in recovered_nasa:
        kept_cycles, soh_map, rul_map, eol_cycle, censored = get_recovered_battery("NASA", cid)
        X, soh, rul, idxs = _tensors_from_cycles(kept_cycles, soh_map, rul_map)
        if X is not None:
            out[cid] = (X.astype(np.float32), soh, rul, "NASA")
        print(f"[nasaheavy-tensors] loaded RECOVERED NASA/{cid}: {X.shape if X is not None else None}")

    for cid in rw_ids:
        cycles = list(iterate_nasa_randomized_cycles(cid))
        eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=None)
        soh_map = soh_per_cycle(cycles)
        X, soh, rul, idxs = _tensors_from_cycles(cycles, soh_map, rul_map)
        if X is not None:
            out[cid] = (X.astype(np.float32), soh, rul, "NASA_RANDOMIZED")
        print(f"[nasaheavy-tensors] loaded NASA_RANDOMIZED/{cid}: {X.shape if X is not None else None}")

    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full = json.load(f)
    mit_lookup = {e["global_id"]: (e["batch_file"], e["cell_index"]) for e in mit_full}
    for cid in mit_ids:
        batch_file, cell_index = mit_lookup[cid]
        cycles = list(iterate_mit_cycles(batch_file, cell_index))
        eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=cid)
        soh_map = soh_per_cycle(cycles)
        X, soh, rul, idxs = _tensors_from_cycles(cycles, soh_map, rul_map)
        if X is not None:
            out[cid] = (X.astype(np.float32), soh, rul, "MIT")
        print(f"[nasaheavy-tensors] loaded MIT/{cid}: {X.shape if X is not None else None}")

    return out


def build_hi_rows_nasaheavy():
    """HI-table rows (16 raw HIs + SOH/RUL) for the same pool, same
    convention as run_pool204_step1_feature_regen.py's process_battery,
    reused for the pool's own hi_table_nasaheavy.parquet."""
    clean_nasa, recovered_nasa, rw_ids, mit_ids = nasaheavy_battery_lists()
    rows = []

    def process(dataset, bid, cycles):
        if len(cycles) < 5:
            print(f"[nasaheavy-hi] SKIP {dataset}/{bid}: only {len(cycles)} usable cycles")
            return
        eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(
            cycles, global_id=(bid if dataset != "NASA_RANDOMIZED" else None))
        soh_map = soh_per_cycle(cycles)
        for c in cycles:
            his = compute_health_indicators(c)
            row = {"dataset": dataset, "battery_id": bid, "cycle_idx": c["cycle_idx"],
                   "discharge_capacity": c["discharge_capacity"], "SOH": soh_map[c["cycle_idx"]],
                   "RUL": rul_map[c["cycle_idx"]]}
            row.update(his)
            rows.append(row)
        print(f"[nasaheavy-hi] DONE {dataset}/{bid}: {len(cycles)} cycles, EOL={eol_cycle} censored={censored}")

    for cid in clean_nasa:
        process("NASA", cid, list(iterate_nasa_cycles(cid)))
    for cid in recovered_nasa:
        kept_cycles, soh_map, rul_map, eol_cycle, censored = get_recovered_battery("NASA", cid)
        for c in kept_cycles:
            his = compute_health_indicators(c)
            row = {"dataset": "NASA", "battery_id": cid, "cycle_idx": c["cycle_idx"],
                   "discharge_capacity": c["discharge_capacity"], "SOH": soh_map[c["cycle_idx"]],
                   "RUL": rul_map[c["cycle_idx"]]}
            row.update(his)
            rows.append(row)
        print(f"[nasaheavy-hi] DONE RECOVERED NASA/{cid}: {len(kept_cycles)} cycles")
    for cid in rw_ids:
        process("NASA_RANDOMIZED", cid, list(iterate_nasa_randomized_cycles(cid)))

    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full = json.load(f)
    mit_lookup = {e["global_id"]: (e["batch_file"], e["cell_index"]) for e in mit_full}
    for cid in mit_ids:
        batch_file, cell_index = mit_lookup[cid]
        process("MIT", cid, list(iterate_mit_cycles(batch_file, cell_index)))

    for cid in ["CS2_35", "CS2_36", "CS2_37"]:
        from data_adapters import iterate_calce_cycles
        process("CALCE", cid, list(iterate_calce_cycles(cid)))

    import pandas as pd
    return pd.DataFrame(rows)
