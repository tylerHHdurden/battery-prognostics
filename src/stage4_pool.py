"""
Stage 4: shared sequence-tensor loader for the full retrain pool
(original 32 + Stage 2.1's recovered batteries), covering BOTH the
original batteries (reprocessed with the Severson-aware EOL/RUL
convention, matching the new hi_table.parquet exactly - fixing a
pre-existing, previously-unnoticed inconsistency where
sequence_features.build_dataset_tensors always used the PLAIN
compute_eol_and_rul/soh_per_cycle regardless of hi_table's own
convention) and the recovered batteries (via stage4_recovered_
batteries.py's filtered+corrected loader).

Returns the SAME {battery_id: (X, soh, rul, dataset)} contract as
train_deep_models.load_all_battery_tensors, so every downstream
consumer (make_xy, channel-norm fitting, encoder/VLSTM/joint-model
training) works unchanged against it.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from sequence_features import get_cycle_tensor
from rul_labels import compute_eol_and_rul_severson_aware
from stage4_recovered_batteries import get_recovered_battery, ALL_RECOVERED

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

_ORIGINAL_NASA = ["B0005", "B0006", "B0007", "B0018"]


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
        return None, None, None
    return np.stack(Xs), np.array(sohs, dtype=np.float32), np.array(ruls, dtype=np.float32)


def load_all_battery_tensors_stage4():
    """Returns dict battery_id -> (X, soh, rul, dataset), covering the
    full Stage 4 pool. SOH/RUL match the new hi_table.parquet exactly
    for every battery (originals: Severson-aware, previously not the
    case in this sequence-tensor path specifically; recovered: the
    Stage 2.1-corrected values)."""
    out = {}

    for cid in _ORIGINAL_NASA:
        cycles = list(iterate_nasa_cycles(cid))
        eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=cid)
        from rul_labels import soh_per_cycle
        soh_map = soh_per_cycle(cycles)
        X, soh, rul = _tensors_from_cycles(cycles, soh_map, rul_map)
        if X is not None:
            out[cid] = (X.astype(np.float32), soh, rul, "NASA")
        print(f"[stage4-pool] loaded NASA/{cid}: {X.shape if X is not None else None}")

    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    for entry in mit_subset:
        gid = entry["global_id"]
        cycles = list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))
        eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=gid)
        from rul_labels import soh_per_cycle
        soh_map = soh_per_cycle(cycles)
        X, soh, rul = _tensors_from_cycles(cycles, soh_map, rul_map)
        if X is not None:
            out[gid] = (X.astype(np.float32), soh, rul, "MIT")
        print(f"[stage4-pool] loaded MIT/{gid}: {X.shape if X is not None else None}")

    for ds, bid in ALL_RECOVERED:
        kept_cycles, soh_map, rul_map, eol_cycle, censored = get_recovered_battery(ds, bid)
        X, soh, rul = _tensors_from_cycles(kept_cycles, soh_map, rul_map)
        if X is not None:
            out[bid] = (X.astype(np.float32), soh, rul, ds)
        print(f"[stage4-pool] loaded RECOVERED {ds}/{bid}: {X.shape if X is not None else None}")

    return out
