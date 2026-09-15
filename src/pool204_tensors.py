"""
Data-expansion pass, Part A: shared sequence-tensor loader for the full
204-battery pool (23 NASA + 181 MIT), same {battery_id: (X, soh, rul,
dataset)} contract as stage4_pool.load_all_battery_tensors_stage4 (same
Severson-aware EOL/RUL convention, same get_cycle_tensor pipeline) -
reused via make_xy exactly as Stage 4's own step2a/step2b scripts do,
for full methodological parity with the already-verified, deployed
pipeline rather than a new, separately-trusted approach.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from sequence_features import get_cycle_tensor
from rul_labels import compute_eol_and_rul_severson_aware, soh_per_cycle

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def _tensors_from_cycles(cycles, soh_map, rul_map):
    Xs, sohs, ruls = [], [], []
    for c in cycles:
        x = get_cycle_tensor(c, n_bins=200)
        if x is None:
            continue
        Xs.append(x)
        sohs.append(soh_map[c["cycle_idx"]])
        ruls.append(rul_map[c["cycle_idx"]])
    if not Xs:
        return None, None, None
    return np.stack(Xs), np.array(sohs, dtype=np.float32), np.array(ruls, dtype=np.float32)


def load_battery_tensors_pool204():
    split = json.loads((PROC_DIR / "battery_split_expanded_b0018pinned.json").read_text())
    all_204_ids = set(split["train_ids"] + split["test_ids"])
    nasa_ids = sorted(b for b in all_204_ids if b.startswith("B0"))
    mit_ids = sorted(b for b in all_204_ids if not b.startswith("B0"))

    mit_full = json.loads((PROC_DIR / "mit_full_cells.json").read_text())
    mit_lookup = {e["global_id"]: (e["batch_file"], e["cell_index"]) for e in mit_full}

    out = {}
    for cid in nasa_ids:
        cycles = list(iterate_nasa_cycles(cid))
        eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=cid)
        soh_map = soh_per_cycle(cycles)
        X, soh, rul = _tensors_from_cycles(cycles, soh_map, rul_map)
        if X is not None:
            out[cid] = (X.astype(np.float32), soh, rul, "NASA")
        print(f"[pool204-tensors] loaded NASA/{cid}: {X.shape if X is not None else None}")

    for cid in mit_ids:
        batch_file, cell_index = mit_lookup[cid]
        cycles = list(iterate_mit_cycles(batch_file, cell_index))
        eol_cycle, censored, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=cid)
        soh_map = soh_per_cycle(cycles)
        X, soh, rul = _tensors_from_cycles(cycles, soh_map, rul_map)
        if X is not None:
            out[cid] = (X.astype(np.float32), soh, rul, "MIT")
        print(f"[pool204-tensors] loaded MIT/{cid}: {X.shape if X is not None else None}")

    return out
