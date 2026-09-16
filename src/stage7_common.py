"""
Stage 7 shared utilities: raw/less-processed sequence-tensor loading for
the canonical 42-battery pool (train/test split) and zero-retrain
loading for the 4 held-out datasets (CALCE, Oxford, HUST, XJTU), reused
identically by all three Stage 7 items (7.1 World Model, 7.2 self-
supervised pretraining, 7.3 neural-operator SPM surrogate) so they train/
evaluate on the SAME data representation, differing only in architecture
- the same discipline as stage1_common.py for the point-prediction
pipeline.

Deliberately NOT the engineered HI-feature pipeline (stage1_common's
CANONICAL_FEATURES / hi_table.parquet) - all three Stage 7 items operate
on raw per-cycle tensors (V_t, I_t, T_t, dQdV, dVdQ, dIdV; see
sequence_features.py), per the task's own explicit instruction that a
World Model (and, by the same reasoning, a physics operator surrogate
and a curve-shape pretext task) needs less-processed sequence data, not
hand-engineered ratios.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import (
    iterate_calce_cycles,
    oxford_cell_ids, iterate_oxford_cycles,
    hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids, iterate_xjtu_cycles,
)
from sequence_features import build_dataset_tensors, compute_channel_norm_stats, apply_channel_norm
from stage4_pool import load_all_battery_tensors_stage4

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]
# Same exclusion as Stage 5.1/5.2 - 8 XJTU Batch-6/Sim_satellite cells
# have a shared SOH-labeling artifact (identical suspicious minimum SOH
# across all 8), excluded from every SOH-based comparison in this
# project, not newly introduced here.
XJTU_EXCLUDED_PREFIX = "Batch-6/Sim_satellite"


def xjtu_cell_ids_soh_valid():
    return [c for c in xjtu_cell_ids() if not c.startswith(XJTU_EXCLUDED_PREFIX)]


def load_pool_train_test():
    """Returns (train, test): each {battery_id: (X, soh, rul)}, X shape
    (n_cycles, 200, 6) float32, for the canonical 42-battery pool split
    (data/processed/battery_split.json - the SAME fixed split as every
    other Stage 6/7 item)."""
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids, test_ids = set(split["train_ids"]), set(split["test_ids"])
    full = load_all_battery_tensors_stage4()
    train = {bid: (X.astype(np.float32), soh, rul) for bid, (X, soh, rul, ds) in full.items() if bid in train_ids}
    test = {bid: (X.astype(np.float32), soh, rul) for bid, (X, soh, rul, ds) in full.items() if bid in test_ids}
    return train, test


def load_heldout(name: str):
    """Returns {battery_id: (X, soh, rul)} for one of the 4 zero-retrain
    held-out datasets, built via the SAME build_dataset_tensors path
    used for the training pool - no retraining, no fitting on this
    data, purely a forward pass at eval time."""
    if name == "CALCE":
        ids, iterate_fn = CALCE_CELLS, iterate_calce_cycles
    elif name == "Oxford":
        ids, iterate_fn = oxford_cell_ids(), iterate_oxford_cycles
    elif name == "HUST":
        ids, iterate_fn = hust_cell_ids(), iterate_hust_cycles
    elif name == "XJTU":
        ids, iterate_fn = xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles
    else:
        raise ValueError(name)

    out = {}
    for cid in ids:
        cycles = list(iterate_fn(cid))
        if len(cycles) < 5:
            continue
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        out[cid] = (X.astype(np.float32), soh.astype(np.float32), rul.astype(np.float32))
    return out


def load_pool_train_test_keyed():
    """Like load_pool_train_test, but also returns (dataset, battery_id,
    cycle_idx) aligned 1:1 with each row of X - needed ONLY by 7.3's
    DeepONet-embedding merge into the existing HI+fusion feature tables
    (7.1/7.2 don't need row-level keys, hence this separate function
    rather than changing load_pool_train_test's contract). Mirrors
    stage4_pool.load_all_battery_tensors_stage4's own loop exactly
    (same cycle sources, same SOH/RUL convention), the one thing added
    is tracking cycle_idx per kept row, which that function computes
    internally but does not return."""
    import json as _json
    from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
    from rul_labels import compute_eol_and_rul_severson_aware, soh_per_cycle
    from stage4_recovered_batteries import get_recovered_battery, ALL_RECOVERED
    from sequence_features import get_cycle_tensor

    def _tensors_keyed(cycles, soh_map, rul_map):
        Xs, sohs, ruls, idxs = [], [], [], []
        for c in cycles:
            x = get_cycle_tensor(c, n_bins=200)
            if x is None:
                continue
            Xs.append(x); sohs.append(soh_map[c["cycle_idx"]]); ruls.append(rul_map[c["cycle_idx"]])
            idxs.append(c["cycle_idx"])
        if not Xs:
            return None, None, None, None
        return np.stack(Xs).astype(np.float32), np.array(sohs, dtype=np.float32), np.array(ruls, dtype=np.float32), idxs

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids, test_ids = set(split["train_ids"]), set(split["test_ids"])
    out = {}

    for cid in ["B0005", "B0006", "B0007", "B0018"]:
        cycles = list(iterate_nasa_cycles(cid))
        _, _, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=cid)
        soh_map = soh_per_cycle(cycles)
        X, soh, rul, idxs = _tensors_keyed(cycles, soh_map, rul_map)
        if X is not None:
            out[cid] = (X, soh, rul, idxs, "NASA")

    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = _json.load(f)
    for entry in mit_subset:
        gid = entry["global_id"]
        cycles = list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))
        _, _, rul_map = compute_eol_and_rul_severson_aware(cycles, global_id=gid)
        soh_map = soh_per_cycle(cycles)
        X, soh, rul, idxs = _tensors_keyed(cycles, soh_map, rul_map)
        if X is not None:
            out[gid] = (X, soh, rul, idxs, "MIT")

    for ds, bid in ALL_RECOVERED:
        kept_cycles, soh_map, rul_map, eol_cycle, censored = get_recovered_battery(ds, bid)
        X, soh, rul, idxs = _tensors_keyed(kept_cycles, soh_map, rul_map)
        if X is not None:
            out[bid] = (X, soh, rul, idxs, ds)

    train, test = {}, {}
    for bid, (X, soh, rul, idxs, ds) in out.items():
        target = train if bid in train_ids else (test if bid in test_ids else None)
        if target is not None:
            target[bid] = (X, soh, rul, idxs, ds)
    return train, test


def load_heldout_keyed(name: str):
    """Like load_heldout, but also returns cycle_idx per row (needed by
    7.3's embedding merge) - build_dataset_tensors already computes
    idxs internally, load_heldout just doesn't return it; this does."""
    if name == "CALCE":
        ids, iterate_fn = CALCE_CELLS, iterate_calce_cycles
    elif name == "Oxford":
        ids, iterate_fn = oxford_cell_ids(), iterate_oxford_cycles
    elif name == "HUST":
        ids, iterate_fn = hust_cell_ids(), iterate_hust_cycles
    elif name == "XJTU":
        ids, iterate_fn = xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles
    else:
        raise ValueError(name)

    out = {}
    for cid in ids:
        cycles = list(iterate_fn(cid))
        if len(cycles) < 5:
            continue
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        out[cid] = (X.astype(np.float32), soh.astype(np.float32), rul.astype(np.float32), list(idxs))
    return out


def fit_norm_stats(train: dict):
    X_all = np.concatenate([v[0] for v in train.values()], axis=0)
    return compute_channel_norm_stats(X_all, clip_percentile=1.0)


def norm_pool(pool: dict, stats: list[dict]):
    return {bid: (apply_channel_norm(X, stats), soh, rul) for bid, (X, soh, rul) in pool.items()}
