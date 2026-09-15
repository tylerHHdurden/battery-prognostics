"""
RUL (Remaining Useful Life) labeling: cycles until discharge capacity first
falls to 80% of the battery's initial capacity (the standard EOL
definition used across NASA/CALCE/MIT prognostics literature).

ASSUMPTION: "initial capacity" = the discharge capacity of the first
logged cycle for that battery (not the manufacturer's nameplate rating,
which isn't available for all three datasets consistently). This matches
how each dataset's own literature defines SOH for these specific cells.

If a battery's capacity never drops to 80% within the logged cycles (some
MIT cells / NASA B0018 truncate early), EOL_cycle is set to the last
logged cycle + 1 and this is flagged with `censored=True` — a censored
(right-truncated) RUL label, not a true observed failure. Downstream code
must not treat censored EOL cycles as ground truth without care; they are
kept (rather than dropped) because dropping them would bias the training
set toward only fast-fading cells.

MIT batch-1/3 CELLS — SEVERSON-AWARE OVERRIDE (added per the project's
own session-43 finding, formally adopted per session-45's recommendation
follow-through): `compute_eol_and_rul` above returns censored=True for
ALL 92 MIT batch-1/3 cells, because Severson et al.'s own released data
for these specific cells is truncated at (one cycle short of) their own
computed cycle_life — the capacity trace, as released, never reaches an
80%-crossing under ANY threshold formula, this project's own or a
reimplementation of Severson's. This is a data-availability constraint,
not a threshold-definition disagreement `compute_eol_and_rul` could ever
resolve by choosing a different fraction. `compute_eol_and_rul_severson_aware`
below is an ADDITIVE wrapper — `compute_eol_and_rul` itself is UNCHANGED
and remains the function every other caller (NASA, CALCE, MIT batches 2/4)
continues to use unmodified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_PROC_DIR = _ROOT / "data" / "processed"
_SEVERSON_CYCLE_LIFE_OFFSET = 2  # see module docstring in run_eol_convention_reconciliation.py
                                  # for the fully-verified mechanism (session 45): Severson's own
                                  # cycle_life = (this cell's raw stored cycle count) + 1 [confirmed
                                  # exactly on 3 independently-checked cells across 2 batch files],
                                  # and this project's own loader separately subtracts 1 more (from
                                  # skipping the raw array's index-0 low-rate diagnostic cycle) -
                                  # together giving the observed +2 offset, deterministically, not
                                  # from a dropped/invalid final cycle as originally hypothesized
                                  # (checked directly: the actual last few raw-stored cycles for the
                                  # verified cells are all valid, ordinary discharge cycles).

_severson_cycle_life_cache: dict[str, float] | None = None


def _load_severson_cycle_life() -> dict[str, float]:
    """Lazily loads (and caches) Severson et al.'s own precomputed
    cycle_life field directly from the raw MIT HDF5 files, keyed by
    global_id (e.g. "b1c20"). Returns {} if the raw files or the
    mit_full_cells.json manifest aren't available (e.g. running without
    the raw dataset present) - callers must treat a missing entry as
    "no override available", not an error."""
    global _severson_cycle_life_cache
    if _severson_cycle_life_cache is not None:
        return _severson_cycle_life_cache

    cache: dict[str, float] = {}
    try:
        import h5py
        from data_adapters import MIT_DIR

        manifest_path = _PROC_DIR / "mit_full_cells.json"
        if not manifest_path.exists():
            manifest_path = _PROC_DIR / "mit_subset.json"
        with open(manifest_path) as f:
            mit_cells = json.load(f)

        for bf in sorted(set(e["batch_file"] for e in mit_cells)):
            path = MIT_DIR / bf
            if not path.exists():
                continue
            with h5py.File(path, "r") as f:
                batch = f["batch"]
                n = batch["summary"].shape[0]
                cl_by_index = {i: float(f[batch["cycle_life"][i, 0]][:][0][0]) for i in range(n)}
            for e in mit_cells:
                if e["batch_file"] == bf and e["cell_index"] in cl_by_index:
                    cache[e["global_id"]] = cl_by_index[e["cell_index"]]
    except Exception:
        cache = {}  # no raw data available locally - override simply won't fire, not an error

    _severson_cycle_life_cache = cache
    return cache


def compute_eol_and_rul_severson_aware(cycle_records: list[dict], global_id: str | None = None,
                                        eol_fraction: float = 0.8):
    """
    Drop-in replacement for compute_eol_and_rul: identical behavior for
    everything except MIT batch-1/3 cells (global_id starting "b1c" or
    "b3c") that have a published Severson cycle_life available - for
    those specifically, EOL/RUL are computed from Severson's own
    published value (adjusted by the verified +2 offset) instead of this
    project's own threshold-crossing rule, since that rule structurally
    cannot succeed on this specific released data (see module docstring).
    Every other battery (NASA, CALCE, MIT batches 2/4, or any MIT
    batch-1/3 cell without a resolvable cycle_life) falls through
    UNCHANGED to compute_eol_and_rul.
    """
    if global_id is not None and (global_id.startswith("b1c") or global_id.startswith("b3c")):
        cl_map = _load_severson_cycle_life()
        published_cl = cl_map.get(global_id)
        # BUG FOUND AND FIXED (204-battery pool regeneration - never
        # triggered by the smaller 42-battery pool, which doesn't
        # include either affected cell): `published_cl is not None`
        # only guards a MISSING dict key, not a PRESENT key holding
        # NaN - b3c23/b3c32 (already flagged in Stage 2.2's EOL
        # convention reconciliation as the 2 cells with no resolvable
        # Severson match) have a real entry in cl_map whose value IS
        # NaN, so `int(published_cl)` crashed with "cannot convert
        # float NaN to integer" instead of falling through to the
        # manual convention as the docstring says it should. Fixed by
        # also checking np.isfinite - both cells now correctly fall
        # through to compute_eol_and_rul, matching every other MIT
        # batch-1/3 cell without a resolvable cycle_life.
        if published_cl is not None and np.isfinite(published_cl):
            idxs = np.array([c["cycle_idx"] for c in cycle_records], dtype=int)
            eol_cycle = int(published_cl) - _SEVERSON_CYCLE_LIFE_OFFSET
            rul_per_cycle = {int(i): max(eol_cycle - int(i), 0) for i in idxs}
            return eol_cycle, False, rul_per_cycle

    return compute_eol_and_rul(cycle_records, eol_fraction=eol_fraction)


def compute_eol_and_rul(cycle_records: list[dict], eol_fraction: float = 0.8):
    """
    cycle_records: list of per-cycle dicts (from data_adapters), in order,
    each with "cycle_idx" and "discharge_capacity".

    Returns: (eol_cycle:int, censored:bool, rul_per_cycle: dict[cycle_idx->int])
    """
    caps = np.array([c["discharge_capacity"] for c in cycle_records], dtype=float)
    idxs = np.array([c["cycle_idx"] for c in cycle_records], dtype=int)

    if len(caps) == 0:
        return None, True, {}

    initial_cap = float(np.median(caps[: min(3, len(caps))]))  # robust to first-cycle noise
    threshold = eol_fraction * initial_cap

    below = np.where(caps <= threshold)[0]
    if len(below) > 0:
        eol_cycle = int(idxs[below[0]])
        censored = False
    else:
        eol_cycle = int(idxs[-1]) + 1
        censored = True

    rul_per_cycle = {int(i): max(eol_cycle - int(i), 0) for i in idxs}
    return eol_cycle, censored, rul_per_cycle


def soh_per_cycle(cycle_records: list[dict]) -> dict:
    """SOH(%) = discharge_capacity / initial_capacity * 100, per cycle_idx."""
    caps = np.array([c["discharge_capacity"] for c in cycle_records], dtype=float)
    idxs = [c["cycle_idx"] for c in cycle_records]
    initial_cap = float(np.median(caps[: min(3, len(caps))]))
    return {int(i): float(cap / initial_cap * 100.0) for i, cap in zip(idxs, caps)}
