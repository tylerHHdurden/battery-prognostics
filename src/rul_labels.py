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
"""

from __future__ import annotations

import numpy as np


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
