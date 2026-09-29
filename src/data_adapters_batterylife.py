"""
Adapter for the BatteryLife dataset's own standardized, processed format
(Tan et al., "BatteryLife: A Comprehensive Dataset and Benchmark for
Battery Life Prediction," arXiv:2502.18807 - verified via WebFetch/
WebSearch before use, GitHub github.com/Ruifeng-Tan/BatteryLife). Source:
Zenodo record 17756951 ("BatteryLife_Processed"), OPEN ACCESS, no login
required - used INSTEAD of the Hugging Face copy
(`Battery-Life/BatteryLife_Processed`) the task named, because that HF
repo is gated (confirmed directly: WebFetch returned "Access requires
account login and acceptance of data sharing conditions" - no credentials
available in this environment). Every file this module reads was
downloaded directly from Zenodo's own file URLs
(`https://zenodo.org/api/records/17756951/files/<Source>.zip/content`),
confirmed via the Zenodo API's own file listing (exact byte sizes
matched on download), not assumed to exist.

FORMAT (directly inspected, not assumed from the dataset's own README):
each battery is one pickle file, a dict with keys including `cell_id`,
`nominal_capacity_in_Ah`, and `cycle_data` - a list of per-cycle dicts,
each with RAW (not pre-split into charge/discharge, unlike this
project's own `data_adapters.py` convention) arrays: `current_in_A`,
`voltage_in_V`, `charge_capacity_in_Ah`, `discharge_capacity_in_Ah`,
`time_in_s`, `temperature_in_C`. Current-sign convention (I>0 charge,
I<0 discharge) CONFIRMED to already match this project's own
established convention (see `ica_dv_dc.py`'s own documented convention,
reused throughout `data_adapters.py`) - no flip needed. This module's
only real job is SEGMENTING each cycle's one combined array into
{"charge": {...}, "discharge": {...}} sub-dicts by current sign, then
handing off to the exact same `charge`/`discharge`/`discharge_capacity`/
`cycle_idx` cycle-record contract every other adapter in this project
produces - so `health_indicators.compute_health_indicators`,
`sequence_features.build_dataset_tensors`, and `rul_labels`'s
`soh_per_cycle`/`compute_eol_and_rul` all work UNCHANGED on this new
source, exactly as they do on NASA/MIT/CALCE/Oxford/HUST/XJTU.

DISCLOSED SIMPLIFICATION: segmentation assumes each cycle has ONE
contiguous charge phase followed by ONE contiguous discharge phase (true
for every cycle spot-checked across UL_PUR/HNEI/SNL/MICH during item 6's
verification) - a cycle with a more complex multi-step protocol (e.g. a
pulse-test cycle) would have its charge/discharge phases identified as
the FULL min-to-max span of positive-current and negative-current
indices respectively, which would silently include any intermediate
rest/pulse segments within that span. Not observed in this project's own
spot checks, but not exhaustively verified across all 7 sources' full
cycle counts either - disclosed, not silently assumed safe everywhere.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BATTERYLIFE_DIR = ROOT / "data" / "raw" / "batterylife"

# Every sub-source this module has actually been used against (directory
# name under data/raw/batterylife/, matching the Zenodo zip's own name).
KNOWN_SOURCES = ["UL_PUR", "HNEI", "SNL", "MICH", "MICH_EXP", "RWTH", "Stanford", "Stanford_2", "ISU_ILCC"]

CURRENT_EPS_A = 1e-4  # below this magnitude, a sample counts as neither charge nor discharge


def batterylife_cell_ids(source: str) -> list[str]:
    d = BATTERYLIFE_DIR / source
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.pkl"))


def _segment(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (charge_idx, discharge_idx) - the full min..max index span
    of positive- and negative-current samples respectively (see module
    docstring's disclosed simplification)."""
    charge_mask = I > CURRENT_EPS_A
    discharge_mask = I < -CURRENT_EPS_A
    charge_idx = np.where(charge_mask)[0]
    discharge_idx = np.where(discharge_mask)[0]
    return charge_idx, discharge_idx


def iterate_batterylife_cycles(source: str, cell_id: str):
    path = BATTERYLIFE_DIR / source / f"{cell_id}.pkl"
    with open(path, "rb") as f:
        raw = pickle.load(f)

    for cyc in raw["cycle_data"]:
        I = np.asarray(cyc["current_in_A"], dtype=float)
        if len(I) < 8:
            continue
        V = np.asarray(cyc["voltage_in_V"], dtype=float)
        t = np.asarray(cyc["time_in_s"], dtype=float)
        T_raw = cyc.get("temperature_in_C")
        T = np.asarray(T_raw, dtype=float) if T_raw is not None else np.zeros_like(I)
        dcap = np.asarray(cyc["discharge_capacity_in_Ah"], dtype=float)

        charge_idx, discharge_idx = _segment(I)
        if len(charge_idx) < 3 or len(discharge_idx) < 3:
            continue
        c0, c1 = charge_idx.min(), charge_idx.max() + 1
        d0, d1 = discharge_idx.min(), discharge_idx.max() + 1

        def _slice(a, b):
            tt = t[a:b]
            return {"t": tt - tt[0] if len(tt) else tt, "V": V[a:b], "I": I[a:b], "T": T[a:b]}

        charge = _slice(c0, c1)
        discharge = _slice(d0, d1)
        if len(charge["t"]) < 3 or len(discharge["t"]) < 3:
            continue

        cap = float(dcap[d0:d1].max() - dcap[d0:d1].min()) if d1 > d0 else np.nan
        if not np.isfinite(cap) or cap <= 0:
            continue

        yield {
            "cycle_idx": int(cyc["cycle_number"]),
            "discharge_capacity": cap,
            "charge": charge,
            "discharge": discharge,
        }
