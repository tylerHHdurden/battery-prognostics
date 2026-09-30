"""
Run any existing script UNMODIFIED, with the BatteryLife per-source parquets' `fusion_*` columns replaced by the
corrected OLD-encoder embeddings (old norm stats applied before the old encoder; see
build_corrected_old_encoder_embeddings_batterylife.py). Usage:

    python src/run_with_corrected_batterylife_embeddings.py src/run_toolkit_phase2c_label_efficient.py [args...]

How: patches pandas.read_parquet so that whenever a `batterylife_<source>_merged.parquet` is read, its fusion_0..15
columns are swapped for the corrected ones (inner-joined on battery_id + cycle_idx; the corrected parquet covers
exactly the rows the original build produced embeddings for - same skipped-battery set). Nothing else changes.

Why this exists: build_batterylife_hi_table.py encoded RAW tensors (no apply_channel_norm), so every stored
BatteryLife fusion column reaches 1e7-1e13 (isu_ilcc 95%, rwth 99.97% of rows > 100). Every script that fed those
columns to a model scored garbage inputs for those sources.
"""
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FC = [f"fusion_{i}" for i in range(16)]
_orig_read_parquet = pd.read_parquet
_corrected = None
_PAT = re.compile(r"batterylife_(?P<src>.+)_merged\.parquet$")


def _load_corrected():
    global _corrected
    if _corrected is None:
        c = _orig_read_parquet(PROC / "old_encoder_embeddings_batterylife_corrected.parquet")
        c["battery_id"] = c["battery_id"].astype(str)
        _corrected = c
    return _corrected


def patched_read_parquet(path, *args, **kwargs):
    df = _orig_read_parquet(path, *args, **kwargs)
    m = _PAT.search(str(path).replace("\\", "/").split("/")[-1])
    if not m or not all(c in df.columns for c in FC):
        return df
    src = m.group("src")
    corr = _load_corrected()
    corr = corr[corr["source"] == src][["battery_id", "cycle_idx"] + FC]
    hi = df.drop(columns=FC).copy()
    hi["_bid"] = hi["battery_id"].astype(str)
    out = pd.merge(hi, corr.rename(columns={"battery_id": "_bid"}), left_on=["_bid", "cycle_idx"],
                   right_on=["_bid", "cycle_idx"], how="inner").drop(columns=["_bid"])
    assert np.nanmax(np.abs(out[FC].to_numpy(float))) < 30, f"{src}: corrected embeddings should be bounded"
    print(f"[corrected-embeddings] {src}: {len(df)} parquet rows -> {len(out)} rows with corrected old-encoder "
          f"embeddings (max |fusion| {np.nanmax(np.abs(out[FC].to_numpy(float))):.2f})", file=sys.stderr, flush=True)
    return out[list(df.columns)]


pd.read_parquet = patched_read_parquet

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name="__main__")
