"""
Toolkit Phase 1(b): integrate Tongji from the BatteryLife Zenodo record
(same record, 17756951, as the 9 already-integrated sources) using the
SAME adapter (`data_adapters_batterylife.py`'s `build_source`,
unchanged) and the SAME build pipeline `build_batterylife_hi_table.py`
already established - not a new pipeline, just one more source name.

Also checks whether Tongji records a usable post-charge rest period
(relevant to this project's own item 4 - relaxation-voltage features -
which stopped at a feasibility gate for the ORIGINAL 13-dataset panel;
Tongji is new, checked fresh here, not assumed to match any other
source's own verdict).
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from models.ica_encoder import ICAEncoder
from build_batterylife_hi_table import build_source
from stage1_common import add_reformulated_duration_features, ROOT, PROC_DIR
from stage5_extended_reformulation import add_scv_matd_viect_reformulated
from data_adapters_batterylife import batterylife_cell_ids, iterate_batterylife_cycles, BATTERYLIFE_DIR

ZIP_PATH = ROOT / "data" / "raw" / "batterylife" / "_downloads" / "Tongji.zip"
EXTRACT_DIR = BATTERYLIFE_DIR / "Tongji"


def extract():
    if EXTRACT_DIR.exists() and batterylife_cell_ids("Tongji"):
        print(f"[tongji] already extracted: {len(batterylife_cell_ids('Tongji'))} files present")
        return
    print(f"[tongji] extracting {ZIP_PATH} ({ZIP_PATH.stat().st_size/1e6:.1f} MB)...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        print(f"[tongji] zip contains {len(names)} entries, first 5: {names[:5]}")
        pkl_names = [n for n in names if n.lower().endswith(".pkl")]
        print(f"[tongji] {len(pkl_names)} .pkl files found in zip")
        for n in pkl_names:
            target = EXTRACT_DIR / Path(n).name
            with zf.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())
    print(f"[tongji] extracted {len(batterylife_cell_ids('Tongji'))} battery files to {EXTRACT_DIR}")


def check_rest_period():
    print("\n[tongji] === rest-period feasibility check (item 4's own methodology, applied fresh) ===")
    ids = batterylife_cell_ids("Tongji")
    if not ids:
        print("[tongji] no battery files - cannot check")
        return
    gaps = []
    n_points_after_charge = []
    for cid in ids[:5]:
        cycles = list(iterate_batterylife_cycles("Tongji", cid))
        for c in cycles[:20]:
            charge_t = c["charge"]["t"]
            discharge_t = c["discharge"]["t"]
            if len(charge_t) == 0 or len(discharge_t) == 0:
                continue
            gap = float(discharge_t[0] - charge_t[-1])
            gaps.append(gap)
    if gaps:
        gaps = np.array(gaps)
        print(f"[tongji] inter-phase (end-of-charge -> start-of-discharge) time gap: "
              f"median={np.median(gaps):.1f}s, mean={gaps.mean():.1f}s, n={len(gaps)} cycles sampled "
              f"(across {min(5, len(ids))} batteries, first 20 cycles each)")
        verdict = "YES - median gap suggests a real rest period" if np.median(gaps) > 60 else \
                  "NO/NEGLIGIBLE - gap is just ordinary sampling granularity"
        print(f"[tongji] verdict: {verdict}")
    else:
        print("[tongji] no charge->discharge gaps could be measured")


def main():
    extract()
    check_rest_period()

    print("\n[tongji] === building HI+fusion table (reusing build_batterylife_hi_table.build_source unchanged) ===")
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()

    merged = build_source("Tongji", encoder)
    if merged is None:
        print("[tongji] FAILED to build any usable rows - stopping")
        return

    merged = add_reformulated_duration_features(merged)
    merged = add_scv_matd_viect_reformulated(merged)
    out_path = PROC_DIR / "batterylife_tongji_merged.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[tongji] saved {out_path} ({len(merged)} rows, {merged['battery_id'].nunique()} batteries)")


if __name__ == "__main__":
    main()
