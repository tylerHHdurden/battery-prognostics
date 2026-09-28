"""
Dataset-aware routing, step 3: precomputes and saves the TRAIN-pool,
post-reformulation column medians for Stage 5's extended feature set
(data/processed/ext_reformulation_medians.json) - the same statistic
fit_xgb used when models/_experimental_xgb_soh_fusion_extended_
reformulation.json was trained (verified exactly, see src/verify_
extended_live_feature_parity.py's own compute_ext_medians(), which
this script reuses unchanged).

Saved as a static file rather than recomputed on every app startup
(matching this project's own established precompute_app_constants.py
convention) - loading the full NASA+MIT pool + reformulation is a
multi-second operation, not something a Streamlit session should pay
on every cold start just for a constant that never changes unless the
model itself is retrained.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_extended_live_feature_parity import compute_ext_medians

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def main():
    print("[precompute-ext] computing TRAIN-pool post-reformulation column medians...")
    ext_medians = compute_ext_medians()
    with open(PROC_DIR / "ext_reformulation_medians.json", "w") as f:
        json.dump(ext_medians, f, indent=2)
    print(f"[precompute-ext] saved: {ext_medians}")


if __name__ == "__main__":
    main()
