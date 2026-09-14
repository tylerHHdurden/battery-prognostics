"""
Stage 4 completeness fix: recovers and persists the RUL joint-fusion
model's HI-feature z-score stats (hi_mean, hi_std - the 8 canonical
_rel-ratio features, NO cycle_idx) to `data/processed/joint_hi_norm_
stats.json`.

Why this exists as a separate script, disclosed rather than silently
folded in: `run_stage4_step2b_xgb_joint.py`'s original run computed
hi_mean/hi_std in-memory to train the joint model, but never persisted
them - a real gap, caught by this project's own end-to-end smoke test
(run_stage4_smoke_test.py) crashing with a shape mismatch when
live_inference.py tried to feed the joint model an un-z-scored,
9-feature (with cycle_idx) vector instead of the z-scored, 8-feature
one it was actually trained on. This script recomputes the SAME stats
identically (same fit_ids, same build_hi_features call) without
retraining anything - a pure recovery of a derived-but-lost artifact.
Kept as a permanent, rerunnable script (not a one-off) since any future
Stage 4-style retrain will need this same recovery step unless
step2b itself is extended to save it directly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_pool import load_all_battery_tensors_stage4
from train_deep_models import make_xy
from stage1_common import canonical_feature_cols, add_reformulated_duration_features
from run_stage1_followup_partB_joint_rul import build_hi_features

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def main():
    battery_data = load_all_battery_tensors_stage4()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    _, _, _, _, bid_fit, cyc_fit = make_xy(battery_data, fit_ids)
    hi_reformulated = add_reformulated_duration_features(pd.read_parquet(PROC_DIR / "hi_table.parquet"))
    joint_feature_cols = canonical_feature_cols(reformulated=True)
    fit_key_df = pd.DataFrame({"battery_id": bid_fit, "cycle_idx": cyc_fit}).merge(
        hi_reformulated[["battery_id", "cycle_idx"] + joint_feature_cols], on=["battery_id", "cycle_idx"], how="left")
    train_medians_hi = fit_key_df[joint_feature_cols].median(numeric_only=True).to_numpy()
    HI_fit = build_hi_features(bid_fit, cyc_fit, hi_reformulated, joint_feature_cols, train_medians_hi)
    hi_mean, hi_std = HI_fit.mean(axis=0), HI_fit.std(axis=0) + 1e-8

    print("joint_feature_cols:", joint_feature_cols)
    print("hi_mean:", hi_mean.tolist())
    print("hi_std:", hi_std.tolist())

    out = {
        "feature_order": joint_feature_cols,
        "hi_mean": hi_mean.tolist(),
        "hi_std": hi_std.tolist(),
        "train_medians": train_medians_hi.tolist(),
    }
    with open(PROC_DIR / "joint_hi_norm_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved data/processed/joint_hi_norm_stats.json")


if __name__ == "__main__":
    main()
