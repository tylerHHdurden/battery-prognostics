"""
One-time precomputation so the Streamlit dashboard never has to reload
the full NASA+MIT battery set (a multi-minute operation - see the CALCE
zero-retrain session) just to answer a single prediction request.

Saves:
  - data/processed/destandardization_constants.json: SOH y_mean/y_std
    (shared by VLSTM/CNN-LSTM/PiFormer/XGBoost-fusion/Ridge-fusion, all
    trained on the same NASA+MIT fit split) and RUL y_mean/y_std (from
    the joint-adaptive model's own fit-split RUL values).
  - data/processed/shap_background.npy: a small (20-cycle) normalized
    6-channel background tensor from NASA B0005, for DeepSHAP - reused
    across every dashboard session rather than rebuilt per-request.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_nasa_cycles
from sequence_features import build_dataset_tensors, apply_channel_norm
from train_deep_models import load_all_battery_tensors, make_xy

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def main():
    print("[precompute] loading NASA+MIT battery tensors (one-time cost)...")
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    _, soh_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    soh_mean, soh_std = float(soh_fit.mean()), float(soh_fit.std() + 1e-8)
    rul_mean, rul_std = float(rul_fit.mean()), float(rul_fit.std() + 1e-8)

    constants = {
        "soh_mean": soh_mean, "soh_std": soh_std,
        "rul_mean": rul_mean, "rul_std": rul_std,
        "soh_conformal_half_width": 2.3667296955479173,
        # was 871.417/2 - stale: measured before the log_sigma-clamping
        # retrain of joint_adaptive.pt (see DEVELOPMENT_LOG.md, Follow-up
        # session 11). Re-run src/run_conformal.py and copy its RUL
        # avg_interval_width here if joint_adaptive.pt is ever retrained.
        "rul_conformal_half_width": 828.7103271484375 / 2,
        "fit_battery_ids": fit_ids,
    }
    with open(PROC_DIR / "destandardization_constants.json", "w") as f:
        json.dump(constants, f, indent=2)
    print(f"[precompute] SOH: mean={soh_mean:.3f} std={soh_std:.3f}")
    print(f"[precompute] RUL: mean={rul_mean:.3f} std={rul_std:.3f}")

    # DeepSHAP background: 20 early-life cycles from NASA B0005, normalized
    cycles = list(iterate_nasa_cycles("B0005"))[:20]
    X_bg, *_ = build_dataset_tensors(cycles)
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_bg = apply_channel_norm(X_bg.astype(np.float32), norm_stats)
    np.save(PROC_DIR / "shap_background.npy", X_bg)
    print(f"[precompute] saved SHAP background tensor: {X_bg.shape}")

    print("[precompute] DONE")


if __name__ == "__main__":
    main()
