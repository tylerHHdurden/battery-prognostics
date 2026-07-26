"""
Feature-fusion step (additive, does not touch the existing base-learner
pipeline): trains a small CNN encoder on the ICA/DV/DC channels
(dQdV/dVdQ/dIdV - channels 3:6 of the sequence_features tensor) via a
lightweight SOH head, then extracts the pooled embedding for EVERY
NASA+MIT cycle and saves it as a flat feature table.

Reuses train_deep_models.load_all_battery_tensors/make_xy/train_one_model
and the SAME battery_split.json used by every other Phase 2+ script, so
the encoder sees the same fit/val split and its embeddings can be merged
onto hi_table.parquet by (dataset, battery_id, cycle_idx) with no
train/test leakage.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from train_deep_models import load_all_battery_tensors, make_xy, train_one_model
from sequence_features import apply_channel_norm
from models.ica_encoder import ICAEncoder

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

EMBED_DIM = 16
ICA_CHANNEL_SLICE = slice(3, 6)  # dQdV, dVdQ, dIdV


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    all_ids = fit_ids + val_ids + test_ids
    X_all, y_all, rul_all, ds_all, bid_all, cyc_all = make_xy(battery_data, all_ids)

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_all = apply_channel_norm(X_all, norm_stats)

    # rebuild fit/val slices with the SAME normalization (fit stats already
    # computed from these same fit_ids by train_deep_models.py - reused,
    # not recomputed, for consistency with every other script)
    n_fit_cycles = sum(len(battery_data[b][1]) for b in fit_ids)
    n_val_cycles = sum(len(battery_data[b][1]) for b in val_ids)
    X_fit_ica = X_all[:n_fit_cycles, :, ICA_CHANNEL_SLICE]
    y_fit = y_all[:n_fit_cycles]
    X_val_ica = X_all[n_fit_cycles:n_fit_cycles + n_val_cycles, :, ICA_CHANNEL_SLICE]
    y_val = y_all[n_fit_cycles:n_fit_cycles + n_val_cycles]
    print(f"[fusion] fit cycles={len(X_fit_ica)}, val cycles={len(X_val_ica)} "
          f"(loaded in {time.time()-t0:.1f}s)")

    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder, hist = train_one_model(
        "ICAEncoder", encoder, X_fit_ica, y_fit, X_val_ica, y_val,
        epochs=25, patience=6,
    )
    torch.save(encoder.state_dict(), ROOT / "models" / "ica_encoder.pt")
    pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / "ica_encoder_history.csv", index=False)

    # extract embeddings for every cycle (fit+val+test, in the SAME
    # concatenation order as X_all/ds_all/bid_all/cyc_all)
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder.encode(torch.tensor(X_all[:, :, ICA_CHANNEL_SLICE])).numpy()

    out = pd.DataFrame({
        "dataset": ds_all, "battery_id": bid_all, "cycle_idx": cyc_all,
    })
    for i in range(EMBED_DIM):
        out[f"fusion_{i}"] = embeddings[:, i]
    out.to_csv(PROC_DIR / "fusion_embeddings.csv", index=False)

    print(f"[fusion] saved {len(out)} embeddings ({EMBED_DIM}-dim) to "
          f"data/processed/fusion_embeddings.csv")
    print(f"[fusion] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
