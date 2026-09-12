"""
Dataset Expansion Phase 1: ICAEncoder retrained on the expanded pool.
Additive - train_fusion_encoder.py / ica_encoder.pt / fusion_embeddings.csv
untouched. Mirrors train_fusion_encoder.py exactly, only swapping in the
expanded battery loader / expanded split / expanded channel-norm stats.
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
from train_deep_models import make_xy, train_one_model
from train_deep_models_expanded import load_all_battery_tensors_expanded
from sequence_features import apply_channel_norm
from models.ica_encoder import ICAEncoder

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

EMBED_DIM = 16
ICA_CHANNEL_SLICE = slice(3, 6)


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors_expanded()
    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    all_ids = fit_ids + val_ids + test_ids
    X_all, y_all, rul_all, ds_all, bid_all, cyc_all = make_xy(battery_data, all_ids)

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    X_all = apply_channel_norm(X_all, norm_stats)

    n_fit_cycles = sum(len(battery_data[b][1]) for b in fit_ids)
    n_val_cycles = sum(len(battery_data[b][1]) for b in val_ids)
    X_fit_ica = X_all[:n_fit_cycles, :, ICA_CHANNEL_SLICE]
    y_fit = y_all[:n_fit_cycles]
    X_val_ica = X_all[n_fit_cycles:n_fit_cycles + n_val_cycles, :, ICA_CHANNEL_SLICE]
    y_val = y_all[n_fit_cycles:n_fit_cycles + n_val_cycles]
    print(f"[fusion-exp] fit cycles={len(X_fit_ica)}, val cycles={len(X_val_ica)} "
          f"(loaded in {time.time()-t0:.1f}s)")

    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder, hist = train_one_model(
        "ICAEncoder-exp", encoder, X_fit_ica, y_fit, X_val_ica, y_val,
        epochs=25, patience=6,
    )
    torch.save(encoder.state_dict(), ROOT / "models" / "ica_encoder_expanded.pt")
    pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / "ica_encoder_expanded_history.csv", index=False)

    encoder.eval()
    with torch.no_grad():
        embeddings = encoder.encode(torch.tensor(X_all[:, :, ICA_CHANNEL_SLICE])).numpy()

    out = pd.DataFrame({"dataset": ds_all, "battery_id": bid_all, "cycle_idx": cyc_all})
    for i in range(EMBED_DIM):
        out[f"fusion_{i}"] = embeddings[:, i]
    out.to_csv(PROC_DIR / "fusion_embeddings_expanded.csv", index=False)

    print(f"[fusion-exp] saved {len(out)} embeddings ({EMBED_DIM}-dim)")
    print(f"[fusion-exp] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
