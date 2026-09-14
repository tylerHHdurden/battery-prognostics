"""
CORAL-aligned counterpart of train_fusion_encoder_mmd.py (Stage 3, Item
3.4) - structurally identical pipeline (same ICAEncoder, same
supervised-SOH-loss-on-NASA+MIT-source + additive-domain-alignment-term
recipe, same zero-label-leakage discipline: CALCE's SOH/RUL are never
read, only its raw discharge curves feed the alignment term), with the
ONLY substantive change being the alignment loss itself: Deep CORAL
(coral_loss.py, second-order covariance alignment) instead of session
13's multi-kernel MMD. This isolates the CORAL-vs-MMD comparison to the
alignment mechanism alone - everything else (encoder architecture,
optimizer, early-stopping criterion, train/val/CALCE split, channel
normalization) is unchanged from train_fusion_encoder_mmd.py.

Fully additive: writes a parallel `*_coral` file set
(models/ica_encoder_coral.pt, data/processed/fusion_embeddings_coral.csv)
and does not touch any MMD or original (non-aligned) output file.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from data_adapters import iterate_calce_cycles
from train_deep_models import load_all_battery_tensors, make_xy
from sequence_features import build_dataset_tensors, apply_channel_norm
from models.ica_encoder import ICAEncoder
from coral_loss import coral_loss

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

EMBED_DIM = 16
ICA_CHANNEL_SLICE = slice(3, 6)  # dQdV, dVdQ, dIdV
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]
CORAL_LAMBDA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0  # overridable for a lambda sweep

torch.manual_seed(42)
np.random.seed(42)

# reuse the SAME unlabeled-CALCE-tensor cache the MMD script already built
# (identical build: NASA+MIT-normalized ICA/DV/DC channels, no labels) -
# no need to recompute the ~10min raw-CALCE-parsing step twice.
_CALCE_CACHE = PROC_DIR / "_calce_ica_unlabeled_cache.npy"


def build_calce_ica_tensors(norm_stats):
    if _CALCE_CACHE.exists():
        print(f"[fusion-coral] reusing cached CALCE ICA tensors from {_CALCE_CACHE.name}")
        return np.load(_CALCE_CACHE)
    all_X = []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        all_X.append(X.astype(np.float32))
        print(f"[fusion-coral] CALCE/{cid}: {X.shape[0]} cycles (unlabeled, features only)")
    X_all = np.concatenate(all_X)
    X_all = apply_channel_norm(X_all, norm_stats)
    result = X_all[:, :, ICA_CHANNEL_SLICE]
    np.save(_CALCE_CACHE, result)
    return result


def train_with_coral(encoder, X_src_fit, y_src_fit, X_src_val, y_src_val, X_tgt,
                      epochs=25, batch_size=64, lr=1e-3, patience=6):
    """Same optimizer/standardization/early-stopping contract as
    train_deep_models.train_one_model and train_fusion_encoder_mmd's
    train_with_mmd, extended with an additive CORAL term between
    source-batch and target-batch embedding covariances. Early stopping
    is on plain supervised val_loss (NASA+MIT only), matching both the
    non-aligned and MMD baselines so all three encoders are compared on
    equal footing."""
    device = torch.device("cpu")
    encoder.to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    y_mean, y_std = float(y_src_fit.mean()), float(y_src_fit.std() + 1e-8)
    encoder.y_mean_, encoder.y_std_ = y_mean, y_std

    Xt = torch.tensor(X_src_fit)
    yt = torch.tensor((y_src_fit - y_mean) / y_std).unsqueeze(-1)
    Xv = torch.tensor(X_src_val)
    yv = torch.tensor((y_src_val - y_mean) / y_std).unsqueeze(-1)
    Xtgt = torch.tensor(X_tgt)
    n_tgt = len(Xtgt)

    n = len(Xt)
    best_val = np.inf
    best_state = None
    patience_ctr = 0
    history = []

    for epoch in range(epochs):
        encoder.train()
        perm = torch.randperm(n)
        tgt_perm = torch.randperm(n_tgt)
        epoch_sup_loss, epoch_coral_loss = 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]

            tgt_start = i % n_tgt
            tgt_idx = tgt_perm[torch.arange(tgt_start, tgt_start + len(idx)) % n_tgt]
            xb_tgt = Xtgt[tgt_idx]

            opt.zero_grad()
            emb_src = encoder.encode(xb)
            pred = encoder.head(emb_src)
            sup_loss = loss_fn(pred, yb)

            emb_tgt = encoder.encode(xb_tgt)
            coral = coral_loss(emb_src, emb_tgt)

            loss = sup_loss + CORAL_LAMBDA * coral
            loss.backward()
            opt.step()

            epoch_sup_loss += sup_loss.item() * len(idx)
            epoch_coral_loss += coral.item() * len(idx)
        epoch_sup_loss /= n
        epoch_coral_loss /= n

        encoder.eval()
        with torch.no_grad():
            val_pred = encoder.head(encoder.encode(Xv))
            val_loss = loss_fn(val_pred, yv).item()
        history.append({
            "epoch": epoch, "train_sup_loss": epoch_sup_loss,
            "train_coral_loss": epoch_coral_loss, "val_loss": val_loss,
        })
        print(f"[fusion-coral] epoch {epoch:3d} sup_mse={epoch_sup_loss:.4f} "
              f"coral={epoch_coral_loss:.5f} val_mse={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[fusion-coral] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    return encoder, history


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

    n_fit_cycles = sum(len(battery_data[b][1]) for b in fit_ids)
    n_val_cycles = sum(len(battery_data[b][1]) for b in val_ids)
    X_fit_ica = X_all[:n_fit_cycles, :, ICA_CHANNEL_SLICE]
    y_fit = y_all[:n_fit_cycles]
    X_val_ica = X_all[n_fit_cycles:n_fit_cycles + n_val_cycles, :, ICA_CHANNEL_SLICE]
    y_val = y_all[n_fit_cycles:n_fit_cycles + n_val_cycles]
    print(f"[fusion-coral] source fit cycles={len(X_fit_ica)}, source val cycles={len(X_val_ica)}")

    print("[fusion-coral] === building UNLABELED CALCE tensors for CORAL target covariance ===")
    X_tgt_ica = build_calce_ica_tensors(norm_stats)
    print(f"[fusion-coral] target (CALCE, unlabeled) cycles={len(X_tgt_ica)} "
          f"(loaded in {time.time()-t0:.1f}s)")

    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder, hist = train_with_coral(
        encoder, X_fit_ica, y_fit, X_val_ica, y_val, X_tgt_ica,
        epochs=25, patience=6,
    )
    torch.save(encoder.state_dict(), ROOT / "models" / "ica_encoder_coral.pt")
    pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / "ica_encoder_coral_history.csv", index=False)

    encoder.eval()
    with torch.no_grad():
        embeddings = encoder.encode(torch.tensor(X_all[:, :, ICA_CHANNEL_SLICE])).numpy()

    out = pd.DataFrame({
        "dataset": ds_all, "battery_id": bid_all, "cycle_idx": cyc_all,
    })
    for i in range(EMBED_DIM):
        out[f"fusion_{i}"] = embeddings[:, i]
    out.to_csv(PROC_DIR / "fusion_embeddings_coral.csv", index=False)

    print(f"[fusion-coral] saved {len(out)} embeddings ({EMBED_DIM}-dim) to "
          f"data/processed/fusion_embeddings_coral.csv")
    print(f"[fusion-coral] CORAL_LAMBDA={CORAL_LAMBDA} (single fixed weight, not tuned)")
    print(f"[fusion-coral] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
