"""
MMD-aligned variant of train_fusion_encoder.py — addresses Review 1's
finding directly: on CALCE (unseen dataset), R2 collapsed 0.917->0.31
and the conformal interval stayed the SAME width in- and out-of-domain
while true coverage collapsed 95.6%->6.1%. Root hypothesis: the fusion
embedding's distribution is different enough on CALCE that the model is
silently extrapolating, and the interval (calibrated only on NASA+MIT
residuals) has no way to know that.

This script adds an ADDITIVE Maximum Mean Discrepancy (MMD, see
mmd_loss.py) term to the ICAEncoder's training loss: on top of the
original supervised SOH loss (computed on NASA+MIT source data only,
exactly as in train_fusion_encoder.py), each minibatch also pulls the
embedding distribution of a CALCE batch toward the embedding
distribution of the source batch.

Zero-label-leakage discipline (unchanged from the rest of the project):
CALCE's SOH/RUL labels are NEVER read here — only its raw discharge
curves (via data_adapters.iterate_calce_cycles), which feed the same
ICA/DV/DC channel-extraction and channel-normalization (fit on NASA+MIT
only, reused as-is) used everywhere else in this codebase. MMD is
unsupervised by construction: it only compares feature DISTRIBUTIONS,
never labels. This is standard unsupervised domain adaptation, not a
departure from the project's zero-retrain evaluation contract — CALCE's
raw inputs are used to shape the embedding space during training, but
the CALCE evaluation itself (run_calce_zero_retrain_eval_mmd.py) still
does a pure forward pass with no CALCE-specific fitting of its own.

Fully additive: does not touch ica_encoder.pt / fusion_embeddings.csv or
any file written by train_fusion_encoder.py. Writes a parallel
`*_mmd` file set so the original (non-MMD) fusion pipeline stays
intact for comparison.
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
from mmd_loss import multi_kernel_mmd2

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

EMBED_DIM = 16
ICA_CHANNEL_SLICE = slice(3, 6)  # dQdV, dVdQ, dIdV
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]
MMD_LAMBDA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0  # overridable for a lambda sweep

torch.manual_seed(42)
np.random.seed(42)

_CALCE_CACHE = PROC_DIR / "_calce_ica_unlabeled_cache.npy"


def build_calce_ica_tensors(norm_stats):
    """Unlabeled CALCE ICA/DV/DC channels only — SOH/RUL are read off
    iterate_calce_cycles internally by build_dataset_tensors (needed to
    align cycle indices) but discarded immediately below; never used in
    the MMD loss or anywhere else in this script.

    Cached to disk: raw CALCE cycle parsing (~10 min) is the dominant
    cost of this script and is identical across every lambda in a sweep
    (the cache holds NASA+MIT-normalized ICA channels only, no labels -
    no leakage risk in reusing it)."""
    if _CALCE_CACHE.exists():
        print(f"[fusion-mmd] reusing cached CALCE ICA tensors from {_CALCE_CACHE.name}")
        return np.load(_CALCE_CACHE)
    all_X = []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        all_X.append(X.astype(np.float32))
        print(f"[fusion-mmd] CALCE/{cid}: {X.shape[0]} cycles (unlabeled, features only)")
    X_all = np.concatenate(all_X)
    X_all = apply_channel_norm(X_all, norm_stats)
    result = X_all[:, :, ICA_CHANNEL_SLICE]
    np.save(_CALCE_CACHE, result)
    return result


def train_with_mmd(encoder, X_src_fit, y_src_fit, X_src_val, y_src_val, X_tgt,
                    epochs=25, batch_size=64, lr=1e-3, patience=6):
    """Same optimizer/standardization/early-stopping contract as
    train_deep_models.train_one_model, extended with an additive MMD
    term between source-batch and target-batch embeddings. Early
    stopping is on plain supervised val_loss (NASA+MIT only) — the exact
    same model-selection criterion as the non-MMD baseline, so the two
    encoders are compared on equal footing rather than the MMD variant
    getting an easier bar."""
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
        epoch_sup_loss, epoch_mmd_loss = 0.0, 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]

            # sample a same-size CALCE batch (unlabeled), wrapping around
            # if this pass through source is longer than one pass of target
            tgt_start = (i) % n_tgt
            tgt_idx = tgt_perm[torch.arange(tgt_start, tgt_start + len(idx)) % n_tgt]
            xb_tgt = Xtgt[tgt_idx]

            opt.zero_grad()
            emb_src = encoder.encode(xb)
            pred = encoder.head(emb_src)
            sup_loss = loss_fn(pred, yb)

            emb_tgt = encoder.encode(xb_tgt)
            mmd = multi_kernel_mmd2(emb_src, emb_tgt)

            loss = sup_loss + MMD_LAMBDA * mmd
            loss.backward()
            opt.step()

            epoch_sup_loss += sup_loss.item() * len(idx)
            epoch_mmd_loss += mmd.item() * len(idx)
            n_batches += 1
        epoch_sup_loss /= n
        epoch_mmd_loss /= n

        encoder.eval()
        with torch.no_grad():
            val_pred = encoder.head(encoder.encode(Xv))
            val_loss = loss_fn(val_pred, yv).item()
        history.append({
            "epoch": epoch, "train_sup_loss": epoch_sup_loss,
            "train_mmd_loss": epoch_mmd_loss, "val_loss": val_loss,
        })
        print(f"[fusion-mmd] epoch {epoch:3d} sup_mse={epoch_sup_loss:.4f} "
              f"mmd={epoch_mmd_loss:.5f} val_mse={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[fusion-mmd] early stopping at epoch {epoch}")
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
    print(f"[fusion-mmd] source fit cycles={len(X_fit_ica)}, source val cycles={len(X_val_ica)}")

    print("[fusion-mmd] === building UNLABELED CALCE tensors for MMD target distribution ===")
    X_tgt_ica = build_calce_ica_tensors(norm_stats)
    print(f"[fusion-mmd] target (CALCE, unlabeled) cycles={len(X_tgt_ica)} "
          f"(loaded in {time.time()-t0:.1f}s)")

    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder, hist = train_with_mmd(
        encoder, X_fit_ica, y_fit, X_val_ica, y_val, X_tgt_ica,
        epochs=25, patience=6,
    )
    torch.save(encoder.state_dict(), ROOT / "models" / "ica_encoder_mmd.pt")
    pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / "ica_encoder_mmd_history.csv", index=False)

    # extract embeddings for every NASA+MIT cycle, same layout/order as
    # fusion_embeddings.csv, so downstream fusion scripts are drop-in
    # replacements just by pointing at the _mmd file.
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder.encode(torch.tensor(X_all[:, :, ICA_CHANNEL_SLICE])).numpy()

    out = pd.DataFrame({
        "dataset": ds_all, "battery_id": bid_all, "cycle_idx": cyc_all,
    })
    for i in range(EMBED_DIM):
        out[f"fusion_{i}"] = embeddings[:, i]
    out.to_csv(PROC_DIR / "fusion_embeddings_mmd.csv", index=False)

    print(f"[fusion-mmd] saved {len(out)} embeddings ({EMBED_DIM}-dim) to "
          f"data/processed/fusion_embeddings_mmd.csv")
    print(f"[fusion-mmd] MMD_LAMBDA={MMD_LAMBDA} (single fixed weight, not tuned)")
    print(f"[fusion-mmd] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
