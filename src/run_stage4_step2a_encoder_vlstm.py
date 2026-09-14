"""
Stage 4, Step 2a: refit channel_norm_stats.json on the new (32+recovered)
pool, retrain the ICA fusion encoder + VLSTM on it, regenerate
fusion_embeddings.csv for the full new pool.

VLSTM is retrained here even though it is NOT part of the lean SOH
prediction path (that's XGBoost-fusion alone, per session 20/Stage 3.3) -
it stays loaded in the deployed app for its SHAP voltage-region
explainability feature (a UI capability, not touched/removed here), and
keeping it trained on the SAME pool as everything else avoids a
confusing state where explanations come from a battery pool that no
longer matches what predictions are based on. CNN-LSTM/PiFormer are
NOT retrained here - lean deployment drops them from the live pipeline
entirely (they only ever fed the Ridge meta-learner, which lean
deployment also drops).

Checkpointed per-epoch (both trainings) - real risk acknowledged: this
is the first time channel_norm_stats has been refit on a pool including
Stage 2.1's recovered batteries, whose corrected SOH/filtered-cycle
data has never been run through this pipeline before.
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
from stage4_pool import load_all_battery_tensors_stage4
from train_deep_models import make_xy
from sequence_features import compute_channel_norm_stats, apply_channel_norm, CHANNEL_NAMES
from models.ica_encoder import ICAEncoder
from models.vlstm import VLSTM

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
EMBED_DIM = 16
ICA_CHANNEL_SLICE = slice(3, 6)
_CKPT_ENCODER = PROC_DIR / "_stage4_ckpt_encoder.pt"
_CKPT_VLSTM = PROC_DIR / "_stage4_ckpt_vlstm.pt"


def train_resumable(name, model, X_train, y_train, X_val, y_val, ckpt_path,
                     epochs=40, batch_size=64, lr=1e-3, patience=8):
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
        best_val, best_state, patience_ctr = ckpt["best_val"], ckpt["best_state"], ckpt["patience_ctr"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[stage4-2a/{name}] RESUMING from epoch {start_epoch}")
    else:
        y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)
        best_val, best_state, patience_ctr, history, start_epoch = np.inf, None, 0, [], 0

    Xt = torch.tensor(X_train)
    yt = torch.tensor((y_train - y_mean) / y_std).unsqueeze(-1)
    Xv = torch.tensor(X_val)
    yv = torch.tensor((y_val - y_mean) / y_std).unsqueeze(-1)
    n = len(Xt)

    for epoch in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        print(f"[stage4-2a/{name}] epoch {epoch:3d} train_mse={epoch_loss:.4f} val_mse={val_loss:.4f}")

        stop = False
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[stage4-2a/{name}] early stopping at epoch {epoch}")
                stop = True

        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                    "torch_rng_state": torch.get_rng_state(), "y_mean": y_mean, "y_std": y_std,
                    "best_val": best_val, "best_state": best_state, "patience_ctr": patience_ctr,
                    "history": history, "epoch": epoch}, ckpt_path)
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path.unlink(missing_ok=True)
    model.y_mean_, model.y_std_ = y_mean, y_std
    return model, history


def backup_if_needed(rel_path):
    backup_dir = PROC_DIR / "_pre_stage4_backup"
    backup_dir.mkdir(exist_ok=True)
    src = ROOT / rel_path
    if not src.exists():
        return
    dst = backup_dir / Path(rel_path).name
    if not dst.exists():
        import shutil
        shutil.copy2(src, dst)
        print(f"[stage4-2a] backed up {rel_path} -> _pre_stage4_backup/{Path(rel_path).name}")


def main():
    t0 = time.time()
    for rel in ["data/processed/channel_norm_stats.json", "models/ica_encoder.pt",
                "models/vlstm_soh.pt", "data/processed/fusion_embeddings.csv"]:
        backup_if_needed(rel)

    print("[stage4-2a] loading full Stage 4 pool (32 original + recovered)...")
    battery_data = load_all_battery_tensors_stage4()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[stage4-2a] fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)} batteries "
          f"({len(battery_data)} total loaded, {time.time()-t0:.1f}s)")

    all_ids = fit_ids + val_ids + test_ids
    X_all, y_all, rul_all, ds_all, bid_all, cyc_all = make_xy(battery_data, all_ids)

    n_fit_cycles = sum(len(battery_data[b][1]) for b in fit_ids)
    n_val_cycles = sum(len(battery_data[b][1]) for b in val_ids)
    X_fit_raw = X_all[:n_fit_cycles]

    print("\n[stage4-2a] === refitting channel_norm_stats.json on the NEW pool's fit split ===")
    norm_stats = compute_channel_norm_stats(X_fit_raw)
    for i, s in enumerate(norm_stats):
        print(f"[stage4-2a] channel {i} ({CHANNEL_NAMES[i]}): clip=[{s['lo']:.4g},{s['hi']:.4g}] "
              f"mean={s['mean']:.4g} std={s['std']:.4g}")
    with open(PROC_DIR / "channel_norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print("[stage4-2a] saved data/processed/channel_norm_stats.json (OVERWRITES the prior 32-battery-only stats)")

    X_all_norm = apply_channel_norm(X_all, norm_stats)
    X_fit_ica = X_all_norm[:n_fit_cycles, :, ICA_CHANNEL_SLICE]
    X_val_ica = X_all_norm[n_fit_cycles:n_fit_cycles + n_val_cycles, :, ICA_CHANNEL_SLICE]
    y_fit = y_all[:n_fit_cycles]
    y_val = y_all[n_fit_cycles:n_fit_cycles + n_val_cycles]
    X_fit_v = X_all_norm[:n_fit_cycles, :, 0:1]
    X_val_v = X_all_norm[n_fit_cycles:n_fit_cycles + n_val_cycles, :, 0:1]

    print("\n[stage4-2a] === retraining ICA fusion encoder ===")
    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    encoder, hist_enc = train_resumable("encoder", encoder, X_fit_ica, y_fit, X_val_ica, y_val,
                                         _CKPT_ENCODER, epochs=25, patience=6)
    torch.save(encoder.state_dict(), ROOT / "models" / "ica_encoder.pt")
    pd.DataFrame(hist_enc).to_csv(PROC_DIR / "predictions" / "ica_encoder_history_stage4.csv", index=False)
    print("[stage4-2a] saved models/ica_encoder.pt (OVERWRITES the prior 32-battery-only encoder)")

    print("\n[stage4-2a] === retraining VLSTM (kept for SHAP explainability role only) ===")
    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm, hist_vlstm = train_resumable("vlstm", vlstm, X_fit_v, y_fit, X_val_v, y_val,
                                         _CKPT_VLSTM, epochs=40, patience=8)
    torch.save(vlstm.state_dict(), ROOT / "models" / "vlstm_soh.pt")
    pd.DataFrame(hist_vlstm).to_csv(PROC_DIR / "predictions" / "vlstm_history_stage4.csv", index=False)
    print("[stage4-2a] saved models/vlstm_soh.pt (OVERWRITES the prior 32-battery-only VLSTM)")

    print("\n[stage4-2a] === regenerating fusion_embeddings.csv for the full new pool ===")
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder.encode(torch.tensor(X_all_norm[:, :, ICA_CHANNEL_SLICE])).numpy()
    out = pd.DataFrame({"dataset": ds_all, "battery_id": bid_all, "cycle_idx": cyc_all})
    for i in range(EMBED_DIM):
        out[f"fusion_{i}"] = embeddings[:, i]
    out.to_csv(PROC_DIR / "fusion_embeddings.csv", index=False)
    print(f"[stage4-2a] saved {len(out)} embeddings to data/processed/fusion_embeddings.csv "
          f"(OVERWRITES the prior 32-battery-only file)")

    print(f"\n[stage4-2a] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
