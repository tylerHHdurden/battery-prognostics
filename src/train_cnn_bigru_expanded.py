"""
Dataset Expansion Phase 1: CNN-BiGRU (5th base learner) retrained on the
expanded pool. Additive - train_cnn_bigru.py / cnn_bigru_soh.pt /
cnn_bigru_metrics.csv untouched. Same architecture/hyperparameters as
the original, reusing train_deep_models.train_one_model unchanged.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
import torch.nn as nn
from train_deep_models import make_xy
from train_deep_models_expanded import load_all_battery_tensors_expanded
from sequence_features import apply_channel_norm
from models.cnn_bigru import CNNBiGRU

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
(PROC_DIR / "predictions").mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

_EPOCH_CKPT_PATH = PROC_DIR / "_cnn_bigru_expanded_epoch_checkpoint.pt"


def train_one_model_resumable(name, model, X_train, y_train, X_val, y_val,
                               epochs=40, batch_size=64, lr=1e-3, patience=8):
    """Same training logic as train_deep_models.train_one_model
    (untouched, still used everywhere else) - copied here rather than
    modified in place, plus per-EPOCH checkpointing to disk. Added after
    this exact training run was killed TWICE by an external process
    teardown (not a code bug) mid-training with zero progress saved,
    each time forcing a full restart from epoch 0. This version saves
    model/optimizer/RNG state after every epoch, so an interrupted run
    resumes from its last completed epoch instead of losing everything -
    directly satisfies the "checkpoint before anything long-running"
    standard used throughout this session, applied here because CNN-
    BiGRU is the one remaining slow stage without it. Bit-for-bit
    reproducible on resume: torch's RNG state (which torch.randperm
    inside the loop depends on) is saved/restored too, not just the
    model weights."""
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if _EPOCH_CKPT_PATH.exists():
        ckpt = torch.load(_EPOCH_CKPT_PATH, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
        best_val, best_state, patience_ctr = ckpt["best_val"], ckpt["best_state"], ckpt["patience_ctr"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[deep-exp/{name}] RESUMING from epoch {start_epoch} "
              f"(checkpoint found: {_EPOCH_CKPT_PATH.name})")
    else:
        y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)
        best_val, best_state, patience_ctr, history, start_epoch = np.inf, None, 0, [], 0

    model.y_mean_, model.y_std_ = y_mean, y_std
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
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = loss_fn(val_pred, yv).item()
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        print(f"[deep-exp/{name}] epoch {epoch:3d} train_mse={epoch_loss:.4f} val_mse={val_loss:.4f}")

        stop = False
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[deep-exp/{name}] early stopping at epoch {epoch}")
                stop = True

        torch.save({
            "model_state": model.state_dict(), "opt_state": opt.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "y_mean": y_mean, "y_std": y_std,
            "best_val": best_val, "best_state": best_state, "patience_ctr": patience_ctr,
            "history": history, "epoch": epoch,
        }, _EPOCH_CKPT_PATH)

        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _EPOCH_CKPT_PATH.unlink(missing_ok=True)  # training genuinely finished, checkpoint no longer needed
    return model, history


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors_expanded()
    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[cnn-bigru-exp] fit batteries: {len(fit_ids)}, val batteries: {len(val_ids)}, "
          f"test batteries: {len(test_ids)}")

    X_fit, y_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, y_val, rul_val, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, rul_test, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[cnn-bigru-exp] fit cycles={len(X_fit)}, val cycles={len(X_val)}, test cycles={len(X_test)}")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    def predict(model, X, chunk_size: int = 4096):
        # Chunked defensively, matching the fix in
        # train_deep_models_expanded.py after PiFormer's unbatched
        # 123,755-row forward pass there tried to allocate ~79GB.
        # CNN-BiGRU's GRU has no attention score matrix (O(batch) memory,
        # not O(batch x seq^2)), so it likely never would have hit that
        # specific failure - but chunking is free insurance and produces
        # numerically identical output (no cross-row dependency).
        # .eval() added defensively too, matching the fix for a SECOND
        # real bug this session found: a resumed/loaded checkpoint stays
        # in .train() mode unless explicitly set, which silently
        # corrupted CNN-LSTM's BatchNorm-dependent predictions
        # elsewhere. CNN-BiGRU has no BatchNorm/LayerNorm (checked -
        # models/cnn_bigru.py has no normalization layer at all), so this
        # is a no-op here today, not a fix for an active bug - kept for
        # consistency and to guard against a future architecture change.
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), chunk_size):
                batch = torch.tensor(X[i:i + chunk_size])
                outs.append(model(batch).squeeze(-1).numpy())
        raw = np.concatenate(outs)
        return raw * model.y_std_ + model.y_mean_

    cnn_bigru = CNNBiGRU()
    cnn_bigru, hist = train_one_model_resumable("CNNBiGRU-exp", cnn_bigru, X_fit, y_fit, X_val, y_val)
    pred_test = predict(cnn_bigru, X_test)
    torch.save(cnn_bigru.state_dict(), ROOT / "models" / "cnn_bigru_soh_expanded.pt")
    pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / "cnn_bigru_expanded_history.csv", index=False)

    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    print(f"[cnn-bigru-exp] TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    print("[cnn-bigru-exp] (original 32-battery CNNBiGRU was RMSE=3.165 MAE=2.341 R2=0.572)")

    out = pd.DataFrame({
        "dataset": ds_test, "battery_id": bid_test, "cycle_idx": cyc_test,
        "SOH": y_test, "RUL": rul_test, "y_pred_CNNBiGRU": pred_test,
    })
    out.to_csv(PROC_DIR / "predictions" / "cnn_bigru_expanded_test_preds.csv", index=False)
    pd.DataFrame([{"model": "CNNBiGRU-expanded", "rmse": rmse, "mae": mae, "r2": r2}]).to_csv(
        PROC_DIR / "predictions" / "cnn_bigru_expanded_metrics.csv", index=False
    )

    ids_fitval = fit_ids + val_ids
    _, ytr, rultr, dstr, bidtr, cyctr = make_xy(battery_data, ids_fitval)
    Xtr = np.concatenate([X_fit, X_val])
    assert len(Xtr) == len(ytr)
    tr_pred = predict(cnn_bigru, Xtr)
    out_train = pd.DataFrame({
        "dataset": dstr, "battery_id": bidtr, "cycle_idx": cyctr,
        "SOH": ytr, "RUL": rultr, "y_pred_CNNBiGRU": tr_pred,
    })
    out_train.to_csv(PROC_DIR / "predictions" / "cnn_bigru_expanded_train_preds.csv", index=False)

    print(f"[cnn-bigru-exp] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
