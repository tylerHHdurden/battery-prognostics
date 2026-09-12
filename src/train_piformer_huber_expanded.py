"""
Targeted improvement pass, Part 1: retrain PiFormer on the expanded
pool with Huber loss instead of MSE - a surgical fix for the single-
battery-outlier RMSE regression (NASA/B0053, see
detect_early_cycle_outliers.py for why B0053 is NOT a data-quality
artifact and exclusion would be unprincipled - Huber loss is tried
FIRST per instruction, exactly because it's the less invasive option
of the two).

Huber loss is quadratic for small residuals (behaves like MSE, keeps
the model's overall accuracy) and LINEAR for large residuals (unlike
MSE, does not let one catastrophically-wrong point dominate the
gradient) - exactly the property needed here: B0053's few worst
predictions (raw MSE-driven PiFormer was off by up to ~74 SOH points on
that battery) were free to dominate the training signal under squared-
error loss; Huber caps their influence without ignoring them outright
(unlike hard exclusion, which would remove them from training
entirely).

Same architecture, same data, same 204-battery split
(battery_split_expanded.json - the UNPINNED split, deliberately, so
this stays a direct, apples-to-apples RMSE/MAE/R2 comparison against
the just-completed Dataset Expansion report's PiFormer numbers, not
confounded by also changing the split at the same time as changing the
loss function), same 40-epoch/patience-8 budget as every other deep
model in this project. Additive: piformer_soh_expanded.pt (the
existing MSE-trained checkpoint) is left completely untouched;
this script writes piformer_soh_huber_expanded.pt.

delta=1.0 (PyTorch's nn.HuberLoss default, in STANDARDIZED target
units, i.e. z-scored SOH) - a deliberate, unexplored-alternative choice
made explicit here rather than silently tuned: this is the textbook
default, not swept/tuned against this specific outlier, since sweeping
delta specifically to fit B0053 would be circular (tuning a
hyperparameter against the exact failure case being fixed defeats the
point of testing whether the standard method generalizes).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from train_deep_models import make_xy
from train_deep_models_expanded import load_all_battery_tensors_expanded
from sequence_features import apply_channel_norm
from models.piformer import PiFormer

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
(PROC_DIR / "predictions").mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

_EPOCH_CKPT_PATH = PROC_DIR / "_piformer_huber_expanded_epoch_checkpoint.pt"


def train_one_model_resumable(name, model, X_train, y_train, X_val, y_val,
                               epochs=40, batch_size=64, lr=1e-3, patience=8):
    """Same per-epoch checkpoint/resume pattern as
    train_cnn_bigru_expanded.py's train_one_model_resumable (verified
    bit-for-bit reproducible there via a synthetic crash test before
    being trusted) - copied here rather than shared, same reasoning as
    that file: a self-contained copy carries zero risk to any other
    script. Only real difference from the MSE version: loss_fn is
    nn.HuberLoss instead of nn.MSELoss, everything else (optimizer,
    standardization, early stopping, RNG-state checkpointing) identical."""
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)

    if _EPOCH_CKPT_PATH.exists():
        ckpt = torch.load(_EPOCH_CKPT_PATH, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
        best_val, best_state, patience_ctr = ckpt["best_val"], ckpt["best_state"], ckpt["patience_ctr"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[piformer-huber] RESUMING from epoch {start_epoch} "
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
        print(f"[piformer-huber] epoch {epoch:3d} train_huber={epoch_loss:.4f} val_huber={val_loss:.4f}")

        stop = False
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[piformer-huber] early stopping at epoch {epoch}")
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
    _EPOCH_CKPT_PATH.unlink(missing_ok=True)
    return model, history


def predict(model, X, chunk_size: int = 4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), chunk_size):
            batch = torch.tensor(X[i:i + chunk_size])
            outs.append(model(batch).squeeze(-1).numpy())
    raw = np.concatenate(outs)
    return raw * model.y_std_ + model.y_mean_


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors_expanded()
    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[piformer-huber] fit batteries: {len(fit_ids)}, val batteries: {len(val_ids)}, "
          f"test batteries: {len(test_ids)}")

    X_fit, y_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, y_val, rul_val, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, rul_test, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[piformer-huber] fit cycles={len(X_fit)}, val cycles={len(X_val)}, test cycles={len(X_test)}")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    piformer = PiFormer()
    piformer, hist = train_one_model_resumable("PiFormer-huber", piformer, X_fit, y_fit, X_val, y_val)
    pred_test = predict(piformer, X_test)
    torch.save(piformer.state_dict(), ROOT / "models" / "piformer_soh_huber_expanded.pt")
    pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / "piformer_huber_expanded_history.csv", index=False)

    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    print(f"[piformer-huber] TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    print("[piformer-huber] (MSE-trained PiFormer-expanded was RMSE=3.3405 MAE=0.7143 R2=0.7795)")
    print("[piformer-huber] (original 32-battery PiFormer was RMSE=2.993 MAE=1.928 R2=0.617)")

    out = pd.DataFrame({
        "dataset": ds_test, "battery_id": bid_test, "cycle_idx": cyc_test,
        "SOH": y_test, "RUL": rul_test, "y_pred_PiFormer_huber": pred_test,
    })
    out.to_csv(PROC_DIR / "predictions" / "piformer_huber_expanded_test_preds.csv", index=False)
    pd.DataFrame([{"model": "PiFormer-huber-expanded", "rmse": rmse, "mae": mae, "r2": r2}]).to_csv(
        PROC_DIR / "predictions" / "piformer_huber_expanded_metrics.csv", index=False
    )

    # Per-battery breakdown, same diagnostic used to root-cause the MSE
    # version's regression in the first place - confirms (or refutes)
    # whether B0053 specifically improved, not just the pooled average.
    per_batt = out.copy()
    per_batt["abs_err"] = (per_batt["SOH"] - per_batt["y_pred_PiFormer_huber"]).abs()
    g = per_batt.groupby(["dataset", "battery_id"]).agg(
        n=("SOH", "size"),
        rmse=("abs_err", lambda e: float(np.sqrt((e ** 2).mean()))),
    ).reset_index().sort_values("rmse", ascending=False)
    print("\n[piformer-huber] Per-battery RMSE, worst 5 (was NASA/B0053=74.7 under MSE):")
    print(g.head(5).to_string(index=False))
    g.to_csv(PROC_DIR / "predictions" / "piformer_huber_expanded_per_battery.csv", index=False)

    ids_fitval = fit_ids + val_ids
    _, ytr, rultr, dstr, bidtr, cyctr = make_xy(battery_data, ids_fitval)
    Xtr = np.concatenate([X_fit, X_val])
    assert len(Xtr) == len(ytr)
    tr_pred = predict(piformer, Xtr)
    out_train = pd.DataFrame({
        "dataset": dstr, "battery_id": bidtr, "cycle_idx": cyctr,
        "SOH": ytr, "RUL": rultr, "y_pred_PiFormer_huber": tr_pred,
    })
    out_train.to_csv(PROC_DIR / "predictions" / "piformer_huber_expanded_train_preds.csv", index=False)

    print(f"\n[piformer-huber] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
