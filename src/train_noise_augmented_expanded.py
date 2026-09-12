"""
Targeted improvement pass, Part 3: retrain VLSTM/CNN-LSTM/PiFormer with
1x-BMS-grade Gaussian noise injected on V/I/T DURING training (not just
at eval time), then rebuild the fusion ensemble on top of them - does
noise-as-regularization close the robustness-margin gap the Dataset
Expansion session found (expanded-pool model: better clean R2 (0.983)
but a THINNER margin under stress, 5x-noise R2=0.870, actually dipping
BELOW the original 32-battery model's own CLEAN R2 of 0.917)?

Noise convention: IDENTICAL sigma values to run_sensor_noise_robustness
_expanded.py's "1x BMS-grade" tier (sigma_V=1mV, sigma_I=10mA,
sigma_T=0.5C) - same physical units, same magnitude, so "trained on the
noise it's evaluated against" is a literal, not approximate, match.

METHODOLOGICAL LIMITATION, stated explicitly rather than silently
assumed away: the eval script injects noise on RAW cycle V/I/T arrays
BEFORE deriving the dQdV/dVdQ/dIdV differential channels, so noise
propagates physically through the ICA/DV/DC computation too. Re-running
that full raw-to-tensor pipeline on every training batch, every epoch,
for 95,572 fit cycles x 40 epochs, is computationally prohibitive on
this CPU-only machine (the ORIGINAL tensor build alone took ~7 minutes
for the whole pool, once). This script instead injects noise directly
onto the ALREADY-BUILT, ALREADY-CHANNEL-NORMALIZED V_t/I_t/T_t tensor
channels (0/1/2) at each training step, converting the raw-unit sigma
into normalized-space sigma via the same per-channel std used to build
channel_norm_stats_expanded.json (noise is additive under a linear
z-score transform, so this is an exact conversion, not an approximation
of the SCALE - only the propagation through the differential channels
3/4/5 is skipped). This is standard, practical noise-augmentation
practice (perturb the model's actual input tensor, fresh every
forward pass) - not identical to the eval's from-raw-data method, and
that gap is real: report the eval results honestly regardless of
whether this limitation turns out to matter.

Same architecture, same data, same 204-battery split
(battery_split_expanded.json, unpinned, same as Part 1 - see that
script's docstring for why), same 40-epoch/patience-8 budget. Additive:
vlstm_soh_expanded.pt / cnn_lstm_soh_expanded.pt /
piformer_soh_expanded.pt (the existing clean-trained checkpoints) are
left completely untouched; this script writes *_noiseaug_expanded.pt.

Resilience: per-epoch checkpoint/resume (train_cnn_bigru_expanded.py's
verified pattern) PLUS model-level resume (train_deep_models_expanded.
py's load_or_train pattern) stacked together - this is the single
longest-running stage in this entire session (~5.6h for the 3 models
combined, per the Dataset Expansion session's un-augmented timing), and
this project's history says that's exactly the kind of run an external
session teardown targets.
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
from sequence_features import apply_channel_norm, CHANNEL_NAMES
from models.vlstm import VLSTM
from models.cnn_lstm import CNNLSTM
from models.piformer import PiFormer

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
(PROC_DIR / "predictions").mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

# Same physical-unit sigmas as run_sensor_noise_robustness_expanded.py's "1x BMS-grade" tier.
SIGMA_V, SIGMA_I, SIGMA_T = 0.001, 0.01, 0.5


def _epoch_ckpt_path(name: str) -> Path:
    return PROC_DIR / f"_noiseaug_{name}_epoch_checkpoint.pt"


def train_one_model_noise_augmented(name, model, X_train, y_train, X_val, y_val,
                                     noise_std_normalized: np.ndarray,
                                     epochs=40, batch_size=64, lr=1e-3, patience=8):
    """train_cnn_bigru_expanded.py's verified per-epoch checkpoint/resume
    pattern, with ONE addition: fresh Gaussian noise (std given per-
    channel, already converted to normalized-space units) added to
    X_train on every batch, every epoch - never applied to X_val (val
    loss must reflect clean-signal generalization, matching how early
    stopping/model selection works everywhere else in this project) and
    never applied to the model's actual held-out TEST evaluation later
    (that happens separately, in the noise-robustness eval script, which
    controls its own noise levels explicitly)."""
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ckpt_path = _epoch_ckpt_path(name)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
        best_val, best_state, patience_ctr = ckpt["best_val"], ckpt["best_state"], ckpt["patience_ctr"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[noiseaug/{name}] RESUMING from epoch {start_epoch} (checkpoint found)")
    else:
        y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)
        best_val, best_state, patience_ctr, history, start_epoch = np.inf, None, 0, [], 0

    model.y_mean_, model.y_std_ = y_mean, y_std
    Xt = torch.tensor(X_train)
    yt = torch.tensor((y_train - y_mean) / y_std).unsqueeze(-1)
    Xv = torch.tensor(X_val)
    yv = torch.tensor((y_val - y_mean) / y_std).unsqueeze(-1)
    noise_std_t = torch.tensor(noise_std_normalized, dtype=Xt.dtype)  # shape (n_channels,)
    n = len(Xt)

    for epoch in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            noise = torch.randn_like(xb) * noise_std_t  # broadcasts over (batch, seq, channel)
            xb_noisy = xb + noise
            opt.zero_grad()
            pred = model(xb_noisy)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)  # CLEAN val data - no noise at validation time
            val_loss = loss_fn(val_pred, yv).item()
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        print(f"[noiseaug/{name}] epoch {epoch:3d} train_mse(noisy)={epoch_loss:.4f} val_mse(clean)={val_loss:.4f}")

        stop = False
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[noiseaug/{name}] early stopping at epoch {epoch}")
                stop = True

        torch.save({
            "model_state": model.state_dict(), "opt_state": opt.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "y_mean": y_mean, "y_std": y_std,
            "best_val": best_val, "best_state": best_state, "patience_ctr": patience_ctr,
            "history": history, "epoch": epoch,
        }, ckpt_path)
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path.unlink(missing_ok=True)
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
    print(f"[noiseaug] fit batteries: {len(fit_ids)}, val batteries: {len(val_ids)}, "
          f"test batteries: {len(test_ids)}")

    X_fit, y_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, y_val, rul_val, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, rul_test, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[noiseaug] fit cycles={len(X_fit)}, val cycles={len(X_val)}, test cycles={len(X_test)}")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    # Convert raw-unit sigma -> normalized-space sigma per channel
    # (additive noise commutes with the (x-mean)/std transform).
    noise_std_norm = np.zeros(6, dtype=np.float32)
    noise_std_norm[0] = SIGMA_V / norm_stats[0]["std"]  # V_t
    noise_std_norm[1] = SIGMA_I / norm_stats[1]["std"]  # I_t
    noise_std_norm[2] = SIGMA_T / norm_stats[2]["std"]  # T_t
    # channels 3/4/5 (dQdV/dVdQ/dIdV) left at 0 - see module docstring's
    # stated limitation (not re-derived from noisy V/I/T here).
    for c in range(6):
        print(f"[noiseaug] channel {c} ({CHANNEL_NAMES[c]}) train-time noise std "
              f"(normalized space): {noise_std_norm[c]:.5f}")

    results = {}

    def load_or_train(name, model, ckpt_name, X_fit_m, X_val_m, hist_name, noise_std_slice):
        ckpt_path = ROOT / "models" / ckpt_name
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()
            model.y_mean_, model.y_std_ = float(y_fit.mean()), float(y_fit.std() + 1e-8)
            print(f"[noiseaug] {name}: loaded existing checkpoint {ckpt_name}, skipping retraining")
            return model
        t1 = time.time()
        model, hist = train_one_model_noise_augmented(name, model, X_fit_m, y_fit, X_val_m, y_val,
                                                        noise_std_slice)
        print(f"[noiseaug] {name} trained in {time.time()-t1:.1f}s")
        torch.save(model.state_dict(), ckpt_path)
        pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / hist_name, index=False)
        return model

    vlstm = load_or_train("VLSTM-noiseaug", VLSTM(input_size=1, hidden_size=32, n_targets=1),
                           "vlstm_soh_noiseaug_expanded.pt", X_fit[:, :, 0:1], X_val[:, :, 0:1],
                           "vlstm_noiseaug_expanded_history.csv", noise_std_norm[0:1])
    results["VLSTM"] = predict(vlstm, X_test[:, :, 0:1])

    cnn_lstm = load_or_train("CNNLSTM-noiseaug", CNNLSTM(), "cnn_lstm_soh_noiseaug_expanded.pt",
                              X_fit, X_val, "cnn_lstm_noiseaug_expanded_history.csv", noise_std_norm)
    results["CNNLSTM"] = predict(cnn_lstm, X_test)

    piformer = load_or_train("PiFormer-noiseaug", PiFormer(), "piformer_soh_noiseaug_expanded.pt",
                              X_fit, X_val, "piformer_noiseaug_expanded_history.csv", noise_std_norm)
    results["PiFormer"] = predict(piformer, X_test)

    all_metrics = []
    for name, pred in results.items():
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"[noiseaug] {name} (noise-augmented) TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        all_metrics.append({"model": f"{name}-noiseaug", "rmse": rmse, "mae": mae, "r2": r2})

    out = pd.DataFrame({
        "dataset": ds_test, "battery_id": bid_test, "cycle_idx": cyc_test,
        "SOH": y_test, "RUL": rul_test,
        "y_pred_VLSTM": results["VLSTM"], "y_pred_CNNLSTM": results["CNNLSTM"],
        "y_pred_PiFormer": results["PiFormer"],
    })
    out.to_csv(PROC_DIR / "predictions" / "deep_models_noiseaug_expanded_test_preds.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(PROC_DIR / "predictions" / "deep_models_noiseaug_expanded_metrics.csv", index=False)

    ids_fitval = fit_ids + val_ids
    _, ytr, rultr, dstr, bidtr, cyctr = make_xy(battery_data, ids_fitval)
    Xtr = np.concatenate([X_fit, X_val])
    tr_vlstm = predict(vlstm, Xtr[:, :, 0:1])
    tr_cnnlstm = predict(cnn_lstm, Xtr)
    tr_piformer = predict(piformer, Xtr)
    out_train = pd.DataFrame({
        "dataset": dstr, "battery_id": bidtr, "cycle_idx": cyctr,
        "SOH": ytr, "RUL": rultr,
        "y_pred_VLSTM": tr_vlstm, "y_pred_CNNLSTM": tr_cnnlstm, "y_pred_PiFormer": tr_piformer,
    })
    out_train.to_csv(PROC_DIR / "predictions" / "deep_models_noiseaug_expanded_train_preds.csv", index=False)

    print(f"\n[noiseaug] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
