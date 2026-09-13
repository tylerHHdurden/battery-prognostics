"""
Stage 1, Items 1.3 (deep-model half) + 1.4: Huber loss on VLSTM/CNN-LSTM/
CNN-BiGRU (never tried before - only PiFormer had it, session 35 Part 1),
and PiFormer with Huber+noise-injection COMBINED (also never tried -
session 35 tried them separately, Parts 1 and 3).

All on the EXPANDED (204-battery) pool, unpinned battery_split_expanded.json
- same pool session 35's Huber/noise work used, and the ONLY pool where
B0053 (this project's motivating outlier battery for both items) exists
at all (confirmed absent from the original 32-battery pool before writing
any of this stage's code). Same architecture/hyperparameters as every
other deep model in this project (40 epochs, batch=64, lr=1e-3,
patience=8) - only the loss function (and, for PiFormer, ALSO the
training-time noise injection) changes.

Resilience: per-MODEL resume (skip a model entirely if its output
metrics file already exists) layered on top of per-EPOCH checkpoint/
resume (this project's now-standard pattern, copied fresh here per
convention rather than sharing code with train_piformer_huber_expanded.py/
train_noise_augmented_expanded.py, so a bug in one script can't affect
another). Models trained SEQUENTIALLY, not in parallel, to avoid CPU
contention on this 12-core/no-GPU machine.
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
from models.cnn_bigru import CNNBiGRU
from models.piformer import PiFormer

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
PRED_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

SIGMA_V, SIGMA_I, SIGMA_T = 0.001, 0.01, 0.5  # identical to session 35 Part 3's "1x BMS-grade" tier


def _epoch_ckpt_path(name: str) -> Path:
    return PROC_DIR / f"_stage1_3_4_{name}_epoch_checkpoint.pt"


def train_one_model_resumable(name, model, X_train, y_train, X_val, y_val,
                               noise_std_normalized=None,
                               epochs=40, batch_size=64, lr=1e-3, patience=8):
    """Per-epoch checkpoint/resume, Huber loss, OPTIONAL training-time
    Gaussian noise injection on X_train only (never X_val) - a strict
    superset of train_piformer_huber_expanded.py's pattern (Huber alone)
    and train_noise_augmented_expanded.py's pattern (noise alone, MSE):
    passing noise_std_normalized=None reduces to plain Huber (used for
    1.3's VLSTM/CNN-LSTM/CNN-BiGRU); passing an array combines both
    (used for 1.4's PiFormer)."""
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)
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
        print(f"[stage1.3-4/{name}] RESUMING from epoch {start_epoch} (checkpoint found)")
    else:
        y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)
        best_val, best_state, patience_ctr, history, start_epoch = np.inf, None, 0, [], 0

    model.y_mean_, model.y_std_ = y_mean, y_std
    Xt = torch.tensor(X_train)
    yt = torch.tensor((y_train - y_mean) / y_std).unsqueeze(-1)
    Xv = torch.tensor(X_val)
    yv = torch.tensor((y_val - y_mean) / y_std).unsqueeze(-1)
    noise_std_t = torch.tensor(noise_std_normalized, dtype=Xt.dtype) if noise_std_normalized is not None else None
    n = len(Xt)

    for epoch in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            if noise_std_t is not None:
                xb = xb + torch.randn_like(xb) * noise_std_t
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)  # clean val data always, regardless of noise-training
            val_loss = loss_fn(val_pred, yv).item()
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        print(f"[stage1.3-4/{name}] epoch {epoch:3d} train_huber={epoch_loss:.4f} val_huber={val_loss:.4f}")

        stop = False
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[stage1.3-4/{name}] early stopping at epoch {epoch}")
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


def per_battery_rmse(bid_arr, y_true, y_pred):
    df = pd.DataFrame({"battery_id": bid_arr, "abs_err": np.abs(y_true - y_pred)})
    s = df.groupby("battery_id").apply(
        lambda g: float(np.sqrt((g["abs_err"] ** 2).mean())), include_groups=False
    ).sort_values(ascending=False)
    s.name = "rmse"
    return s


def run_model(name, model_ctor, X_input_slicer, metrics_path, preds_path, per_batt_path,
              ckpt_out_path, X_fit, y_fit, X_val, y_val, X_test, y_test, ds_test, bid_test, cyc_test,
              noise_std_normalized=None, baseline_rmse=None, baseline_r2=None):
    if metrics_path.exists():
        print(f"[stage1.3-4] {name}: metrics file already exists, SKIPPING (model-level resume) - {metrics_path.name}")
        return pd.read_csv(metrics_path).iloc[0].to_dict()

    t1 = time.time()
    model = model_ctor()
    model, hist = train_one_model_resumable(name, model, X_input_slicer(X_fit), y_fit,
                                             X_input_slicer(X_val), y_val,
                                             noise_std_normalized=noise_std_normalized)
    elapsed = time.time() - t1
    print(f"[stage1.3-4] {name} trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    torch.save(model.state_dict(), ckpt_out_path)

    pred_test = predict(model, X_input_slicer(X_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))
    print(f"[stage1.3-4] {name} TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} (elapsed_min={elapsed/60:.1f})")
    if baseline_rmse is not None:
        print(f"[stage1.3-4] {name} vs. MSE baseline: RMSE {baseline_rmse:.4f}->{rmse:.4f} "
              f"({(rmse-baseline_rmse)/baseline_rmse*100:+.1f}%), R2 {baseline_r2:.4f}->{r2:.4f}")

    batt_rmse = per_battery_rmse(bid_test, y_test, pred_test)
    print(f"[stage1.3-4] {name} worst 5 batteries:\n{batt_rmse.head(5).to_string()}")
    if "B0053" in batt_rmse.index:
        print(f"[stage1.3-4] {name} B0053 RMSE: {batt_rmse['B0053']:.4f}")

    out = pd.DataFrame({"dataset": ds_test, "battery_id": bid_test, "cycle_idx": cyc_test,
                         "SOH": y_test, f"y_pred_{name}": pred_test})
    out.to_csv(preds_path, index=False)
    batt_rmse.reset_index().to_csv(per_batt_path, index=False)
    metrics = {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "train_seconds": elapsed}
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    return metrics


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors_expanded()
    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[stage1.3-4] fit batteries: {len(fit_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    print(f"[stage1.3-4] B0053 in test set: {'B0053' in test_ids}")

    X_fit, y_fit, _, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, y_val, _, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, _, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[stage1.3-4] fit cycles={len(X_fit)}, val cycles={len(X_val)}, test cycles={len(X_test)}")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    noise_std_norm = np.zeros(6, dtype=np.float32)
    noise_std_norm[0] = SIGMA_V / norm_stats[0]["std"]
    noise_std_norm[1] = SIGMA_I / norm_stats[1]["std"]
    noise_std_norm[2] = SIGMA_T / norm_stats[2]["std"]

    existing_mse = pd.read_csv(PROC_DIR / "predictions" / "deep_models_expanded_metrics.csv").set_index("model")
    existing_bigru_mse = pd.read_csv(PROC_DIR / "predictions" / "cnn_bigru_expanded_metrics.csv")
    bigru_mse_row = existing_bigru_mse.iloc[0]

    results = []

    # --- 1.3: VLSTM-Huber ---
    results.append(run_model(
        "VLSTM-huber", lambda: VLSTM(input_size=1, hidden_size=32, n_targets=1),
        lambda X: X[:, :, 0:1],
        OUT_DIR / "stage1_3_vlstm_huber_metrics.csv", PRED_DIR / "vlstm_huber_expanded_test_preds.csv",
        OUT_DIR / "stage1_3_vlstm_huber_per_battery.csv", ROOT / "models" / "vlstm_soh_huber_expanded.pt",
        X_fit, y_fit, X_val, y_val, X_test, y_test, ds_test, bid_test, cyc_test,
        baseline_rmse=float(existing_mse.loc["VLSTM", "rmse"]), baseline_r2=float(existing_mse.loc["VLSTM", "r2"]),
    ))

    # --- 1.3: CNN-LSTM-Huber ---
    results.append(run_model(
        "CNNLSTM-huber", lambda: CNNLSTM(), lambda X: X,
        OUT_DIR / "stage1_3_cnnlstm_huber_metrics.csv", PRED_DIR / "cnn_lstm_huber_expanded_test_preds.csv",
        OUT_DIR / "stage1_3_cnnlstm_huber_per_battery.csv", ROOT / "models" / "cnn_lstm_soh_huber_expanded.pt",
        X_fit, y_fit, X_val, y_val, X_test, y_test, ds_test, bid_test, cyc_test,
        baseline_rmse=float(existing_mse.loc["CNNLSTM", "rmse"]), baseline_r2=float(existing_mse.loc["CNNLSTM", "r2"]),
    ))

    # --- 1.3: CNN-BiGRU-Huber ---
    results.append(run_model(
        "CNNBiGRU-huber", lambda: CNNBiGRU(), lambda X: X,
        OUT_DIR / "stage1_3_cnnbigru_huber_metrics.csv", PRED_DIR / "cnn_bigru_huber_expanded_test_preds.csv",
        OUT_DIR / "stage1_3_cnnbigru_huber_per_battery.csv", ROOT / "models" / "cnn_bigru_soh_huber_expanded.pt",
        X_fit, y_fit, X_val, y_val, X_test, y_test, ds_test, bid_test, cyc_test,
        baseline_rmse=float(bigru_mse_row["rmse"]), baseline_r2=float(bigru_mse_row["r2"]),
    ))

    # --- 1.4: PiFormer Huber + noise, COMBINED ---
    existing_piformer_huber = pd.read_csv(PROC_DIR / "predictions" / "piformer_huber_expanded_metrics.csv").iloc[0]
    existing_piformer_noiseaug = None
    noiseaug_metrics_path = PROC_DIR / "predictions" / "deep_models_noiseaug_expanded_metrics.csv"
    if noiseaug_metrics_path.exists():
        na = pd.read_csv(noiseaug_metrics_path).set_index("model")
        if "PiFormer-noiseaug" in na.index:
            existing_piformer_noiseaug = na.loc["PiFormer-noiseaug"]

    results.append(run_model(
        "PiFormer-huber-noise", lambda: PiFormer(), lambda X: X,
        OUT_DIR / "stage1_4_piformer_huber_noise_metrics.csv",
        PRED_DIR / "piformer_huber_noise_expanded_test_preds.csv",
        OUT_DIR / "stage1_4_piformer_huber_noise_per_battery.csv",
        ROOT / "models" / "piformer_soh_huber_noise_expanded.pt",
        X_fit, y_fit, X_val, y_val, X_test, y_test, ds_test, bid_test, cyc_test,
        noise_std_normalized=noise_std_norm,
        baseline_rmse=float(existing_mse.loc["PiFormer", "rmse"]), baseline_r2=float(existing_mse.loc["PiFormer", "r2"]),
    ))

    print("\n[stage1.3-4] === FINAL SUMMARY (1.3 + 1.4) ===")
    for r in results:
        print(f"[stage1.3-4] {r['model']}: RMSE={r['rmse']:.4f} MAE={r['mae']:.4f} R2={r['r2']:.4f}")
    print(f"\n[stage1.3-4] PiFormer-huber-noise (1.4) vs. (a) MSE baseline RMSE={existing_mse.loc['PiFormer','rmse']:.4f} "
          f"R2={existing_mse.loc['PiFormer','r2']:.4f}; (b) Huber-only (session 35 Part 1) "
          f"RMSE={existing_piformer_huber['rmse']:.4f} R2={existing_piformer_huber['r2']:.4f}"
          + (f"; (c) noise-only (session 35 Part 3) RMSE={existing_piformer_noiseaug['rmse']:.4f} "
             f"R2={existing_piformer_noiseaug['r2']:.4f}" if existing_piformer_noiseaug is not None else "; (c) noise-only: NOT FOUND on disk"))

    pd.DataFrame(results).to_csv(OUT_DIR / "stage1_3_4_all_results.csv", index=False)
    print(f"\n[stage1.3-4] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
