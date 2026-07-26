"""
Physics-informed retrain of VLSTM/CNN-LSTM/PiFormer: adds a monotonicity
penalty (src/physics_loss.py) to the standard MSE loss, weighted by ONE
small fixed lambda (not tuned, per instruction). Entirely additive - new
model files (`*_soh_physics.pt`) and prediction/metric files
(`*_physics_*`), the original (non-physics) models and their files are
untouched.

lambda_physics = 0.1, chosen once up front (not tuned): both loss terms
are O(1) in standardized-SOH units based on every prior training run in
this project, so 0.1 makes the monotonicity term a secondary regularizer
(~10% relative weight) rather than something that could dominate the
primary SOH regression objective.

Divergence safety net (per instruction): after training, if the training
loss's final epoch is not below ~1.5x its epoch-0 value, or contains
NaN/Inf, the SAME model is retrained ONCE MORE from scratch with
lambda_physics halved. No further iteration beyond that one retry.
Because the loss here is a literal SUM of two non-negative terms
(MSE >= 0, relu-penalty >= 0), it can mathematically never go negative -
unlike the earlier Kendall-uncertainty joint-loss formula, which used a
log(sigma) term that could and did go negative. That failure mode does
not apply to this design; the "or go negative" check is kept anyway as a
guard against a coding mistake, not because this formulation is expected
to trigger it.
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
from train_deep_models import load_all_battery_tensors, make_xy
from sequence_features import apply_channel_norm
from physics_loss import fit_fade_curves, monotonicity_penalty
from models.vlstm import VLSTM
from models.cnn_lstm import CNNLSTM
from models.piformer import PiFormer

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
LAMBDA_PHYSICS_INITIAL = 0.1
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 8

torch.manual_seed(42)
np.random.seed(42)


def train_physics(name, model_ctor, X_fit, y_fit, bid_fit_int, cyc_fit,
                   X_val, y_val, lambda_physics, log_fn=print):
    model = model_ctor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()

    y_mean, y_std = float(y_fit.mean()), float(y_fit.std() + 1e-8)
    model.y_mean_, model.y_std_ = y_mean, y_std

    Xt = torch.tensor(X_fit)
    yt = torch.tensor((y_fit - y_mean) / y_std).unsqueeze(-1)
    bid_t = torch.tensor(bid_fit_int, dtype=torch.long)
    cyc_t = torch.tensor(cyc_fit, dtype=torch.float32)
    Xv = torch.tensor(X_val)
    yv = torch.tensor((y_val - y_mean) / y_std).unsqueeze(-1)

    n = len(Xt)
    best_val, best_state, patience_ctr = np.inf, None, 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        ep_mse, ep_pen, ep_total = 0.0, 0.0, 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            pred = model(Xt[idx])
            mse = mse_fn(pred, yt[idx])
            pen = monotonicity_penalty(pred.squeeze(-1), bid_t[idx], cyc_t[idx])
            total = mse + lambda_physics * pen
            total.backward()
            opt.step()
            ep_mse += mse.item() * len(idx)
            ep_pen += pen.item() * len(idx)
            ep_total += total.item() * len(idx)
        ep_mse, ep_pen, ep_total = ep_mse / n, ep_pen / n, ep_total / n

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_mse = mse_fn(val_pred, yv).item()
        history.append({"epoch": epoch, "train_mse": ep_mse, "train_penalty": ep_pen,
                         "train_total": ep_total, "val_mse": val_mse})
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            log_fn(f"[physics/{name}] epoch {epoch:3d} train_mse={ep_mse:.4f} "
                   f"train_penalty={ep_pen:.4f} train_total={ep_total:.4f} val_mse={val_mse:.4f}")

        if val_mse < best_val - 1e-4:
            best_val, best_state, patience_ctr = val_mse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log_fn(f"[physics/{name}] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # divergence check
    total_losses = [h["train_total"] for h in history]
    diverged = (
        any(np.isnan(t) or np.isinf(t) for t in total_losses)
        or any(t < 0 for t in total_losses)
        or total_losses[-1] > 1.5 * total_losses[0]
    )
    return model, history, diverged


def run_one_model(name, model_ctor, X_fit, y_fit, bid_fit_int, cyc_fit, X_val, y_val, log_fn=print):
    lam = LAMBDA_PHYSICS_INITIAL
    model, history, diverged = train_physics(
        name, model_ctor, X_fit, y_fit, bid_fit_int, cyc_fit, X_val, y_val, lam, log_fn
    )
    if diverged:
        log_fn(f"[physics/{name}] DIVERGED at lambda={lam} (train_total went up >1.5x, "
               f"negative, or NaN/Inf) - retraining ONCE with lambda={lam/2}, no further retries.")
        lam = lam / 2
        model, history, diverged = train_physics(
            name, model_ctor, X_fit, y_fit, bid_fit_int, cyc_fit, X_val, y_val, lam, log_fn
        )
        log_fn(f"[physics/{name}] retry {'still diverged' if diverged else 'stable'} at lambda={lam}")
    else:
        log_fn(f"[physics/{name}] stable at lambda={lam}, no retry needed")
    return model, history, lam, diverged


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    X_fit, y_fit, rul_fit, ds_fit, bid_fit, cyc_fit = make_xy(battery_data, fit_ids)
    X_val, y_val, rul_val, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, rul_test, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[physics] fit={len(X_fit)} val={len(X_val)} test={len(X_test)} "
          f"(loaded in {time.time()-t0:.1f}s)")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    # physics prior: fit exponential fade curves on the fit battery split
    # (used only for the sanity-check log printed inside fit_fade_curves;
    # not looked up per-sample in the loss - see physics_loss.py docstring)
    fit_fade_curves(battery_data, fit_ids, log_fn=print)

    battery_to_int = {b: i for i, b in enumerate(sorted(set(bid_fit)))}
    bid_fit_int = np.array([battery_to_int[b] for b in bid_fit], dtype=np.int64)
    cyc_fit_arr = np.array(cyc_fit, dtype=np.float32)

    def predict(model, X):
        with torch.no_grad():
            raw = model(torch.tensor(X)).squeeze(-1).numpy()
        return raw * model.y_std_ + model.y_mean_

    results = {}
    all_metrics = []
    lambdas_used = {}

    model_specs = [
        ("VLSTM", lambda: VLSTM(input_size=1, hidden_size=32, n_targets=1), X_fit[:, :, 0:1], X_val[:, :, 0:1], X_test[:, :, 0:1]),
        ("CNNLSTM", lambda: CNNLSTM(), X_fit, X_val, X_test),
        ("PiFormer", lambda: PiFormer(), X_fit, X_val, X_test),
    ]

    for name, ctor, Xf, Xv, Xte in model_specs:
        print(f"\n[physics] === training {name} with physics-informed loss ===")
        model, hist, lam_used, diverged = run_one_model(
            name, ctor, Xf, y_fit, bid_fit_int, cyc_fit_arr, Xv, y_val
        )
        lambdas_used[name] = lam_used
        pred_test = predict(model, Xte)
        results[name] = pred_test
        torch.save(model.state_dict(), ROOT / "models" / f"{name.lower()}_soh_physics.pt")
        pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / f"{name.lower()}_physics_history.csv", index=False)

        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        mae = mean_absolute_error(y_test, pred_test)
        r2 = r2_score(y_test, pred_test)
        print(f"[physics] {name} TEST (physics-informed) RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} "
              f"(lambda={lam_used}, diverged_at_initial_lambda={diverged if lam_used != LAMBDA_PHYSICS_INITIAL else False})")
        all_metrics.append({"model": f"{name}-physics", "rmse": rmse, "mae": mae, "r2": r2,
                             "lambda_physics": lam_used})

    out = pd.DataFrame({
        "dataset": ds_test, "battery_id": bid_test, "cycle_idx": cyc_test, "SOH": y_test,
        "y_pred_VLSTM_physics": results["VLSTM"],
        "y_pred_CNNLSTM_physics": results["CNNLSTM"],
        "y_pred_PiFormer_physics": results["PiFormer"],
    })
    out.to_csv(PROC_DIR / "predictions" / "deep_models_physics_test_preds.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(PROC_DIR / "predictions" / "deep_models_physics_metrics.csv", index=False)

    baseline = pd.read_csv(PROC_DIR / "predictions" / "deep_models_metrics.csv")
    print("\n[physics] === BASELINE (no physics loss) vs PHYSICS-INFORMED ===")
    print(baseline.to_string(index=False))
    print(pd.DataFrame(all_metrics)[["model", "rmse", "mae", "r2"]].to_string(index=False))
    print(f"[physics] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
