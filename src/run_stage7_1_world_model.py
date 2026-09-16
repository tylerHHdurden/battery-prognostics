"""
Stage 7.1: train and evaluate the battery-specific World Model
(src/models/world_model.py - see its docstring for the disclosed
architecture judgment calls). Input: raw per-cycle tensors (NOT
engineered HI features) from the canonical 42-battery pool's own
fixed train/test split.

Task: given a window of W=10 past raw cycles (+ their real SOH), roll
out N=10 future-cycle SOH forecasts PURELY IN LATENT SPACE (no future
raw curves used, matching genuine forecasting - you cannot have the
future's own curves at inference time). Evaluated two ways:
1. In-domain: held-out TEST batteries from the same 42-pool split.
2. Zero-retrain: CALCE/Oxford/HUST/XJTU, no fitting on any of them,
   using the SAME model + the SAME train-fit normalization stats.

"Plausible forecast" is checked concretely: per-horizon RMSE should
grow with horizon (harder to predict further ahead) without diverging
(blowing up) or flatlining (a degenerate constant-SOH forecast, which
would show near-ZERO per-horizon RMSE GROWTH despite the true
trajectory itself changing over the horizon - checked explicitly by
also reporting the TRUE trajectory's own std/range across the horizon
for comparison).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage7_common import load_pool_train_test, load_heldout, fit_norm_stats, norm_pool, OUT_DIR, MODEL_DIR
from models.world_model import WorldModel

SEED = 42
WINDOW = 10
HORIZON = 10
WINDOW_STRIDE = 5
MAX_WINDOWS_PER_BATTERY = 150
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 25
DEVICE = torch.device("cpu")


def make_windows(pool: dict, window=WINDOW, horizon=HORIZON, stride=WINDOW_STRIDE,
                  max_per_battery=MAX_WINDOWS_PER_BATTERY, rng=None):
    """Returns list of (X_window (W,200,6), soh_window (W,), soh_target (N,), battery_id, start_cycle_idx)."""
    rng = rng or np.random.default_rng(SEED)
    samples = []
    for bid, (X, soh, rul) in pool.items():
        n = len(soh)
        starts = list(range(0, n - window - horizon + 1, stride))
        if not starts:
            continue
        if len(starts) > max_per_battery:
            starts = list(rng.choice(starts, size=max_per_battery, replace=False))
        for s in starts:
            samples.append((X[s:s + window], soh[s:s + window], soh[s + window:s + window + horizon], bid, s))
    return samples


def batches(samples, batch_size, shuffle=True, rng=None):
    idx = np.arange(len(samples))
    if shuffle:
        (rng or np.random.default_rng(SEED)).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i:i + batch_size]
        Xw = np.stack([samples[j][0] for j in sel])
        soh_w = np.stack([samples[j][1] for j in sel])
        soh_t = np.stack([samples[j][2] for j in sel])
        yield (torch.tensor(Xw, dtype=torch.float32), torch.tensor(soh_w, dtype=torch.float32),
               torch.tensor(soh_t, dtype=torch.float32))


def train_model(train_samples, val_samples):
    model = WorldModel(embed_dim=32, window=WINDOW, patch_len=2, d_model=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    best_val = float("inf")
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        tr_losses = []
        for Xw, soh_w, soh_t in batches(train_samples, BATCH_SIZE, shuffle=True, rng=rng):
            opt.zero_grad()
            preds = model(Xw, soh_w, HORIZON)
            loss = nn.functional.mse_loss(preds, soh_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xw, soh_w, soh_t in batches(val_samples, BATCH_SIZE, shuffle=False):
                preds = model(Xw, soh_w, HORIZON)
                val_losses.append(nn.functional.mse_loss(preds, soh_t).item())
        tr_mean, val_mean = float(np.mean(tr_losses)), float(np.mean(val_losses))
        print(f"[wm] epoch {epoch+1}/{EPOCHS}: train_mse={tr_mean:.4f} val_mse={val_mean:.4f}")
        if not np.isfinite(val_mean):
            print("[wm] NON-CONVERGENCE: val loss is NaN/Inf - stopping early, reporting as-is.")
            break
        if val_mean < best_val:
            best_val = val_mean
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def eval_rollout(model, samples, label):
    """Reports the model's own per-horizon RMSE ALONGSIDE a naive
    persistence baseline (predict the last-observed SOH, unchanged, for
    every future horizon) - the concrete, decisive check for "plausible
    forecast, not degenerate": a model producing near-flat per-horizon
    RMSE could either be a genuinely stable, accurate forecaster OR a
    degenerate one that has simply learned to output ~persistence and
    is being bailed out by real SOH trajectories not moving much over
    a 10-cycle horizon. Comparing directly against persistence resolves
    the ambiguity rather than eyeballing the model's RMSE curve alone."""
    model.eval()
    per_horizon_sq_err = [[] for _ in range(HORIZON)]
    persistence_sq_err = [[] for _ in range(HORIZON)]
    true_vals_per_horizon = [[] for _ in range(HORIZON)]
    with torch.no_grad():
        for Xw, soh_w, soh_t in batches(samples, 64, shuffle=False):
            preds = model(Xw, soh_w, HORIZON).numpy()
            true = soh_t.numpy()
            persist = soh_w[:, -1].numpy()  # last observed SOH, held constant
            for h in range(HORIZON):
                per_horizon_sq_err[h].extend(((preds[:, h] - true[:, h]) ** 2).tolist())
                persistence_sq_err[h].extend(((persist - true[:, h]) ** 2).tolist())
                true_vals_per_horizon[h].extend(true[:, h].tolist())
    rows = []
    for h in range(HORIZON):
        rmse_h = float(np.sqrt(np.mean(per_horizon_sq_err[h])))
        rmse_persist_h = float(np.sqrt(np.mean(persistence_sq_err[h])))
        true_std_h = float(np.std(true_vals_per_horizon[h]))
        rows.append({"dataset": label, "horizon": h + 1, "rmse_model": rmse_h,
                     "rmse_persistence_baseline": rmse_persist_h,
                     "model_beats_persistence": rmse_h < rmse_persist_h,
                     "true_soh_std_at_horizon": true_std_h, "n": len(per_horizon_sq_err[h])})
    return rows


def main():
    t0 = time.time()
    print("=== Stage 7.1: World Model (forecasting) ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train, test = load_pool_train_test()
    print(f"[wm] pool: {len(train)} train batteries, {len(test)} test batteries")
    stats = fit_norm_stats(train)
    train_n = norm_pool(train, stats)
    test_n = norm_pool(test, stats)

    rng = np.random.default_rng(SEED)
    all_train_samples = make_windows(train_n, rng=rng)
    rng.shuffle(all_train_samples)
    n_val = max(1, int(0.15 * len(all_train_samples)))
    val_samples = all_train_samples[:n_val]
    tr_samples = all_train_samples[n_val:]
    print(f"[wm] windows: {len(tr_samples)} train, {len(val_samples)} val (from train batteries, "
          f"held out by WINDOW not by battery - battery-level held-out is the TEST split below)")

    model, best_val = train_model(tr_samples, val_samples)
    print(f"[wm] best val MSE: {best_val:.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "_experimental_world_model.pt")

    all_rows = []
    test_samples = make_windows(test_n, stride=WINDOW_STRIDE, max_per_battery=1000)
    print(f"[wm] in-domain TEST windows: {len(test_samples)}")
    all_rows += eval_rollout(model, test_samples, "in-domain (test batteries)")

    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        print(f"\n[wm] loading zero-retrain set: {name}...")
        held = load_heldout(name)
        held_n = norm_pool(held, stats)
        held_samples = make_windows(held_n, stride=WINDOW_STRIDE, max_per_battery=1000)
        print(f"[wm] {name}: {len(held)} batteries, {len(held_samples)} windows")
        if not held_samples:
            print(f"[wm] {name}: no battery has >= {WINDOW+HORIZON} cycles - cannot evaluate, skipped honestly.")
            continue
        all_rows += eval_rollout(model, held_samples, name)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT_DIR / "stage7_1_world_model_results.csv", index=False)
    print("\n=== FULL PER-HORIZON RESULTS ===")
    print(results_df.to_string(index=False))
    print(f"\n[wm] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
