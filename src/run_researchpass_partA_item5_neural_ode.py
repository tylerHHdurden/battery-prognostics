"""
Part A, item 5: Neural-ODE, continuous-TIME modeling of a per-cycle
curve for SOH estimation - positioned deliberately as an alternative
AXIS to Stage 7.3's DeepONet, which models continuity in SPACE
(branch/trunk over an arbitrary query point within a cycle's I(t)->V(t)
operator; see DEVELOPMENT_LOG.md's "7.3 - DeepONet as a real SPM/SPMe
surrogate" entry, re-read before writing this item, not from memory
alone: in-pool R2=0.976-0.982, CALCE=-2.652, Oxford=-4.240, HUST=
-9.176, XJTU=-5.541 - these are Stage 7.3's OWN transcribed numbers,
used below as the explicit different-axis comparison this item's task
description calls for).

Reference, verified via direct WebFetch of the arXiv abstract page
before writing this (not assumed from the task's own description): Li,
He, Liu, "A novel Neural-ODE model for the state of health estimation
of lithium-ion battery using charging curve," arXiv:2505.05803 -
PREPRINT ONLY, not peer-reviewed (submitted May 2025). Their own
architecture ("ACLA": attention + CNN/LSTM + augmented Neural ODE)
processes normalized CC-phase CHARGING-curve time data, reporting RMSE
1.01%/2.24% on TJU/HUST.

TWO DISCLOSED, DELIBERATE SCOPE REDUCTIONS from the paper, stated
plainly rather than silently substituted:
1. CURVE TYPE: the paper uses CHARGING curves specifically. This
   project's existing raw-per-cycle-tensor infrastructure
   (`stage7_common.py`, reused unchanged by every Stage 7 item and by
   this pass's own item 2) is built around DISCHARGE curves across all
   6 dataset adapters - building an analogous, separately-normalized
   CHARGING-curve tensor pipeline from scratch across NASA/MIT/CALCE/
   Oxford/HUST/XJTU's six different raw-data adapters was judged out of
   this pass's time budget and NOT attempted. This item instead applies
   the Neural-ODE mechanism to the discharge curve's own V_t/I_t/T_t
   channels (channels 0-2 of the existing 6-channel tensor,
   `sequence_features.py`) - the project's own established per-cycle
   continuous-time curve representation. The item's actual axis of
   interest (continuous-in-TIME vs. DeepONet's continuous-in-SPACE) is
   preserved; the specific curve TYPE is not. Not presented as a
   reproduction of the paper's own charging-curve input.
2. ARCHITECTURE: no attention/CNN/LSTM front-end (the paper's "ACLA"),
   no `torchdiffeq` (not installed in this environment - checked
   directly, not assumed) and no adjoint method. A genuine Neural ODE
   is still used: a small MLP `f_theta(z, u_t, t)` integrated by a
   hand-rolled, FIXED-STEP RK4 solver (49 steps, direct backprop through
   the unrolled steps - a standard, well-known black-box-ODE-via-
   unrolling approach for a short, fixed-length sequence on CPU), with
   each cycle's OWN observed (V, I, T) at the current time bin injected
   as a piecewise-constant (zero-order-hold) forcing input `u_t` at
   every RK4 sub-step within that step - a disclosed simplification
   (a true Neural CDE would interpolate `u` continuously between
   observations; ZOH is used here for simplicity, not claimed to be
   the more general continuous-control formulation).
"""
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
from stage7_common import load_pool_train_test, load_heldout, fit_norm_stats, norm_pool, OUT_DIR, MODEL_DIR

SEED = 42
DEVICE = torch.device("cpu")
N_STEPS = 49         # RK4 steps (50 curve points -> 49 intervals) - downsampled from 200 bins, disclosed
STRIDE = 4            # 200 // 50 ~= 4 -> 50 points
Z_DIM = 16
EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3

# Stage 7.3's DeepONet reconstruction-only numbers, transcribed directly
# from DEVELOPMENT_LOG.md's own "7.3" entry (re-read before writing this
# script) - the explicit different-axis (continuous-space) reference.
DEEPONET_STAGE7_3 = {"in-domain (fixed split)": 0.982, "CALCE": -2.652, "Oxford": -4.240, "HUST": -9.176, "XJTU": -5.541}
ROUTED_BASELINE = {"in-domain (fixed split)": 0.974, "CALCE": 0.740, "Oxford": 0.953, "HUST": 0.800, "XJTU": -1.062}


class ODEFunc(nn.Module):
    def __init__(self, z_dim=Z_DIM, u_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + u_dim + 1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, z_dim),
        )

    def forward(self, z, u, t_scalar):
        t_col = torch.full((z.shape[0], 1), float(t_scalar), device=z.device)
        return self.net(torch.cat([z, u, t_col], dim=-1))


class NeuralODESOH(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = ODEFunc()
        self.head = nn.Sequential(nn.Linear(Z_DIM, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        """curve: (batch, N_STEPS+1, 3) - V/I/T at 50 normalized time
        points. RK4-integrates z from t=0 to t=1 with ZOH forcing."""
        batch = curve.shape[0]
        dt = 1.0 / N_STEPS
        z = torch.zeros(batch, Z_DIM, device=curve.device)
        for step in range(N_STEPS):
            u = curve[:, step, :]  # ZOH forcing for this interval
            t0 = step * dt
            k1 = self.func(z, u, t0)
            k2 = self.func(z + 0.5 * dt * k1, u, t0 + 0.5 * dt)
            k3 = self.func(z + 0.5 * dt * k2, u, t0 + 0.5 * dt)
            k4 = self.func(z + dt * k3, u, t0 + dt)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.head(z).squeeze(-1)


def downsample_curves(X6: np.ndarray) -> np.ndarray:
    """(n, 200, 6) -> (n, 50, 3), channels V_t/I_t/T_t only, strided."""
    return X6[:, ::STRIDE, :3][:, :N_STEPS + 1, :]


def build_flat(pool: dict):
    Xs, sohs = [], []
    for bid, (X, soh, rul) in pool.items():
        Xs.append(downsample_curves(X)); sohs.append(soh)
    return np.concatenate(Xs), np.concatenate(sohs)


def score(y_true, pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred)),
            "r2": float(r2_score(y_true, pred)), "n": int(len(y_true))}


def main():
    t0 = time.time()
    print("=== Part A, item 5: Neural-ODE, continuous-TIME per-cycle-curve SOH model ===")
    torch.manual_seed(SEED); np.random.seed(SEED)

    print("[item5] loading canonical 42-battery pool (raw tensors, reused from stage7_common)...")
    train_raw, test_raw = load_pool_train_test()
    stats = fit_norm_stats(train_raw)
    train_norm = norm_pool(train_raw, stats)
    test_norm = norm_pool(test_raw, stats)

    X_tr, soh_tr = build_flat(train_norm)
    soh_mean, soh_std = float(soh_tr.mean()), float(soh_tr.std() + 1e-8)
    print(f"[item5] {len(X_tr)} train cycles, soh_mean={soh_mean:.3f} soh_std={soh_std:.3f}, "
          f"curve shape per cycle: {X_tr.shape[1:]}")

    model = NeuralODESOH().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor((soh_tr - soh_mean) / soh_std, dtype=torch.float32)
    n = len(Xt)
    rng = np.random.default_rng(SEED)

    print("\n[item5] training (RK4-integrated Neural ODE, direct backprop through unrolled steps)...")
    for epoch in range(EPOCHS):
        idx = rng.permutation(n)
        losses = []
        for s in range(0, n, BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            pred = model(Xt[b])
            loss = nn.functional.mse_loss(pred, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        print(f"[item5] epoch {epoch+1}/{EPOCHS}: mean_mse(z-scored)={mean_loss:.4f}")
        if not np.isfinite(mean_loss):
            print("[item5] NON-CONVERGENCE: loss non-finite - stopping early.")
            break
    torch.save(model.state_dict(), MODEL_DIR / "_experimental_neural_ode_soh.pt")

    @torch.no_grad()
    def predict(X):
        pred_z = model(torch.tensor(X, dtype=torch.float32))
        return pred_z.numpy() * soh_std + soh_mean

    results = []
    X_test, soh_test = build_flat(test_norm)
    pred_test = predict(X_test)
    r_in = score(soh_test, pred_test)
    print(f"\n[item5] in-domain (fixed split): R2={r_in['r2']:.4f} RMSE={r_in['rmse']:.4f} n={r_in['n']} | "
          f"vs routed deployed baseline {ROUTED_BASELINE['in-domain (fixed split)']:.3f} | "
          f"vs Stage 7.3 DeepONet (continuous-SPACE) {DEEPONET_STAGE7_3['in-domain (fixed split)']:.3f}")
    results.append({"dataset": "in-domain (fixed split)", **r_in,
                     "routed_baseline_r2": ROUTED_BASELINE["in-domain (fixed split)"],
                     "deeponet_stage7_3_r2": DEEPONET_STAGE7_3["in-domain (fixed split)"]})

    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        print(f"\n[item5] loading {name} (raw cycles, zero-retrain)...")
        held_raw = load_heldout(name)
        if not held_raw:
            print(f"[item5] WARNING: no usable {name} data, skipping")
            continue
        held_norm = norm_pool(held_raw, stats)
        X_h, soh_h = build_flat(held_norm)
        pred_h = predict(X_h)
        r = score(soh_h, pred_h)
        v = "WIN" if r["r2"] > ROUTED_BASELINE[name] else "LOSS"
        vs_deeponet = "BEATS DeepONet" if r["r2"] > DEEPONET_STAGE7_3[name] else "WORSE than DeepONet"
        print(f"[item5] {name}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={r['n']} | "
              f"vs routed deployed baseline {ROUTED_BASELINE[name]:.3f} -> {v} | "
              f"vs Stage 7.3 DeepONet (continuous-SPACE) {DEEPONET_STAGE7_3[name]:.3f} -> {vs_deeponet}")
        results.append({"dataset": name, **r, "routed_baseline_r2": ROUTED_BASELINE[name],
                         "deeponet_stage7_3_r2": DEEPONET_STAGE7_3[name], "verdict_vs_routed": v,
                         "verdict_vs_deeponet": vs_deeponet})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass_partA_item5_neural_ode.csv", index=False)
    print("\n=== Item 5 (Neural-ODE, continuous-time) FULL RESULTS ===")
    print(results_df.to_string(index=False))
    print(f"\n[item5] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
