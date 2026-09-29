"""
Part A, item 2: physics-guided test-time training (TTT), following the
DESIGN IDEA of Feng, Hu, Li, Zhang, "Adapting Amidst Degradation: Cross
Domain Li-ion Battery Health Estimation via Physics-Guided Test-Time
Training" ("BatteryTTT" / "GPT4Battery"), arXiv:2402.00068 - VERIFIED
via direct WebFetch of the arXiv abstract page before writing this:
this is a PREPRINT (submission history shows v1 Jan 2024, v3 Nov 2024,
no confirmed peer-reviewed venue found) - NOT presented here as more
validated than that. The paper's own core claim: the model "adapts
continually using each unlabeled target data collected amidst
degradation," combining physics-informed constraints with self-
supervised learning DURING TESTING (not just a one-time pretraining
phase). This project already has ONE prior physics-simulated
CONTRASTIVE PRETRAINING attempt (Research Pass 2, item 3, PyBaMM SPM-
based - a pretraining-phase method, weights frozen/fine-tuned once
before any evaluation). This item is mechanically different: continuous
GRADIENT-BASED ADAPTATION DURING evaluation itself, one unlabeled cycle
at a time, per held-out battery - not a repeat.

DISCLOSED ARCHITECTURE CHANGE (same standing as Stage 7.2's own
precedent): TTT requires a differentiable model whose weights can be
updated by gradient steps at test time - this project's DEPLOYED model
is XGBoost (not differentiable in that sense). This item therefore uses
a FRESH small CNN encoder (`CycleCNNEncoder`, already used elsewhere in
this project's Stage 7/Research-Pass-2 work) on the RAW per-cycle
tensor representation (`stage7_common`'s own pipeline, not the deployed
HI+fusion tabular pipeline) - an intentionally-unfair-by-architecture
comparison against the deployed XGBoost-fusion model, disclosed
upfront exactly as Stage 7.2 disclosed it for its own small-CNN
comparison. The PRIMARY, fair comparison for this item's own merit is
therefore TTT-adapted vs. a ZERO-SHOT control using the IDENTICAL
architecture/weights with TTT switched off - isolating what continuous
test-time adaptation itself contributes, independent of the known
architecture gap. The SECONDARY comparison (both variants vs. the
routed deployed baseline) is also reported, per this research pass's
own full-protocol instruction, with the architecture gap disclosed
plainly rather than glossed over.

PHYSICS-GUIDED SELF-SUPERVISED SIGNAL (a genuine, disclosed
simplification of a full ECM, not a literal circuit-parameter fit):
this project's own raw-tensor pipeline (`sequence_features.py`)
resamples each cycle's discharge V(t)/I(t) onto 200 NORMALIZED time
bins (fraction of discharge duration, not real seconds) - a full
nonlinear multi-RC ECM relaxation fit is not meaningfully recoverable
from that representation (no real time axis is retained). Instead, a
SINGLE-RESISTOR ECM proxy is used: R_apparent = (V_t[0] - V_t[-1]) /
mean(|I_t|) per cycle - the whole-discharge IR-drop-over-current ratio,
a real, physically-motivated (if simplified - ignores OCV(SOC) curvature)
Ohmic-resistance-style quantity, computable directly and instantly from
the SAME raw cycle's own unlabeled V_t/I_t channels, no SOH label
involved. A small auxiliary head is trained (JOINTLY with the SOH head,
during normal supervised training on the TRAIN pool) to regress
R_apparent from the shared encoder's embedding - this is the physics-
consistency task TTT continues to optimize, unsupervised, at test time.

TTT LOOP (per held-out battery, cycles processed IN ORDER, weights
carried forward across cycles within that battery - "continual"
adaptation, matching the paper's own framing): for each new cycle,
ONE gradient step (a common, disclosed TTT design point - not the
paper's own literal per-step count, which is not stated in the
abstract) on the ENCODER + R-head ONLY (the SOH head is frozen at test
time - adapting it with no SOH signal at all would let it drift with no
grounding signal, so only the shared feature extractor is adapted, the
standard TTT design choice), using ONLY the physics R_apparent MSE loss
- never sees or uses the target battery's own SOH label at any point.
After the step, the (adapted) encoder + (frozen) SOH head produce that
cycle's SOH prediction.

DISCLOSED COMPUTE-BUDGET SUBSAMPLING: HUST (77 cells, ~146k cycles) and
XJTU (47 cells, ~19k cycles) are subsampled to every 5th cycle PER
BATTERY for the TTT loop (still a continuous, in-order adaptation
sequence, just sparser) - a real compute-budget necessity for a
per-cycle gradient loop on CPU within this run's time budget, the same
class of disclosed subsampling this project has used before (Stage 7
closeout's own stratified HUST 8-cell subsample). CALCE/Oxford/in-domain
use every cycle, unsubsampled.
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
from sequence_features import apply_channel_norm
from models.world_model import CycleCNNEncoder

SEED = 42
DEVICE = torch.device("cpu")
EMBED_DIM = 32
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-3
PHYSICS_LOSS_WEIGHT = 0.3
TTT_LR = 1e-4
TTT_SUBSAMPLE_EVERY = {"HUST": 5, "XJTU": 5, "CALCE": 1, "Oxford": 1}

ROUTED_BASELINE = {"in-domain (fixed split)": 0.974, "CALCE": 0.740, "Oxford": 0.953, "HUST": 0.800, "XJTU": -1.062}


class TTTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CycleCNNEncoder(in_channels=6, embed_dim=EMBED_DIM)
        self.soh_head = nn.Sequential(nn.Linear(EMBED_DIM, 16), nn.ReLU(), nn.Linear(16, 1))
        self.r_head = nn.Sequential(nn.Linear(EMBED_DIM, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        z = self.encoder(x)
        return self.soh_head(z).squeeze(-1), self.r_head(z).squeeze(-1)


def r_apparent(X_raw: np.ndarray) -> np.ndarray:
    """X_raw: (n, 200, 6) UN-normalized. Channel 0=V_t, 1=I_t."""
    v_drop = X_raw[:, 0, 0] - X_raw[:, -1, 0]
    i_mean = np.mean(np.abs(X_raw[:, :, 1]), axis=1) + 1e-6
    return v_drop / i_mean


def build_flat_pool(pool: dict, pool_raw: dict):
    """Concatenate all batteries' cycles into flat arrays, also returning
    the physics target computed from the RAW (pre-normalization) tensor."""
    Xs, sohs, rs, bids = [], [], [], []
    for bid in pool:
        X_norm, soh, rul = pool[bid]
        X_raw, _, _ = pool_raw[bid]
        Xs.append(X_norm); sohs.append(soh); rs.append(r_apparent(X_raw))
        bids += [bid] * len(soh)
    return np.concatenate(Xs), np.concatenate(sohs), np.concatenate(rs), bids


def train_supervised(train_pool_norm, train_pool_raw, soh_mean, soh_std, r_mean, r_std):
    torch.manual_seed(SEED); np.random.seed(SEED)
    X, soh, r, _ = build_flat_pool(train_pool_norm, train_pool_raw)
    soh_z = (soh - soh_mean) / soh_std
    r_z = (r - r_mean) / r_std
    Xt, soht, rt = torch.tensor(X, dtype=torch.float32), torch.tensor(soh_z, dtype=torch.float32), torch.tensor(r_z, dtype=torch.float32)

    model = TTTModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(Xt)
    rng = np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        idx = rng.permutation(n)
        losses = []
        for s in range(0, n, BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            soh_pred, r_pred = model(Xt[b])
            loss = nn.functional.mse_loss(soh_pred, soht[b]) + PHYSICS_LOSS_WEIGHT * nn.functional.mse_loss(r_pred, rt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[item2] supervised pretrain epoch {epoch+1}/{EPOCHS}: mean_loss={np.mean(losses):.4f}")
        if not np.isfinite(losses[-1]):
            print("[item2] NON-CONVERGENCE: loss non-finite - stopping early.")
            break
    return model


@torch.no_grad()
def predict_soh(model, X_batch, soh_mean, soh_std):
    soh_pred_z, _ = model(torch.tensor(X_batch, dtype=torch.float32))
    return soh_pred_z.numpy() * soh_std + soh_mean


def run_zero_shot(model, pool_norm: dict, soh_mean, soh_std):
    all_pred, all_true = [], []
    for bid, (X, soh, rul) in pool_norm.items():
        pred = predict_soh(model, X, soh_mean, soh_std)
        all_pred.append(pred); all_true.append(soh)
    return np.concatenate(all_pred), np.concatenate(all_true)


def run_ttt(base_model: TTTModel, pool_norm: dict, pool_raw: dict, soh_mean, soh_std, r_mean, r_std,
            subsample_every: int):
    all_pred, all_true = [], []
    for bid in pool_norm:
        X_norm, soh, rul = pool_norm[bid]
        X_raw, _, _ = pool_raw[bid]
        order = np.arange(len(soh))
        if subsample_every > 1:
            order = order[::subsample_every]
        r_targets = r_apparent(X_raw)
        r_targets_z = (r_targets - r_mean) / r_std

        model = TTTModel().to(DEVICE)
        model.load_state_dict(base_model.state_dict())
        opt = torch.optim.Adam([p for p in model.encoder.parameters()] + [p for p in model.r_head.parameters()], lr=TTT_LR)
        model.soh_head.eval()
        for p in model.soh_head.parameters():
            p.requires_grad_(False)

        preds = np.full(len(soh), np.nan)
        for i in order:
            x_i = torch.tensor(X_norm[i:i + 1], dtype=torch.float32)
            r_i = torch.tensor([r_targets_z[i]], dtype=torch.float32)

            # ONE self-supervised physics gradient step (no label used)
            model.train()
            _, r_pred = model(x_i)
            phys_loss = nn.functional.mse_loss(r_pred, r_i)
            if torch.isfinite(phys_loss):
                opt.zero_grad(); phys_loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                soh_pred_z, _ = model(x_i)
            preds[i] = float(soh_pred_z.item()) * soh_std + soh_mean

        mask = order
        all_pred.append(preds[mask]); all_true.append(soh[mask])
    return np.concatenate(all_pred), np.concatenate(all_true)


def score(y_true, pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred)),
            "r2": float(r2_score(y_true, pred)), "n": int(len(y_true))}


def main():
    t0 = time.time()
    print("=== Part A, item 2: physics-guided test-time training (TTT) ===")

    print("[item2] loading canonical 42-battery pool (raw tensors)...")
    train_raw, test_raw = load_pool_train_test()
    stats = fit_norm_stats(train_raw)
    train_norm = norm_pool(train_raw, stats)
    test_norm = norm_pool(test_raw, stats)

    X_tr, soh_tr, r_tr, _ = build_flat_pool(train_norm, train_raw)
    soh_mean, soh_std = float(soh_tr.mean()), float(soh_tr.std() + 1e-8)
    r_mean, r_std = float(r_tr.mean()), float(r_tr.std() + 1e-8)
    print(f"[item2] soh_mean={soh_mean:.3f} soh_std={soh_std:.3f} | r_apparent_mean={r_mean:.4f} r_apparent_std={r_std:.4f}")

    print("\n[item2] supervised training (SOH head + physics R-head, TRAIN pool only)...")
    model = train_supervised(train_norm, train_raw, soh_mean, soh_std, r_mean, r_std)
    torch.save(model.state_dict(), MODEL_DIR / "_experimental_ttt_physics_encoder.pt")

    results = []
    print("\n[item2] in-domain (TEST split): zero-shot control (no TTT, held-out battery IDs but same distribution)...")
    pred_zs, true_zs = run_zero_shot(model, test_norm, soh_mean, soh_std)
    r_zs = score(true_zs, pred_zs)
    print(f"[item2] in-domain zero-shot (same small architecture, no TTT): R2={r_zs['r2']:.4f} RMSE={r_zs['rmse']:.4f} n={r_zs['n']}")
    results.append({"dataset": "in-domain (fixed split)", "variant": "zero_shot_same_arch", **r_zs,
                     "routed_baseline_r2": ROUTED_BASELINE["in-domain (fixed split)"]})

    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        print(f"\n[item2] loading {name} (raw cycles, zero-retrain)...")
        held_raw = load_heldout(name)
        if not held_raw:
            print(f"[item2] WARNING: no usable {name} data, skipping")
            continue
        held_norm = norm_pool(held_raw, stats)

        pred_zs, true_zs = run_zero_shot(model, held_norm, soh_mean, soh_std)
        r_zs = score(true_zs, pred_zs)
        print(f"[item2] {name} zero-shot (same small architecture, no TTT): R2={r_zs['r2']:.4f} RMSE={r_zs['rmse']:.4f} n={r_zs['n']}")
        results.append({"dataset": name, "variant": "zero_shot_same_arch", **r_zs,
                         "routed_baseline_r2": ROUTED_BASELINE[name]})

        sub = TTT_SUBSAMPLE_EVERY.get(name, 1)
        t_ttt0 = time.time()
        pred_ttt, true_ttt = run_ttt(model, held_norm, held_raw, soh_mean, soh_std, r_mean, r_std, sub)
        r_ttt = score(true_ttt, pred_ttt)
        print(f"[item2] {name} PHYSICS-GUIDED TTT (subsample_every={sub}, {len(true_ttt)} cycles adapted over "
              f"{(time.time()-t_ttt0)/60:.2f} min): R2={r_ttt['r2']:.4f} RMSE={r_ttt['rmse']:.4f} n={r_ttt['n']} "
              f"| vs zero-shot control R2={r_zs['r2']:.4f} -> {'TTT HELPS' if r_ttt['r2']>r_zs['r2'] else 'TTT HURTS/NO HELP'} "
              f"| vs routed deployed baseline {ROUTED_BASELINE[name]:.3f} -> "
              f"{'WIN' if r_ttt['r2']>ROUTED_BASELINE[name] else 'LOSS'}")
        results.append({"dataset": name, "variant": "physics_ttt", **r_ttt,
                         "routed_baseline_r2": ROUTED_BASELINE[name], "subsample_every": sub})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass_partA_item2_physics_ttt.csv", index=False)
    print("\n=== Item 2 (physics-guided TTT) FULL RESULTS ===")
    print(results_df.to_string(index=False))
    print(f"\n[item2] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
