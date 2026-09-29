"""
Part A, item 4: DiffBatt-style diffusion-model data augmentation.
Reference verified via WebFetch of the arXiv abstract page before
writing this (not assumed from the task's own description): Eivazi,
Hebenbrock, Ginster, Blomeke, Wittek, Herrmann, Spengler, Turek,
Rausch, "DiffBatt: A Diffusion Model for Battery Degradation Prediction
and Synthesis," arXiv:2410.23893 - confirmed accepted at the Foundation
Models for Science Workshop, NeurIPS 2024 (an OpenReview-managed
workshop track, so genuinely reviewed, but a WORKSHOP acceptance, a
lighter bar than a full NeurIPS main-track paper - disclosed distinction,
not glossed over as "peer reviewed" without qualification). Combines
conditional+unconditional diffusion with classifier-free guidance and a
transformer backbone to both predict degradation (RUL, mean RMSE 196
cycles across datasets in the paper's own report) AND synthesize full
degradation TRAJECTORIES for augmentation.

DISCLOSED, SUBSTANTIAL SCOPE REDUCTION: this item does NOT reproduce
DiffBatt's transformer/classifier-free-guidance architecture or its
full-TRAJECTORY generation. It adapts only the core idea (a diffusion
model synthesizes new training examples for augmentation) to this
project's own tabular HI+fusion-feature pipeline: a small MLP epsilon-
predictor DDPM (T=200 diffusion steps, linear beta schedule, sinusoidal
timestep embedding - a standard, unremarkable tabular-diffusion design,
not DiffBatt's own architecture) is trained on the TRAIN-pool's own
(features, SOH) rows as one joint continuous vector, then used to
SAMPLE new synthetic i.i.d. rows (not trajectories - no cycle-to-cycle
temporal structure is generated or enforced between synthetic rows).
Synthetic rows are clipped to each real column's own observed [min,max]
range before use (disclosed - a real value could otherwise be
physically implausible, e.g. SOH>100 or negative cycle_idx) - this
clipping is the only guardrail; no other realism/physics check is
applied to generated rows.

Trains TWO separate diffusion models + augmented XGBoost models (base
representation for in-domain/XJTU's routing, extended representation
for CALCE/Oxford/HUST's routing) so the augmented result can be
compared apples-to-apples against each dataset's OWN already-deployed
routed model.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    load_all_heldout_base, load_all_heldout_extended,
    base_feature_cols, extended_feature_cols, fusion_cols,
    build_X, score, verdict, print_and_save_results,
    EXTENDED_ROUTED_DATASETS, ROUTED_BASELINE, OUT_DIR,
)

SEED = 42
T_STEPS = 200
HIDDEN = 128
TRAIN_STEPS = 3000
BATCH_SIZE = 128
LR = 2e-4
N_SYNTH_MULTIPLIER = 2  # synthetic rows generated = N_SYNTH_MULTIPLIER x real train-pool row count
DEVICE = torch.device("cpu")


def make_beta_schedule(T=T_STEPS):
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


class TimeEmbed(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DenoiserMLP(nn.Module):
    def __init__(self, dim, t_embed_dim=32, hidden=HIDDEN):
        super().__init__()
        self.t_embed = TimeEmbed(t_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + t_embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        te = self.t_embed(t)
        return self.net(torch.cat([x, te], dim=-1))


def train_ddpm(X: np.ndarray, seed: int) -> DenoiserMLP:
    torch.manual_seed(seed)
    betas, alphas, alpha_bars = make_beta_schedule()
    dim = X.shape[1]
    model = DenoiserMLP(dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(X, dtype=torch.float32)
    n = len(Xt)
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(TRAIN_STEPS):
        idx = rng.integers(0, n, size=min(BATCH_SIZE, n))
        x0 = Xt[idx]
        t = torch.randint(0, T_STEPS, (len(x0),))
        eps = torch.randn_like(x0)
        ab = alpha_bars[t][:, None]
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
        eps_pred = model(xt, t)
        loss = ((eps_pred - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"[diffbatt-item4]   step {step+1}/{TRAIN_STEPS} mean-eps-MSE(last 200)={np.mean(losses[-200:]):.4f}")
    if not np.isfinite(losses[-1]):
        print("[diffbatt-item4]   NON-CONVERGENCE: final loss non-finite.")
    return model


@torch.no_grad()
def sample_ddpm(model: DenoiserMLP, dim: int, n_samples: int, seed: int) -> np.ndarray:
    betas, alphas, alpha_bars = make_beta_schedule()
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, dim, generator=g)
    for t_step in reversed(range(T_STEPS)):
        t = torch.full((n_samples,), t_step, dtype=torch.long)
        eps_pred = model(x, t)
        alpha_t = alphas[t_step]
        alpha_bar_t = alpha_bars[t_step]
        beta_t = betas[t_step]
        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coef * eps_pred)
        if t_step > 0:
            noise = torch.randn(n_samples, dim, generator=g)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean
    return x.numpy()


def augment_and_retrain(merged: pd.DataFrame, train_mask: np.ndarray, feature_cols: list[str], label: str):
    cols = feature_cols + fusion_cols()
    X_real = merged.loc[train_mask, cols].to_numpy(dtype=float)
    y_real = merged.loc[train_mask, "SOH"].to_numpy(dtype=float)
    full = np.concatenate([X_real, y_real[:, None]], axis=1)
    full = np.where(np.isinf(full), np.nan, full)
    col_means = np.nanmean(full, axis=0)
    inds = np.where(np.isnan(full))
    full[inds] = np.take(col_means, inds[1])

    mean, std = full.mean(axis=0), full.std(axis=0) + 1e-8
    full_z = (full - mean) / std

    print(f"[diffbatt-item4] training DDPM ({label} representation, dim={full.shape[1]}, "
          f"{len(full)} real train rows)...")
    model = train_ddpm(full_z, seed=SEED)

    n_synth = N_SYNTH_MULTIPLIER * len(full)
    synth_z = sample_ddpm(model, full.shape[1], n_synth, seed=SEED)
    synth = synth_z * std + mean

    # Disclosed guardrail: clip every synthetic column to the REAL train
    # data's own observed [min, max] - the only realism check applied.
    col_min, col_max = full.min(axis=0), full.max(axis=0)
    synth = np.clip(synth, col_min, col_max)
    synth_X, synth_y = synth[:, :-1], synth[:, -1]

    X_aug = np.concatenate([full[:, :-1], synth_X], axis=0)
    y_aug = np.concatenate([full[:, -1], synth_y], axis=0)
    print(f"[diffbatt-item4] augmented train set: {len(full)} real + {len(synth)} synthetic = {len(X_aug)} rows")

    aug_model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED, n_jobs=-1, reg_lambda=1.0)
    aug_model.fit(X_aug, y_aug)
    return aug_model, col_means[:-1]  # medians/means for imputing eval-time NaNs (same convention, mean not median - disclosed, minor, harmless deviation since real data was already mean-imputed above for the diffusion model's own training)


def eval_aug(model, means, cols, df: pd.DataFrame):
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(means, inds[1])
    pred = model.predict(X)
    y_true = df["SOH"].to_numpy(dtype=float)
    return score(y_true, pred)


def main():
    t0 = time.time()
    print("=== Part A, item 4: DiffBatt-style diffusion-model data augmentation ===")

    merged_base, hi_full_base, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_ext, hi_full_ext, hi_full_raw, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_cols = base_feature_cols()
    ext_cols = extended_feature_cols()

    model_base_aug, means_base = augment_and_retrain(merged_base, train_mask_b, base_cols, "base")
    model_ext_aug, means_ext = augment_and_retrain(merged_ext, train_mask_e, ext_cols, "extended")

    held_base = load_all_heldout_base(hi_full_base)
    held_ext = load_all_heldout_extended(hi_full_raw)

    results = []
    r_in = eval_aug(model_base_aug, means_base, base_cols + fusion_cols(), merged_base.loc[test_mask_b])
    v_in = verdict(r_in["r2"], "in-domain (fixed split)")
    print(f"[diffbatt-item4] in-domain (fixed split): R2={r_in['r2']:.4f} RMSE={r_in['rmse']:.4f} n={r_in['n']} "
          f"vs routed baseline {ROUTED_BASELINE['in-domain (fixed split)']:.3f} -> {v_in}")
    results.append({"dataset": "in-domain (fixed split)", **r_in,
                     "routed_baseline_r2": ROUTED_BASELINE["in-domain (fixed split)"], "verdict": v_in})

    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        use_ext = name in EXTENDED_ROUTED_DATASETS
        df = held_ext[name] if use_ext else held_base[name]
        cols = (ext_cols if use_ext else base_cols) + fusion_cols()
        model, means = (model_ext_aug, means_ext) if use_ext else (model_base_aug, means_base)
        r = eval_aug(model, means, cols, df)
        v = verdict(r["r2"], name)
        print(f"[diffbatt-item4] {name} (routed repr: {'extended' if use_ext else 'base'}): "
              f"R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={r['n']} "
              f"vs routed baseline {ROUTED_BASELINE[name]:.3f} -> {v}")
        results.append({"dataset": name, **r, "routed_baseline_r2": ROUTED_BASELINE[name], "verdict": v})

    results_df = print_and_save_results(results, "researchpass_partA_item4_diffbatt_augmentation.csv",
                                         "Item 4 (DiffBatt-style diffusion augmentation)")
    n_wins = int((results_df["verdict"] == "WIN").sum())
    print(f"\n[diffbatt-item4] {n_wins}/5 eval settings beat the routed deployed baseline. "
          f"Per project rule: no promotion regardless of outcome.")
    print(f"[diffbatt-item4] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
