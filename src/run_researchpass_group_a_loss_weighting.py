"""
Research pass Group A, items 2-3: CoV (Coefficient-of-Variation) and
EMA-based loss weighting for the joint SOH+RUL model, as alternatives
to the deployed AdaptiveLossWeighting (learnable homoscedastic-
uncertainty, log-sigma-clamped) scheme.

Reuses run_stage1_followup_partB_joint_rul.py's EXACT data pipeline,
architecture (JointSOHRULModelFusion), split, epoch budget, and
optimizer settings unchanged - the ONLY thing that differs between
this run and the deployed model's own training run is which loss-
weighting module computes (total_loss, alpha, beta) from
(loss_soh, loss_rul) each step. Baseline to beat (already on record,
not re-run here): SOH R2=0.9241, RUL R2=0.6657
(outputs/stage1_followup_partB_joint_rul_results.csv).

CoV-Weighting (Groenendijk et al. 2020 "Multi-Loss Weighting with
Coefficient of Variation", reimplemented from the paper's own
description - the paper's own code was not consulted, a disclosed
adaptation not a byte-exact port, same standing as this project's
other from-description implementations e.g. BatLiNet): per task i,
tracks the loss RATIO l_hat_i(t) = L_i(t) / L_i(t-1) (step-to-step,
not vs. the very first step, to stay responsive throughout a 25-epoch
run rather than anchored to one potentially-noisy initial value) via
running mean/std (Welford's online algorithm, exact not EMA-
approximated), CoV_i = std_i / mean_i, and weights w_i =
CoV_i / mean(CoV) (so weights average to 1 across tasks, the paper's
own convention) - a task whose relative loss improvement is more
ERRATIC gets weighted MORE, the paper's own stated mechanism for
directing attention toward tasks that aren't yet stably converging.

EMA-based balancing (simpler, disclosed as such, not attributed to a
specific paper): maintains an exponential moving average of each
task's OWN loss scale, weights each task inversely to its own EMA so
every task contributes on a comparable O(1) scale regardless of its
natural loss magnitude (SOH and RUL losses are both z-scored already,
so this mostly corrects for how FAST each task's own loss is moving,
not raw scale) - decay=0.9, a fixed, untuned choice, same standing as
this project's own session 4 lambda=0.1 precedent.
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
from stage1_common import canonical_feature_cols, add_reformulated_duration_features
from run_stage1_followup_partB_joint_rul import JointSOHRULModelFusion, standardize, build_hi_features

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
EPOCHS = 25
BATCH_SIZE = 64
BASELINE = {"soh_r2": 0.9241040945053101, "rul_r2": 0.6656929850578308}


class CoVLossWeighting(nn.Module):
    """No learnable parameters - weights are computed deterministically
    from the running loss-ratio statistics, per Groenendijk et al.'s
    own design (their method is explicitly NOT gradient-learned)."""

    def __init__(self, n_tasks: int = 2, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.prev_loss = [None] * n_tasks
        # Welford's online mean/variance per task, over the loss-ratio stream
        self._count = [0] * n_tasks
        self._mean = [0.0] * n_tasks
        self._m2 = [0.0] * n_tasks

    def _update(self, i, ratio):
        self._count[i] += 1
        delta = ratio - self._mean[i]
        self._mean[i] += delta / self._count[i]
        delta2 = ratio - self._mean[i]
        self._m2[i] += delta * delta2

    def _cov(self, i):
        if self._count[i] < 2 or abs(self._mean[i]) < self.eps:
            return 1.0  # no history yet - neutral weight
        var = self._m2[i] / self._count[i]
        return float(np.sqrt(max(var, 0.0)) / abs(self._mean[i]))

    def forward(self, loss_soh, loss_rul):
        losses = [loss_soh, loss_rul]
        covs = []
        for i, l in enumerate(losses):
            l_val = float(l.detach().item())
            if self.prev_loss[i] is not None and abs(self.prev_loss[i]) > self.eps:
                ratio = l_val / self.prev_loss[i]
                self._update(i, ratio)
            self.prev_loss[i] = l_val
            covs.append(self._cov(i))
        mean_cov = float(np.mean(covs)) if np.mean(covs) > self.eps else 1.0
        weights = [c / mean_cov for c in covs]
        total = weights[0] * losses[0] + weights[1] * losses[1]
        return total, weights[0], weights[1]


class EMALossWeighting(nn.Module):
    def __init__(self, decay: float = 0.9, eps: float = 1e-8):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.ema = [None, None]

    def forward(self, loss_soh, loss_rul):
        losses = [loss_soh, loss_rul]
        for i, l in enumerate(losses):
            l_val = float(l.detach().item())
            self.ema[i] = l_val if self.ema[i] is None else self.decay * self.ema[i] + (1 - self.decay) * l_val
        inv = [1.0 / (e + self.eps) for e in self.ema]
        mean_inv = float(np.mean(inv))
        weights = [w / mean_inv for w in inv]
        total = weights[0] * losses[0] + weights[1] * losses[1]
        return total, weights[0], weights[1]


def train_variant(name, weighting_module, X_fit, HI_fit, soh_fit_z, rul_fit_z,
                   X_val, HI_val, soh_val_z, rul_val_z, X_test, HI_test,
                   soh_test, rul_test, soh_mean, soh_std, rul_mean, rul_std, n_hi_features):
    t0 = time.time()
    torch.manual_seed(42)
    model = JointSOHRULModelFusion(n_hi_features=n_hi_features)
    params = list(model.parameters())
    if isinstance(weighting_module, nn.Module) and len(list(weighting_module.parameters())) > 0:
        params += list(weighting_module.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    mse = nn.MSELoss()

    Xt, HIt = torch.tensor(X_fit), torch.tensor(HI_fit)
    soh_t = torch.tensor(soh_fit_z).unsqueeze(-1)
    rul_t = torch.tensor(rul_fit_z).unsqueeze(-1)
    Xv, HIv = torch.tensor(X_val), torch.tensor(HI_val)
    soh_v = torch.tensor(soh_val_z).unsqueeze(-1)
    rul_v = torch.tensor(rul_val_z).unsqueeze(-1)

    n = len(Xt)
    history = []
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            pred_soh, pred_rul = model(Xt[idx], HIt[idx])
            l_soh = mse(pred_soh, soh_t[idx])
            l_rul = mse(pred_rul, rul_t[idx])
            total, a, b = weighting_module(l_soh, l_rul)
            total.backward()
            opt.step()
            ep_loss += float(total.item()) * len(idx)
        ep_loss /= n
        model.eval()
        with torch.no_grad():
            vp_soh, vp_rul = model(Xv, HIv)
            v_l_soh = mse(vp_soh, soh_v).item()
            v_l_rul = mse(vp_rul, rul_v).item()
        history.append({"epoch": epoch, "train_loss": ep_loss, "val_loss_soh": v_l_soh,
                        "val_loss_rul": v_l_rul, "alpha": float(a), "beta": float(b)})
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"[{name}] epoch {epoch:2d} train={ep_loss:.4f} val_soh={v_l_soh:.4f} "
                  f"val_rul={v_l_rul:.4f} alpha={float(a):.3f} beta={float(b):.3f}")

    model.eval()
    with torch.no_grad():
        pred_soh_z, pred_rul_z = model(torch.tensor(X_test), torch.tensor(HI_test))
    pred_soh = pred_soh_z.squeeze(-1).numpy() * soh_std + soh_mean
    pred_rul = pred_rul_z.squeeze(-1).numpy() * rul_std + rul_mean

    def m(y_true, y_pred):
        return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}
    soh_metrics = m(soh_test, pred_soh)
    rul_metrics = m(rul_test, pred_rul)
    elapsed = time.time() - t0
    print(f"[{name}] TEST SOH: {soh_metrics}")
    print(f"[{name}] TEST RUL: {rul_metrics}")
    print(f"[{name}] elapsed: {elapsed/60:.1f} min")
    return soh_metrics, rul_metrics, history, elapsed


def main():
    t0 = time.time()
    print("=== Research pass Group A, items 2-3: CoV and EMA loss weighting ===")
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[researchA23] fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)} batteries")

    X_fit, soh_fit, rul_fit, _, bid_fit, cyc_fit = make_xy(battery_data, fit_ids)
    X_val, soh_val, rul_val, _, bid_val, cyc_val = make_xy(battery_data, val_ids)
    X_test, soh_test, rul_test, _, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[researchA23] fit={len(X_fit)} val={len(X_val)} test={len(X_test)} cycles")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_reformulated = add_reformulated_duration_features(hi_full)
    feature_cols = canonical_feature_cols(reformulated=True)

    fit_key_df = pd.DataFrame({"battery_id": bid_fit, "cycle_idx": cyc_fit}).merge(
        hi_reformulated[["battery_id", "cycle_idx"] + feature_cols], on=["battery_id", "cycle_idx"], how="left")
    train_medians = fit_key_df[feature_cols].median(numeric_only=True).to_numpy()

    HI_fit = build_hi_features(bid_fit, cyc_fit, hi_reformulated, feature_cols, train_medians)
    HI_val = build_hi_features(bid_val, cyc_val, hi_reformulated, feature_cols, train_medians)
    HI_test = build_hi_features(bid_test, cyc_test, hi_reformulated, feature_cols, train_medians)
    hi_mean, hi_std = HI_fit.mean(axis=0), HI_fit.std(axis=0) + 1e-8
    HI_fit = (HI_fit - hi_mean) / hi_std
    HI_val = (HI_val - hi_mean) / hi_std
    HI_test = (HI_test - hi_mean) / hi_std

    soh_fit_z, soh_mean, soh_std = standardize(soh_fit)
    rul_fit_z, rul_mean, rul_std = standardize(rul_fit)
    soh_val_z = (soh_val - soh_mean) / soh_std
    rul_val_z = (rul_val - rul_mean) / rul_std

    results = []
    for name, weighting in [("CoV-weighting", CoVLossWeighting()), ("EMA-weighting", EMALossWeighting(decay=0.9))]:
        soh_m, rul_m, history, elapsed = train_variant(
            name, weighting, X_fit, HI_fit, soh_fit_z, rul_fit_z, X_val, HI_val, soh_val_z, rul_val_z,
            X_test, HI_test, soh_test, rul_test, soh_mean, soh_std, rul_mean, rul_std, len(feature_cols))
        results.append({"variant": name, "soh_r2": soh_m["r2"], "soh_rmse": soh_m["rmse"],
                        "rul_r2": rul_m["r2"], "rul_rmse": rul_m["rmse"], "elapsed_min": elapsed / 60})
        pd.DataFrame(history).to_csv(PROC_DIR / "predictions" / f"joint_{name.split('-')[0].lower()}_history.csv", index=False)

    results.append({"variant": "adaptive-clamped (DEPLOYED, on record)", "soh_r2": BASELINE["soh_r2"],
                    "soh_rmse": None, "rul_r2": BASELINE["rul_r2"], "rul_rmse": None, "elapsed_min": None})
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass_groupA23_loss_weighting.csv", index=False)
    print("\n=== SUMMARY: SOH R2 / RUL R2, all 3 methods ===")
    print(results_df[["variant", "soh_r2", "rul_r2", "elapsed_min"]].to_string(index=False))

    for r in results[:2]:
        soh_verdict = "WIN" if r["soh_r2"] > BASELINE["soh_r2"] else "LOSS"
        rul_verdict = "WIN" if r["rul_r2"] > BASELINE["rul_r2"] else "LOSS"
        print(f"\n[researchA23] {r['variant']} vs. deployed (SOH R2={BASELINE['soh_r2']:.4f}, "
              f"RUL R2={BASELINE['rul_r2']:.4f}): SOH {soh_verdict} ({r['soh_r2']:.4f}), "
              f"RUL {rul_verdict} ({r['rul_r2']:.4f})")

    print(f"\n[researchA23] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
