"""
Phase 4: joint SOH+RUL prediction ablation.

4 variants of the SAME JointSOHRULModel backbone, differing only in how
the two task losses are combined:
    fixed_balanced : total = 0.5*L_soh + 0.5*L_rul            (fixed weights)
    soh_only       : total = 1.0*L_soh + 0.0*L_rul            (RUL head never
                                                                 gets gradient)
    rul_only       : total = 0.0*L_soh + 1.0*L_rul            (SOH head never
                                                                 gets gradient)
    adaptive       : learnable homoscedastic-uncertainty weighting
                     (see models/joint_model.py docstring for why this is
                     the correct way to make alpha/beta "trained
                     parameters" without them collapsing to zero)

Both SOH and RUL heads are ALWAYS evaluated on the test set for every
variant, specifically so soh_only/rul_only visibly "collapse" on the task
they never trained (their head stays near random init -> near-constant
output -> large RMSE / near-zero or negative R2). That collapse is the
point of the ablation, not a bug to fix.

Epoch budget: 25 epochs x 4 variants (100 total training epochs across
the ablation) — reduced from Phase 2's 40 per the task's own "if a step
is taking too long, use fewer epochs, log it" allowance, since this is a
4x-cost ablation stacked after already-long Phase 2 deep training. Logged
here, not hidden.
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
from models.joint_model import JointSOHRULModel, AdaptiveLossWeighting

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
(PROC_DIR / "predictions").mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

EPOCHS = 25
BATCH_SIZE = 64


def standardize(arr):
    mean, std = float(arr.mean()), float(arr.std() + 1e-8)
    return (arr - mean) / std, mean, std


def train_variant(mode: str, X_fit, soh_fit, rul_fit, X_val, soh_val, rul_val):
    model = JointSOHRULModel()
    soh_fit_z, soh_mean, soh_std = standardize(soh_fit)
    rul_fit_z, rul_mean, rul_std = standardize(rul_fit)
    soh_val_z = (soh_val - soh_mean) / soh_std
    rul_val_z = (rul_val - rul_mean) / rul_std

    adaptive = AdaptiveLossWeighting() if mode == "adaptive" else None
    params = list(model.parameters()) + (list(adaptive.parameters()) if adaptive else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    mse = nn.MSELoss()

    Xt = torch.tensor(X_fit)
    soh_t = torch.tensor(soh_fit_z).unsqueeze(-1)
    rul_t = torch.tensor(rul_fit_z).unsqueeze(-1)
    Xv = torch.tensor(X_val)
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
            pred_soh, pred_rul = model(Xt[idx])
            l_soh = mse(pred_soh, soh_t[idx])
            l_rul = mse(pred_rul, rul_t[idx])

            if mode == "fixed_balanced":
                total = 0.5 * l_soh + 0.5 * l_rul
                a, b = 0.5, 0.5
            elif mode == "soh_only":
                total = 1.0 * l_soh + 0.0 * l_rul
                a, b = 1.0, 0.0
            elif mode == "rul_only":
                total = 0.0 * l_soh + 1.0 * l_rul
                a, b = 0.0, 1.0
            elif mode == "adaptive":
                total, a, b = adaptive(l_soh, l_rul)
            else:
                raise ValueError(mode)

            total.backward()
            opt.step()
            if mode == "adaptive":
                # Belt-and-suspenders on top of the in-forward clamp: Adam's
                # momentum can still nudge the raw parameter slightly past
                # the clamp boundary even when the clamped-value gradient is
                # exactly zero there (momentum carries over from prior
                # steps). Clamping the raw parameter in-place after every
                # optimizer step guarantees it never actually exceeds the
                # bound, not just the value used inside the loss.
                with torch.no_grad():
                    adaptive.log_sigma_soh.clamp_(-adaptive.LOG_SIGMA_CLAMP, adaptive.LOG_SIGMA_CLAMP)
                    adaptive.log_sigma_rul.clamp_(-adaptive.LOG_SIGMA_CLAMP, adaptive.LOG_SIGMA_CLAMP)
            ep_loss += total.item() * len(idx)
        ep_loss /= n

        model.eval()
        with torch.no_grad():
            vp_soh, vp_rul = model(Xv)
            v_l_soh = mse(vp_soh, soh_v).item()
            v_l_rul = mse(vp_rul, rul_v).item()
        history.append({"epoch": epoch, "train_loss": ep_loss,
                         "val_loss_soh": v_l_soh, "val_loss_rul": v_l_rul,
                         "alpha": a, "beta": b})
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"[joint/{mode}] epoch {epoch:2d} train={ep_loss:.4f} "
                  f"val_soh={v_l_soh:.4f} val_rul={v_l_rul:.4f} alpha={a:.3f} beta={b:.3f}")

    return model, (soh_mean, soh_std), (rul_mean, rul_std), history


def evaluate(model, X_test, soh_test, rul_test, soh_scale, rul_scale):
    soh_mean, soh_std = soh_scale
    rul_mean, rul_std = rul_scale
    model.eval()
    with torch.no_grad():
        pred_soh_z, pred_rul_z = model(torch.tensor(X_test))
    pred_soh = pred_soh_z.squeeze(-1).numpy() * soh_std + soh_mean
    pred_rul = pred_rul_z.squeeze(-1).numpy() * rul_std + rul_mean

    def m(y_true, y_pred):
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    return m(soh_test, pred_soh), m(rul_test, pred_rul)


def main(modes: list[str] | None = None):
    """
    modes: subset of ["fixed_balanced","soh_only","rul_only","adaptive"]
    to (re-)run. Defaults to all 4. Results for re-run modes REPLACE their
    rows in the existing joint_ablation.csv/joint_ablation_history.csv
    (rather than overwriting the whole file), so e.g. re-running just
    "adaptive" after a fix doesn't require re-running the other 3
    variants that weren't touched by that fix.
    """
    modes = modes or ["fixed_balanced", "soh_only", "rul_only", "adaptive"]
    t0 = time.time()
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    X_fit, soh_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, soh_val, rul_val, _, _, _ = make_xy(battery_data, val_ids)
    X_test, soh_test, rul_test, _, _, _ = make_xy(battery_data, test_ids)
    print(f"[joint] fit={len(X_fit)} val={len(X_val)} test={len(X_test)} "
          f"(loaded in {time.time()-t0:.1f}s)")

    # Same per-channel normalization fix as train_deep_models.py (see
    # sequence_features.compute_channel_norm_stats docstring): the raw
    # dVdQ channel reaches ~9.5M, which broke CNN-LSTM's BatchNorm and
    # would equally break this joint model's identical backbone. Loads
    # the stats FILE saved by train_deep_models.py (same fit_ids battery
    # set, same computation) rather than recomputing, so both scripts use
    # an identical transform.
    stats_path = PROC_DIR / "channel_norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            "channel_norm_stats.json not found - run train_deep_models.py first "
            "(it computes and saves the normalization stats this script reuses)."
        )
    norm_stats = json.loads(stats_path.read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    rows = []
    all_histories = []
    for mode in modes:
        print(f"\n[joint] === training variant: {mode} ===")
        model, soh_scale, rul_scale, hist = train_variant(
            mode, X_fit, soh_fit, rul_fit, X_val, soh_val, rul_val
        )
        soh_metrics, rul_metrics = evaluate(model, X_test, soh_test, rul_test, soh_scale, rul_scale)
        print(f"[joint/{mode}] TEST SOH: {soh_metrics}")
        print(f"[joint/{mode}] TEST RUL: {rul_metrics}")
        rows.append({"variant": mode, "target": "SOH", **soh_metrics})
        rows.append({"variant": mode, "target": "RUL", **rul_metrics})
        for h in hist:
            h["variant"] = mode
        all_histories.extend(hist)
        torch.save(model.state_dict(), ROOT / "models" / f"joint_{mode}.pt")

    ablation_path = PROC_DIR / "predictions" / "joint_ablation.csv"
    history_path = PROC_DIR / "predictions" / "joint_ablation_history.csv"
    new_ablation_df = pd.DataFrame(rows)
    new_history_df = pd.DataFrame(all_histories)

    if set(modes) != {"fixed_balanced", "soh_only", "rul_only", "adaptive"} and ablation_path.exists():
        old_ablation_df = pd.read_csv(ablation_path)
        old_ablation_df = old_ablation_df[~old_ablation_df["variant"].isin(modes)]
        ablation_df = pd.concat([old_ablation_df, new_ablation_df], ignore_index=True)

        old_history_df = pd.read_csv(history_path)
        old_history_df = old_history_df[~old_history_df["variant"].isin(modes)]
        history_df = pd.concat([old_history_df, new_history_df], ignore_index=True)
        print(f"[joint] merged re-run of {modes} into existing ablation tables "
              f"(kept {old_ablation_df['variant'].unique().tolist()} from the prior run)")
    else:
        ablation_df = new_ablation_df
        history_df = new_history_df

    ablation_df.to_csv(ablation_path, index=False)
    history_df.to_csv(history_path, index=False)

    print("\n[joint] === ABLATION SUMMARY ===")
    print(ablation_df.pivot(index="variant", columns="target", values="rmse").to_string())
    print(f"[joint] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    # optional CLI: `python train_joint_adaptive.py adaptive` re-runs only
    # that variant and merges it into the existing ablation table.
    cli_modes = sys.argv[1:] or None
    main(cli_modes)
