"""
Stage 3, Item 3.2: Gradient-level multi-task loss balancing (GradNorm,
Chen et al. 2018; PCGrad, Yu et al. 2020) - a mechanistically different
family from all 3 prior LOSS-level weighting attempts (fixed 50/50,
Kendall homoscedastic/"adaptive-clamped", softmax-normalized), none of
which beat fixed 50/50 on SOH.

CONFIGURATION: `JointSOHRULModelFusion` (session 41 Part B's improved
joint architecture: 4-branch CNN+LSTM backbone, HI-feature-fusion heads)
- used rather than the original bare `JointSOHRULModel` because session
41 Part B already established this is the best-performing joint
architecture found in this project (SOH R2 0.344->0.924, RUL R2
0.432->0.666 over the original joint_adaptive model), so testing a NEW
loss-balancing mechanism on the OLD, already-known-worse architecture
would confound "did gradient-level balancing help" with "did we regress
to a worse backbone." Same 8 canonical (1.1-reformulated) HI features,
same original 32-battery pool, same 25-epoch budget, same train/val/test
split as session 41 Part B and its own baselines - so results below are
directly comparable to every number in the comparison table.

Both mechanics (GradNorm weight update, PCGrad gradient projection)
were smoke-tested on toy random data/labels BEFORE this script was
written to train on real data, per this project's own "verify before
trusting" standard for any new gradient-level mechanism - confirmed:
GradNorm's task weights move away from init and renormalize to sum=2
each step with no NaN; PCGrad's projection triggers correctly whenever
the two task gradients' dot product is negative and leaves both
gradients untouched otherwise, also with no NaN.

Checkpointed per-epoch per-method (two independent checkpoint files) so
a crash in either training run doesn't require restarting the other or
losing partial progress - same convention as every other long-running
training script in this project (run_stage1_followup_partB_joint_rul.py
etc).
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
from run_stage1_followup_partB_joint_rul import (
    JointSOHRULModelFusion, build_hi_features, standardize,
)

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

EPOCHS = 25
BATCH_SIZE = 64
GRADNORM_ALPHA = 1.5   # standard default from the GradNorm paper
GRADNORM_LR = 0.025    # standard default (weight LR >> network LR)
_CKPT_GRADNORM = PROC_DIR / "_gradnorm_epoch_checkpoint.pt"
_CKPT_PCGRAD = PROC_DIR / "_pcgrad_epoch_checkpoint.pt"


def m(y_true, y_pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}


def load_data():
    t0 = time.time()
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[stage3.2] fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)} batteries")

    X_fit, soh_fit, rul_fit, _, bid_fit, cyc_fit = make_xy(battery_data, fit_ids)
    X_val, soh_val, rul_val, _, bid_val, cyc_val = make_xy(battery_data, val_ids)
    X_test, soh_test, rul_test, _, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[stage3.2] fit={len(X_fit)} val={len(X_val)} test={len(X_test)} cycles "
          f"(loaded in {time.time()-t0:.1f}s)")

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

    data = dict(
        Xt=torch.tensor(X_fit), HIt=torch.tensor(HI_fit),
        soh_t=torch.tensor(soh_fit_z).unsqueeze(-1), rul_t=torch.tensor(rul_fit_z).unsqueeze(-1),
        Xv=torch.tensor(X_val), HIv=torch.tensor(HI_val),
        soh_v=torch.tensor(soh_val_z).unsqueeze(-1), rul_v=torch.tensor(rul_val_z).unsqueeze(-1),
        Xtest=torch.tensor(X_test), HItest=torch.tensor(HI_test),
        soh_test=soh_test, rul_test=rul_test,
        soh_mean=soh_mean, soh_std=soh_std, rul_mean=rul_mean, rul_std=rul_std,
        n_hi=len(feature_cols),
    )
    return data


def train_gradnorm(data):
    torch.manual_seed(42)
    model = JointSOHRULModelFusion(n_hi_features=data["n_hi"])
    mse = nn.MSELoss()
    w_soh = torch.nn.Parameter(torch.tensor(1.0))
    w_rul = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w_opt = torch.optim.Adam([w_soh, w_rul], lr=GRADNORM_LR)
    W = model.lstm.weight_hh_l0
    l0_soh, l0_rul = None, None

    Xt, HIt, soh_t, rul_t = data["Xt"], data["HIt"], data["soh_t"], data["rul_t"]
    Xv, HIv, soh_v, rul_v = data["Xv"], data["HIv"], data["soh_v"], data["rul_v"]
    n = len(Xt)

    start_epoch, history = 0, []
    if _CKPT_GRADNORM.exists():
        ckpt = torch.load(_CKPT_GRADNORM, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        w_opt.load_state_dict(ckpt["w_opt_state"])
        with torch.no_grad():
            w_soh.copy_(ckpt["w_soh"]); w_rul.copy_(ckpt["w_rul"])
        l0_soh, l0_rul = ckpt["l0_soh"], ckpt["l0_rul"]
        torch.set_rng_state(ckpt["torch_rng_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        print(f"[gradnorm] RESUMING from epoch {start_epoch} (checkpoint found)")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            pred_soh, pred_rul = model(Xt[idx], HIt[idx])
            l_soh = mse(pred_soh, soh_t[idx])
            l_rul = mse(pred_rul, rul_t[idx])
            if l0_soh is None:
                l0_soh, l0_rul = l_soh.detach().clone(), l_rul.detach().clone()

            weighted_loss = w_soh.detach() * l_soh + w_rul.detach() * l_rul
            opt.zero_grad()
            weighted_loss.backward(retain_graph=True)

            gw_soh = torch.autograd.grad(w_soh * l_soh, W, retain_graph=True, create_graph=True)[0]
            gw_rul = torch.autograd.grad(w_rul * l_rul, W, retain_graph=True, create_graph=True)[0]
            G_soh, G_rul = gw_soh.norm(), gw_rul.norm()
            G_avg = 0.5 * (G_soh + G_rul)
            ratio_soh = l_soh.detach() / l0_soh
            ratio_rul = l_rul.detach() / l0_rul
            r_avg = 0.5 * (ratio_soh + ratio_rul)
            target_soh = (G_avg * (ratio_soh / r_avg) ** GRADNORM_ALPHA).detach()
            target_rul = (G_avg * (ratio_rul / r_avg) ** GRADNORM_ALPHA).detach()
            loss_gradnorm = (G_soh - target_soh).abs() + (G_rul - target_rul).abs()

            w_opt.zero_grad()
            w_soh.grad, w_rul.grad = torch.autograd.grad(loss_gradnorm, [w_soh, w_rul])

            opt.step()
            w_opt.step()
            with torch.no_grad():
                w_soh.clamp_(min=1e-3)
                w_rul.clamp_(min=1e-3)
                coef = 2.0 / (w_soh + w_rul)
                w_soh.mul_(coef)
                w_rul.mul_(coef)

            ep_loss += weighted_loss.item() * len(idx)
        ep_loss /= n

        model.eval()
        with torch.no_grad():
            vp_soh, vp_rul = model(Xv, HIv)
            v_l_soh = mse(vp_soh, soh_v).item()
            v_l_rul = mse(vp_rul, rul_v).item()
        history.append({"epoch": epoch, "train_loss": ep_loss, "val_loss_soh": v_l_soh,
                         "val_loss_rul": v_l_rul, "w_soh": w_soh.item(), "w_rul": w_rul.item()})
        print(f"[gradnorm] epoch {epoch:2d} train={ep_loss:.4f} val_soh={v_l_soh:.4f} "
              f"val_rul={v_l_rul:.4f} w_soh={w_soh.item():.3f} w_rul={w_rul.item():.3f}")

        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                    "w_opt_state": w_opt.state_dict(), "w_soh": w_soh.detach(), "w_rul": w_rul.detach(),
                    "l0_soh": l0_soh, "l0_rul": l0_rul, "torch_rng_state": torch.get_rng_state(),
                    "epoch": epoch, "history": history}, _CKPT_GRADNORM)

    _CKPT_GRADNORM.unlink(missing_ok=True)
    return model, history


def train_pcgrad(data):
    torch.manual_seed(42)
    model = JointSOHRULModelFusion(n_hi_features=data["n_hi"])
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    shared_params = list(model.branches.parameters()) + list(model.lstm.parameters())
    head_params = list(model.soh_head.parameters()) + list(model.rul_head.parameters())

    Xt, HIt, soh_t, rul_t = data["Xt"], data["HIt"], data["soh_t"], data["rul_t"]
    Xv, HIv, soh_v, rul_v = data["Xv"], data["HIv"], data["soh_v"], data["rul_v"]
    n = len(Xt)

    start_epoch, history = 0, []
    if _CKPT_PCGRAD.exists():
        ckpt = torch.load(_CKPT_PCGRAD, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        print(f"[pcgrad] RESUMING from epoch {start_epoch} (checkpoint found)")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        n_conflict, n_batches = 0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            pred_soh, pred_rul = model(Xt[idx], HIt[idx])
            l_soh = mse(pred_soh, soh_t[idx])
            l_rul = mse(pred_rul, rul_t[idx])

            grads_soh = torch.autograd.grad(l_soh, shared_params, retain_graph=True)
            grads_rul = torch.autograd.grad(l_rul, shared_params, retain_graph=True)
            g1 = torch.cat([g.flatten() for g in grads_soh])
            g2 = torch.cat([g.flatten() for g in grads_rul])
            dot = torch.dot(g1, g2)
            if dot < 0:
                n_conflict += 1
                g1p = g1 - dot / (g2.norm() ** 2 + 1e-12) * g2
                g2p = g2 - dot / (g1.norm() ** 2 + 1e-12) * g1
            else:
                g1p, g2p = g1, g2
            combined = g1p + g2p

            opt.zero_grad()
            pos = 0
            for p in shared_params:
                cnt = p.numel()
                p.grad = combined[pos:pos + cnt].view_as(p).clone()
                pos += cnt
            head_grads = torch.autograd.grad(l_soh + l_rul, head_params)
            for p, g in zip(head_params, head_grads):
                p.grad = g
            opt.step()

            ep_loss += (l_soh.item() + l_rul.item()) * len(idx)
            n_batches += 1
        ep_loss /= n

        model.eval()
        with torch.no_grad():
            vp_soh, vp_rul = model(Xv, HIv)
            v_l_soh = mse(vp_soh, soh_v).item()
            v_l_rul = mse(vp_rul, rul_v).item()
        history.append({"epoch": epoch, "train_loss": ep_loss, "val_loss_soh": v_l_soh,
                         "val_loss_rul": v_l_rul, "frac_conflict": n_conflict / max(1, n_batches)})
        print(f"[pcgrad] epoch {epoch:2d} train={ep_loss:.4f} val_soh={v_l_soh:.4f} "
              f"val_rul={v_l_rul:.4f} conflict_frac={n_conflict/max(1,n_batches):.2f}")

        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "epoch": epoch, "history": history}, _CKPT_PCGRAD)

    _CKPT_PCGRAD.unlink(missing_ok=True)
    return model, history


def evaluate(model, data, tag):
    model.eval()
    with torch.no_grad():
        pred_soh_z, pred_rul_z = model(data["Xtest"], data["HItest"])
    pred_soh = pred_soh_z.squeeze(-1).numpy() * data["soh_std"] + data["soh_mean"]
    pred_rul = pred_rul_z.squeeze(-1).numpy() * data["rul_std"] + data["rul_mean"]
    soh_metrics = m(data["soh_test"], pred_soh)
    rul_metrics = m(data["rul_test"], pred_rul)
    print(f"[{tag}] TEST SOH: {soh_metrics}")
    print(f"[{tag}] TEST RUL: {rul_metrics}")
    return soh_metrics, rul_metrics


def main():
    t0 = time.time()
    data = load_data()

    print("\n[stage3.2] === Training GradNorm ===")
    model_gn, hist_gn = train_gradnorm(data)
    soh_gn, rul_gn = evaluate(model_gn, data, "gradnorm")
    torch.save(model_gn.state_dict(), ROOT / "models" / "joint_fusion_gradnorm.pt")
    pd.DataFrame(hist_gn).to_csv(PROC_DIR / "predictions" / "joint_fusion_gradnorm_history.csv", index=False)

    print("\n[stage3.2] === Training PCGrad ===")
    model_pc, hist_pc = train_pcgrad(data)
    soh_pc, rul_pc = evaluate(model_pc, data, "pcgrad")
    torch.save(model_pc.state_dict(), ROOT / "models" / "joint_fusion_pcgrad.pt")
    pd.DataFrame(hist_pc).to_csv(PROC_DIR / "predictions" / "joint_fusion_pcgrad_history.csv", index=False)

    print("\n[stage3.2] === COMPARISON vs. all prior loss-weighting variants ===")
    print("[stage3.2] fixed_balanced (session 4):           SOH R2=0.416  RUL R2=0.428")
    print("[stage3.2] adaptive-clamped (session 4):          SOH R2=0.344  RUL R2=0.432")
    print("[stage3.2] softmax-normalized:                    SOH R2=0.091  RUL R2=0.244 (loses both)")
    print("[stage3.2] session 41 HI-fused (adaptive-clamped + 1.1 features, PRIMARY baseline "
          "for this item since it is the same architecture/feature-config as GradNorm/PCGrad "
          "here - isolates the loss-balancing mechanism as the only variable): "
          "SOH R2=0.9241  RUL R2=0.6657")
    print(f"[stage3.2] GradNorm (THIS ITEM):  SOH R2={soh_gn['r2']:.4f}  RUL R2={rul_gn['r2']:.4f}")
    print(f"[stage3.2] PCGrad (THIS ITEM):    SOH R2={soh_pc['r2']:.4f}  RUL R2={rul_pc['r2']:.4f}")

    baseline_soh, baseline_rul = 0.9241, 0.6657
    for tag, soh, rul in [("GradNorm", soh_gn, rul_gn), ("PCGrad", soh_pc, rul_pc)]:
        beats_soh = soh['r2'] > baseline_soh
        beats_rul = rul['r2'] > baseline_rul
        verdict = ("beats existing best on BOTH tasks" if beats_soh and beats_rul else
                    "beats existing best on SOH only" if beats_soh else
                    "beats existing best on RUL only" if beats_rul else
                    "beats existing best on NEITHER task")
        print(f"[stage3.2] {tag} verdict: {verdict}")

    pd.DataFrame([
        {"method": "fixed_balanced", "soh_r2": 0.416, "rul_r2": 0.428},
        {"method": "adaptive_clamped", "soh_r2": 0.344, "rul_r2": 0.432},
        {"method": "softmax_normalized", "soh_r2": 0.091, "rul_r2": 0.244},
        {"method": "session41_HI_fused_baseline", "soh_r2": baseline_soh, "rul_r2": baseline_rul},
        {"method": "GradNorm", "soh_r2": soh_gn['r2'], "rul_r2": rul_gn['r2'],
         "soh_rmse": soh_gn['rmse'], "rul_rmse": rul_gn['rmse']},
        {"method": "PCGrad", "soh_r2": soh_pc['r2'], "rul_r2": rul_pc['r2'],
         "soh_rmse": soh_pc['rmse'], "rul_rmse": rul_pc['rmse']},
    ]).to_csv(OUT_DIR / "stage3_2_gradnorm_pcgrad_results.csv", index=False)
    print(f"\n[stage3.2] saved outputs/stage3_2_gradnorm_pcgrad_results.csv")
    print(f"[stage3.2] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
