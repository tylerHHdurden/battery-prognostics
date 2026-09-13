"""
Stage 1 closeout, Check 1: per-battery breakdown of the 1.4 robustness-
margin anomaly (MSE-baseline PiFormer's pooled R2 IMPROVED under 5x-
stress noise). Reuses run_stage1_4_robustness_margin.py's own
load_cycles/add_noise/predict functions UNCHANGED - only difference is
this script evaluates ONLY the MSE-baseline model (not also 1.4's
combined model, cheaper) and saves PER-BATTERY R2/RMSE for both
conditions, which the original script did not save.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from sequence_features import build_dataset_tensors, apply_channel_norm
from models.piformer import PiFormer
from train_deep_models import make_xy
from train_deep_models_expanded import load_all_battery_tensors_expanded
from run_stage1_4_robustness_margin import load_cycles, add_noise, CONDITIONS, SEED

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"


def predict(model, X, chunk_size=4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), chunk_size):
            outs.append(model(torch.tensor(X[i:i + chunk_size])).squeeze(-1).numpy())
    raw = np.concatenate(outs)
    return raw * model.y_std_ + model.y_mean_


def main():
    t0 = time.time()
    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    test_ids = split["test_ids"]
    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full_raw = json.load(f)

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())

    model = PiFormer()
    model.load_state_dict(torch.load(ROOT / "models" / "piformer_soh_expanded.pt"))
    model.eval()

    battery_data = load_all_battery_tensors_expanded()
    all_train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val_batteries = max(1, len(all_train_ids) // 5)
    val_ids = sorted(all_train_ids)[-n_val_batteries:]
    fit_ids = [b for b in all_train_ids if b not in val_ids]
    _, y_fit, _, _, _, _ = make_xy(battery_data, fit_ids)
    model.y_mean_, model.y_std_ = float(y_fit.mean()), float(y_fit.std() + 1e-8)
    print(f"[check1] y stats (fit-only, matches training): mean={model.y_mean_:.4f} std={model.y_std_:.4f}")

    all_rows = []
    per_battery_rows = []
    for cond in CONDITIONS:
        rng = np.random.RandomState(SEED)
        cond_X, cond_y, cond_bid = [], [], []
        for bid in test_ids:
            dataset = "NASA" if bid.startswith("B0") else "MIT"
            cycles = load_cycles(dataset, bid, mit_full_raw)
            noisy_cycles = [add_noise(c, cond["sigma_V"], cond["sigma_I"], cond["sigma_T"], rng) for c in cycles]
            X, soh, rul, idxs, censored = build_dataset_tensors(noisy_cycles)
            if X is None:
                continue
            cond_X.append(X.astype(np.float32))
            cond_y.append(soh.astype(np.float32))
            cond_bid += [bid] * len(soh)
        X_cond = np.concatenate(cond_X)
        y_cond = np.concatenate(cond_y)
        X_cond = apply_channel_norm(X_cond, norm_stats)
        pred_cond = predict(model, X_cond)
        print(f"[check1] condition={cond['name']}: {len(X_cond)} cycles built ({time.time()-t0:.0f}s elapsed)")

        pooled_r2 = float(r2_score(y_cond, pred_cond))
        pooled_rmse = float(np.sqrt(mean_squared_error(y_cond, pred_cond)))
        print(f"[check1] [{cond['name']}] POOLED: R2={pooled_r2:.4f} RMSE={pooled_rmse:.4f}")

        df = pd.DataFrame({"battery_id": cond_bid, "y_true": y_cond, "pred": pred_cond})
        for bid, g in df.groupby("battery_id"):
            if len(g) < 2 or g["y_true"].std() == 0:
                r2_b, rmse_b = float("nan"), float(np.sqrt(np.mean((g.y_true - g.pred) ** 2)))
            else:
                r2_b = float(r2_score(g["y_true"], g["pred"]))
                rmse_b = float(np.sqrt(mean_squared_error(g["y_true"], g["pred"])))
            per_battery_rows.append({"condition": cond["name"], "battery_id": bid,
                                      "n_cycles": len(g), "r2": r2_b, "rmse": rmse_b})
        all_rows.append({"condition": cond["name"], "pooled_r2": pooled_r2, "pooled_rmse": pooled_rmse})

    per_batt_df = pd.DataFrame(per_battery_rows)
    piv_r2 = per_batt_df.pivot(index="battery_id", columns="condition", values="r2")
    piv_rmse = per_batt_df.pivot(index="battery_id", columns="condition", values="rmse")
    clean_col, stress_col = "clean", "5x BMS-grade (stress)"
    piv_r2["delta_r2"] = piv_r2[stress_col] - piv_r2[clean_col]
    piv_rmse["delta_rmse"] = piv_rmse[stress_col] - piv_rmse[clean_col]

    n_improved = int((piv_r2["delta_r2"] > 0.01).sum())
    n_degraded = int((piv_r2["delta_r2"] < -0.01).sum())
    n_flat = len(piv_r2) - n_improved - n_degraded
    print(f"\n[check1] === PER-BATTERY BREAKDOWN (clean vs. 5x-stress, MSE-baseline PiFormer) ===")
    print(f"[check1] {n_improved} batteries improved (delta R2 > +0.01), "
          f"{n_degraded} degraded (delta R2 < -0.01), {n_flat} roughly flat")
    print(piv_r2.sort_values("delta_r2").to_string())

    nasa_ids = [b for b in test_ids if b.startswith("B0")]
    nasa_deltas = piv_r2.loc[[b for b in nasa_ids if b in piv_r2.index], "delta_r2"]
    mit_deltas = piv_r2.loc[[b for b in piv_r2.index if b not in nasa_ids], "delta_r2"]
    print(f"\n[check1] NASA test batteries' delta R2 (clean->5x-stress): "
          f"{nasa_deltas.to_dict()}")
    print(f"[check1] NASA mean delta R2: {nasa_deltas.mean():+.4f} (n={len(nasa_deltas)})")
    print(f"[check1] MIT mean delta R2:  {mit_deltas.mean():+.4f} (n={len(mit_deltas)})")
    print(f"[check1] MIT batteries improving (delta>+0.01): {int((mit_deltas>0.01).sum())} of {len(mit_deltas)} "
          f"({(mit_deltas>0.01).mean()*100:.0f}%)")
    print(f"[check1] NASA batteries improving (delta>+0.01): {int((nasa_deltas>0.01).sum())} of {len(nasa_deltas)} "
          f"({(nasa_deltas>0.01).mean()*100:.0f}%)")

    matches_session26 = mit_deltas.mean() > 0 and nasa_deltas.mean() <= mit_deltas.mean()
    print(f"\n[check1] Matches session 26's pattern (majority mild-improve, NASA/hard batteries genuinely "
          f"degrade or improve less): {matches_session26}")

    piv_r2.reset_index().to_csv(OUT_DIR / "stage1_followup_check1_perbattery_r2.csv", index=False)
    piv_rmse.reset_index().to_csv(OUT_DIR / "stage1_followup_check1_perbattery_rmse.csv", index=False)
    print(f"\n[check1] saved outputs/stage1_followup_check1_perbattery_{{r2,rmse}}.csv")
    print(f"[check1] DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
