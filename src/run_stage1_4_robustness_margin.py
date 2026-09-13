"""
Stage 1, Item 1.4 (remaining piece): clean vs. 5x-stress robustness
margin for PiFormer-huber-noise (the Huber+noise COMBINED retrain),
compared against the plain MSE-trained PiFormer-expanded baseline.

SCOPE DEVIATION FROM session 35 Part 3's evaluation script, stated
explicitly: that script evaluates the FULL 4-branch ensemble on the
ORIGINAL 32-battery pool's 6 test battery IDs (hardcoded
TEST_BATTERIES). This script evaluates PiFormer STANDALONE (not the
ensemble - isolating the model this item actually changed) on the
EXPANDED 204-battery pool's own 40 test battery IDs (from
battery_split_expanded.json) - the correct test set for a model that
was TRAINED on the expanded pool (whose test-battery membership differs
from the original 32-battery pool's 6 - e.g. B0018 moved into TRAIN in
the expanded split, per session 35 Part 2). Only 2 of session 35's 4
noise tiers are run (clean, 5x-stress) - the two the "robustness
margin" metric is actually defined from - to keep this affordable.
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
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from sequence_features import build_dataset_tensors, apply_channel_norm
from models.piformer import PiFormer
from train_deep_models import make_xy
from train_deep_models_expanded import load_all_battery_tensors_expanded

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

SEED = 42
CONDITIONS = [
    {"name": "clean", "sigma_V": 0.0, "sigma_I": 0.0, "sigma_T": 0.0},
    {"name": "5x BMS-grade (stress)", "sigma_V": 0.005, "sigma_I": 0.05, "sigma_T": 2.5},
]


def load_cycles(dataset: str, battery_id: str, mit_full: list) -> list[dict]:
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    entry = next(e for e in mit_full if e["global_id"] == battery_id)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def add_noise(cycle: dict, sigma_V: float, sigma_I: float, sigma_T: float, rng) -> dict:
    if sigma_V == 0 and sigma_I == 0 and sigma_T == 0:
        return cycle
    out = {"cycle_idx": cycle["cycle_idx"], "discharge_capacity": cycle["discharge_capacity"]}
    for phase in ("charge", "discharge"):
        d = cycle[phase]
        out[phase] = {
            "t": d["t"],
            "V": d["V"] + rng.normal(0, sigma_V, size=d["V"].shape) if sigma_V else d["V"],
            "I": d["I"] + rng.normal(0, sigma_I, size=d["I"].shape) if sigma_I else d["I"],
            "T": d["T"] + rng.normal(0, sigma_T, size=d["T"].shape) if sigma_T else d["T"],
        }
    return out


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
    print(f"[robustness-margin] expanded-pool test batteries: {len(test_ids)}")

    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full_raw = json.load(f)

    # dataset lookup: NASA ids look like B00xx, MIT ids look like bNcMM
    nasa_ids = [b for b in test_ids if b.startswith("B0")]
    mit_ids = [b for b in test_ids if not b.startswith("B0")]
    print(f"[robustness-margin] {len(nasa_ids)} NASA + {len(mit_ids)} MIT test batteries")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())

    def load_piformer(ckpt_name):
        model = PiFormer()
        model.load_state_dict(torch.load(ROOT / "models" / ckpt_name))
        model.eval()
        return model

    piformer_mse = load_piformer("piformer_soh_expanded.pt")
    piformer_huber_noise = load_piformer("piformer_soh_huber_noise_expanded.pt")

    # y_mean_/y_std_ are needed to de-standardize predictions and are NOT
    # saved in the .pt state_dict - recompute EXACTLY the fit_ids-only
    # (not fit+val) SOH stats both this stage's training script and
    # train_deep_models_expanded.py used, via the identical
    # sorted(train_ids)[-n_val:] convention - not approximated from a
    # fit+val CSV average, which would silently bias every de-
    # standardized prediction.
    battery_data = load_all_battery_tensors_expanded()
    all_train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val_batteries = max(1, len(all_train_ids) // 5)
    val_ids = sorted(all_train_ids)[-n_val_batteries:]
    fit_ids = [b for b in all_train_ids if b not in val_ids]
    _, y_fit, _, _, _, _ = make_xy(battery_data, fit_ids)
    y_mean_fit, y_std_fit = float(y_fit.mean()), float(y_fit.std() + 1e-8)
    print(f"[robustness-margin] fit-only y stats (matches training exactly): mean={y_mean_fit:.4f} std={y_std_fit:.4f}")
    piformer_mse.y_mean_, piformer_mse.y_std_ = y_mean_fit, y_std_fit
    piformer_huber_noise.y_mean_, piformer_huber_noise.y_std_ = y_mean_fit, y_std_fit

    results = []
    for cond in CONDITIONS:
        rng = np.random.RandomState(SEED)
        all_X, all_y, all_bid = [], [], []
        for bid in test_ids:
            dataset = "NASA" if bid.startswith("B0") else "MIT"
            cycles = load_cycles(dataset, bid, mit_full_raw)
            noisy_cycles = [add_noise(c, cond["sigma_V"], cond["sigma_I"], cond["sigma_T"], rng) for c in cycles]
            X, soh, rul, idxs, censored = build_dataset_tensors(noisy_cycles)
            if X is None:
                continue
            all_X.append(X.astype(np.float32))
            all_y.append(soh.astype(np.float32))
            all_bid += [bid] * len(soh)
        X_cond = np.concatenate(all_X)
        y_cond = np.concatenate(all_y)
        X_cond = apply_channel_norm(X_cond, norm_stats)
        print(f"[robustness-margin] condition={cond['name']}: {len(X_cond)} test cycles built "
              f"({time.time()-t0:.0f}s elapsed)")

        pred_mse = predict(piformer_mse, X_cond)
        pred_hn = predict(piformer_huber_noise, X_cond)
        r2_mse = float(r2_score(y_cond, pred_mse))
        r2_hn = float(r2_score(y_cond, pred_hn))
        rmse_mse = float(np.sqrt(mean_squared_error(y_cond, pred_mse)))
        rmse_hn = float(np.sqrt(mean_squared_error(y_cond, pred_hn)))
        print(f"[robustness-margin] [{cond['name']}] MSE-baseline: R2={r2_mse:.4f} RMSE={rmse_mse:.4f} | "
              f"Huber+noise: R2={r2_hn:.4f} RMSE={rmse_hn:.4f}")
        results.append({"condition": cond["name"], "model": "MSE-baseline", "r2": r2_mse, "rmse": rmse_mse})
        results.append({"condition": cond["name"], "model": "Huber+noise-1.4", "r2": r2_hn, "rmse": rmse_hn})

    df = pd.DataFrame(results)
    piv = df.pivot(index="model", columns="condition", values="r2")
    margin_mse = piv.loc["MSE-baseline", "clean"] - piv.loc["MSE-baseline", "5x BMS-grade (stress)"]
    margin_hn = piv.loc["Huber+noise-1.4", "clean"] - piv.loc["Huber+noise-1.4", "5x BMS-grade (stress)"]
    print(f"\n[robustness-margin] Robustness margin (clean R2 - 5x-stress R2): "
          f"MSE-baseline={margin_mse:.4f}, Huber+noise(1.4)={margin_hn:.4f} - "
          f"{'NARROWER (1.4 is more robust)' if margin_hn < margin_mse else 'WIDER (1.4 is LESS robust)'}")

    df.to_csv(OUT_DIR / "stage1_4_robustness_margin.csv", index=False)
    print(f"\n[robustness-margin] saved outputs/stage1_4_robustness_margin.csv")
    print(f"[robustness-margin] DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
