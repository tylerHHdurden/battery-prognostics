"""
Dataset Expansion Phase 1, Step 4: re-runs session 26's sensor-noise
robustness test using the EXPANDED-pool lean pipeline
(ica_encoder_expanded.pt + xgb_soh_fusion_expanded.json), evaluated on
the SAME 6 original test batteries (NASA/B0018, MIT/b1c4, b2c24, b3c0,
b3c35, b4c38) as session 26 - a deliberate scope choice, not the full
expanded test set: the point of this script is a direct, apples-to-
apples "does more training data fix B0018's noise-robustness weak
point" comparison against session 26's own per-battery numbers, which
requires holding the EVALUATION set fixed and only changing the model
being evaluated. Running the full ~40+-battery expanded test set here
was not requested and would not serve that specific comparison any
better. Same 3 noise levels/interpretation/method as the original,
unchanged.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from health_indicators import compute_health_indicators
from rul_labels import soh_per_cycle
from sequence_features import get_cycle_tensor, apply_channel_norm
from models.ica_encoder import ICAEncoder

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

ICA_CHANNEL_SLICE = slice(3, 6)
SEED = 42

NOISE_LEVELS = [
    {"name": "clean (baseline)", "sigma_V": 0.0, "sigma_I": 0.0, "sigma_T": 0.0},
    {"name": "1x BMS-grade", "sigma_V": 0.001, "sigma_I": 0.01, "sigma_T": 0.5},
    {"name": "2x BMS-grade", "sigma_V": 0.002, "sigma_I": 0.02, "sigma_T": 1.0},
    {"name": "5x BMS-grade (stress)", "sigma_V": 0.005, "sigma_I": 0.05, "sigma_T": 2.5},
]

TEST_BATTERIES = [("NASA", "B0018"), ("MIT", "b1c4"), ("MIT", "b2c24"),
                   ("MIT", "b3c0"), ("MIT", "b3c35"), ("MIT", "b4c38")]

ORIGINAL_R2 = {"NASA/B0018": {"clean": 0.576, "1x": 0.565, "2x": 0.525, "5x": 0.513},
               "MIT/b2c24": {"clean": 0.939, "1x": 0.948, "2x": 0.943, "5x": 0.897}}


def load_cycles(dataset: str, battery_id: str) -> list[dict]:
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    entry = next(e for e in mit_subset if e["global_id"] == battery_id)
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


def main():
    with open(PROC_DIR / "bfa_selected_features_expanded.txt") as f:
        BFA_SELECTED = [l.strip() for l in f if l.strip()]
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())

    hi_df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    train_medians = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])][BFA_SELECTED].median(numeric_only=True)

    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder_expanded.pt"))
    encoder.eval()

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion_expanded.json"))

    feature_cols = BFA_SELECTED + [f"fusion_{i}" for i in range(16)]

    print("[noise-exp] loading raw test-battery cycles ONCE...")
    all_cycles = {}
    for dataset, battery_id in TEST_BATTERIES:
        cycles = load_cycles(dataset, battery_id)
        all_cycles[(dataset, battery_id)] = cycles
        print(f"[noise-exp]   {dataset}/{battery_id}: {len(cycles)} cycles loaded")

    rng = np.random.default_rng(SEED)
    level_results = []
    per_battery_results = []
    per_cycle_rows = []

    for level in NOISE_LEVELS:
        print(f"\n[noise-exp] === noise level: {level['name']} ===")
        y_true_all, y_pred_all = [], []
        for dataset, battery_id in TEST_BATTERIES:
            cycles = all_cycles[(dataset, battery_id)]
            soh_map = soh_per_cycle(cycles)
            noisy_cycles = [add_noise(c, level["sigma_V"], level["sigma_I"], level["sigma_T"], rng)
                             for c in cycles]

            hi_rows, tensors, y_true, valid_idx = [], [], [], []
            for c in noisy_cycles:
                his = compute_health_indicators(c)
                tensor = get_cycle_tensor(c)
                if tensor is None:
                    continue
                hi_rows.append(his)
                tensors.append(tensor)
                y_true.append(soh_map[c["cycle_idx"]])
                valid_idx.append(c["cycle_idx"])

            hi_df_noisy = pd.DataFrame(hi_rows)[BFA_SELECTED]
            for col in BFA_SELECTED:
                hi_df_noisy[col] = hi_df_noisy[col].fillna(train_medians[col])

            X_all = apply_channel_norm(np.stack(tensors).astype(np.float32), norm_stats)
            with torch.no_grad():
                emb = encoder.encode(torch.tensor(X_all[:, :, ICA_CHANNEL_SLICE])).numpy()

            feat_df = hi_df_noisy.reset_index(drop=True)
            for i in range(16):
                feat_df[f"fusion_{i}"] = emb[:, i]
            pred = xgb_fusion.predict(feat_df[feature_cols].to_numpy(dtype=float))

            y_true_arr, pred_arr = np.array(y_true), np.array(pred)
            batt_r2 = float(r2_score(y_true_arr, pred_arr)) if len(y_true_arr) > 1 else float("nan")
            batt_rmse = float(np.sqrt(mean_squared_error(y_true_arr, pred_arr)))
            per_battery_results.append({"noise_level": level["name"], "dataset": dataset,
                                         "battery_id": battery_id, "n": len(y_true_arr),
                                         "rmse": batt_rmse, "r2": batt_r2})

            y_true_all.extend(y_true)
            y_pred_all.extend(pred.tolist())
            for cyc, yt, yp in zip(valid_idx, y_true, pred):
                per_cycle_rows.append({"noise_level": level["name"], "dataset": dataset,
                                        "battery_id": battery_id, "cycle_idx": cyc,
                                        "SOH_true": yt, "SOH_pred": float(yp)})

        y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
        rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
        mae = float(mean_absolute_error(y_true_all, y_pred_all))
        r2 = float(r2_score(y_true_all, y_pred_all))
        print(f"[noise-exp] {level['name']}: n={len(y_true_all)} RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        level_results.append({"noise_level": level["name"], "sigma_V_mV": level["sigma_V"] * 1000,
                               "sigma_I_mA": level["sigma_I"] * 1000, "sigma_T_degC": level["sigma_T"],
                               "n": len(y_true_all), "rmse": rmse, "mae": mae, "r2": r2})

    results_df = pd.DataFrame(level_results)
    baseline = results_df.iloc[0]
    results_df["delta_rmse_vs_clean"] = results_df["rmse"] - baseline["rmse"]
    results_df["delta_r2_vs_clean"] = results_df["r2"] - baseline["r2"]
    print("\n[noise-exp] === SUMMARY (pooled) ===")
    print(results_df.to_string(index=False))

    per_batt_df = pd.DataFrame(per_battery_results)
    b0018 = per_batt_df[per_batt_df["battery_id"] == "B0018"]
    print("\n[noise-exp] === NASA/B0018 specifically (per noise level) ===")
    print(b0018.to_string(index=False))
    print("[noise-exp] ORIGINAL 32-battery reference for B0018: clean R2=0.576, "
          "1x=0.565, 2x=0.525, 5x=0.513 (monotonic degradation)")
    b0018_r2 = dict(zip(b0018["noise_level"], b0018["r2"]))
    is_monotonic = (b0018_r2.get("clean (baseline)", 1) >= b0018_r2.get("1x BMS-grade", 1) >=
                     b0018_r2.get("2x BMS-grade", 1) >= b0018_r2.get("5x BMS-grade (stress)", 1))
    print(f"[noise-exp] B0018 still degrades monotonically with noise: {is_monotonic}")

    results_df.to_csv(OUT_DIR / "sensor_noise_robustness_expanded_summary.csv", index=False)
    per_batt_df.to_csv(OUT_DIR / "sensor_noise_robustness_expanded_per_battery.csv", index=False)
    pd.DataFrame(per_cycle_rows).to_csv(PRED_DIR / "sensor_noise_robustness_expanded_per_cycle.csv", index=False)
    print("\n[noise-exp] DONE")


if __name__ == "__main__":
    main()
