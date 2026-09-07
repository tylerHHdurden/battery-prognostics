"""
Session 26: sensor-noise robustness test for the "lean" (session 20)
XGBoost-fusion pipeline. Tests something genuinely new relative to
prior sessions: robustness to realistic INPUT IMPERFECTION on the SAME
battery/domain, as opposed to domain shift (a different battery/dataset
entirely - sessions 5/13/19) or statistical significance of the
existing predictions (session 21). No retraining: `ica_encoder.pt` and
`xgb_soh_fusion.json` are used purely for inference, exactly as
deployed.

**Noise levels, stated explicitly per instruction**: Gaussian noise is
added independently to every raw V/I/T sample of each test cycle's
charge AND discharge phases, at 3 levels:
    1x  (BMS-grade, as literally specified by the task): sigma_V=1mV,
        sigma_I=10mA, sigma_T=0.5 degC
    2x: sigma_V=2mV, sigma_I=20mA, sigma_T=1.0 degC
    5x  (stress test): sigma_V=5mV, sigma_I=50mA, sigma_T=2.5 degC
**Interpretation stated explicitly**: the task's "+/-1mV" etc. is
interpreted here as the Gaussian noise STANDARD DEVIATION (sigma), not
a hard clip bound - a common engineering shorthand for roughly a
1-sigma sensor noise floor. This choice is stated because it directly
sets the actual noise magnitude injected; a hard-bounded uniform
interpretation would inject less extreme values for the same "+/-X".

Time (`t`) arrays are left untouched - a BMS's timing/sampling clock is
not the "sensor noise" in question here, only its V/I/T transducers.
`discharge_capacity` (the SOH ground-truth source, a separately-
measured/coulomb-counted field per data_adapters.py, not derived from
the V/I/T arrays being perturbed) is also left untouched, so ground-
truth SOH is IDENTICAL before and after noise injection for every
cycle - this experiment isolates "does the PREDICTION degrade under
noisy inputs", not a confound from a shifting target too.

Each test battery's raw cycles are loaded ONCE and reused in-memory
across all 3 noise levels (noisy copies only, originals untouched) -
avoids repeating the slow part (raw NASA .mat / MIT HDF5 parsing) 3x
for no reason.
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
OUT_DIR.mkdir(exist_ok=True)

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


def load_cycles(dataset: str, battery_id: str) -> list[dict]:
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    entry = next(e for e in mit_subset if e["global_id"] == battery_id)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def add_noise(cycle: dict, sigma_V: float, sigma_I: float, sigma_T: float, rng) -> dict:
    """Returns a NEW cycle dict with independent Gaussian noise added to
    every V/I/T sample of both phases; `t` and `discharge_capacity`
    untouched (see module docstring)."""
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
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        BFA_SELECTED = [l.strip() for l in f if l.strip()]
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())

    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    train_medians = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])][BFA_SELECTED].median(numeric_only=True)

    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))

    feature_cols = BFA_SELECTED + [f"fusion_{i}" for i in range(16)]

    print("[noise] loading raw test-battery cycles ONCE (reused across all noise levels)...")
    all_cycles = {}
    for dataset, battery_id in TEST_BATTERIES:
        cycles = load_cycles(dataset, battery_id)
        all_cycles[(dataset, battery_id)] = cycles
        print(f"[noise]   {dataset}/{battery_id}: {len(cycles)} cycles loaded")

    rng = np.random.default_rng(SEED)
    level_results = []
    per_cycle_rows = []

    for level in NOISE_LEVELS:
        print(f"\n[noise] === noise level: {level['name']} "
              f"(sigma_V={level['sigma_V']*1000:.1f}mV, sigma_I={level['sigma_I']*1000:.1f}mA, "
              f"sigma_T={level['sigma_T']:.2f}degC) ===")
        y_true_all, y_pred_all = [], []
        for dataset, battery_id in TEST_BATTERIES:
            cycles = all_cycles[(dataset, battery_id)]
            soh_map = soh_per_cycle(cycles)  # from UNTOUCHED discharge_capacity - identical at every noise level
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
        print(f"[noise] {level['name']}: n={len(y_true_all)} RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        level_results.append({"noise_level": level["name"], "sigma_V_mV": level["sigma_V"] * 1000,
                               "sigma_I_mA": level["sigma_I"] * 1000, "sigma_T_degC": level["sigma_T"],
                               "n": len(y_true_all), "rmse": rmse, "mae": mae, "r2": r2})

    results_df = pd.DataFrame(level_results)
    baseline = results_df.iloc[0]
    results_df["delta_rmse_vs_clean"] = results_df["rmse"] - baseline["rmse"]
    results_df["delta_r2_vs_clean"] = results_df["r2"] - baseline["r2"]

    print("\n[noise] === SUMMARY: accuracy degradation vs. noise level ===")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUT_DIR / "sensor_noise_robustness_summary.csv", index=False)
    pd.DataFrame(per_cycle_rows).to_csv(PRED_DIR / "sensor_noise_robustness_per_cycle.csv", index=False)
    print(f"\n[noise] saved outputs/sensor_noise_robustness_summary.csv and "
          f"predictions/sensor_noise_robustness_per_cycle.csv")
    print("[noise] DONE")


if __name__ == "__main__":
    main()
