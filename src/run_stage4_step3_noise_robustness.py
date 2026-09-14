"""
Stage 4, Step 3: re-run session 26's sensor-noise robustness methodology
(unchanged noise levels/battery set/logic) against the NEW Stage 4
model (canonical 1.1-reformulated features + 1.5 monotone constraints,
retrained ica_encoder.pt/xgb_soh_fusion.json) - neither this nor
second-life grading has been run against a model incorporating Stage
1+2's full fixes together with the recovered batteries.

Adaptation required, stated explicitly: session 26's original script
used the OLD 7-feature BFA set with no reformulation. The canonical
1.1-reformulated features need a per-battery baseline (this battery's
OWN cycle-10 value for ICHV/TEVD/TEVI) - computed HERE at the SAME
noise level as the rest of that battery's cycles (i.e. the baseline
itself is subject to the same simulated sensor noise, which is the
physically correct thing to do: a real BMS's cycle-10 reading would
also be noisy).
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
from stage1_common import canonical_feature_cols, fusion_cols, DURATION_FEATURES, BASELINE_CYCLE

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
            "T": d["T"] + rng.normal(0, sigma_T, size=d["T"].shape) if (sigma_T and d["T"] is not None) else d["T"],
        }
    return out


def reformulate(hi_df_noisy: pd.DataFrame, cycle_idxs: list[int]) -> pd.DataFrame:
    """Same ratio-to-cycle-10-baseline logic as stage1_common.
    add_reformulated_duration_features, applied to THIS noisy battery's
    own (noisy) cycle-10 value - not re-derived from clean data."""
    out = hi_df_noisy.copy()
    out["cycle_idx"] = cycle_idxs
    for feat in DURATION_FEATURES:
        base_row = out[out["cycle_idx"] == BASELINE_CYCLE]
        if len(base_row):
            base_val = float(base_row[feat].iloc[0])
        else:
            base_val = float(out.sort_values("cycle_idx")[feat].iloc[0])
        if not np.isfinite(base_val) or abs(base_val) < 1e-6:
            base_val = float(out[feat].median())
        out[f"{feat}_rel"] = out[feat] / base_val if (np.isfinite(base_val) and abs(base_val) >= 1e-6) else np.nan
    return out


def main():
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    base_features = canonical_feature_cols(reformulated=False)  # raw names, reformulate below
    feature_cols_final = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    train_medians_raw = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])][base_features].median(numeric_only=True)

    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    fcols = fusion_cols()
    feature_cols = feature_cols_final + fcols
    print(f"[stage4-noise] canonical (Stage 4) feature set: {feature_cols_final}")

    print("[stage4-noise] loading raw test-battery cycles ONCE (reused across all noise levels)...")
    all_cycles = {}
    for dataset, battery_id in TEST_BATTERIES:
        cycles = load_cycles(dataset, battery_id)
        all_cycles[(dataset, battery_id)] = cycles
        print(f"[stage4-noise]   {dataset}/{battery_id}: {len(cycles)} cycles loaded")

    rng = np.random.default_rng(SEED)
    level_results = []
    per_battery_rows = []
    per_cycle_rows = []

    for level in NOISE_LEVELS:
        print(f"\n[stage4-noise] === noise level: {level['name']} ===")
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

            hi_df_noisy = pd.DataFrame(hi_rows)[base_features]
            for col in base_features:
                hi_df_noisy[col] = hi_df_noisy[col].fillna(train_medians_raw[col])
            hi_df_ref = reformulate(hi_df_noisy, valid_idx)
            for col in feature_cols_final:
                if col == "cycle_idx":
                    continue
                if hi_df_ref[col].isna().any():
                    hi_df_ref[col] = hi_df_ref[col].fillna(hi_df_ref[col].median())

            X_all = apply_channel_norm(np.stack(tensors).astype(np.float32), norm_stats)
            with torch.no_grad():
                emb = encoder.encode(torch.tensor(X_all[:, :, ICA_CHANNEL_SLICE])).numpy()

            feat_df = hi_df_ref.reset_index(drop=True)
            for i in range(16):
                feat_df[f"fusion_{i}"] = emb[:, i]
            pred = xgb_fusion.predict(feat_df[feature_cols].to_numpy(dtype=float))

            y_true_all.extend(y_true)
            y_pred_all.extend(pred.tolist())
            b_rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
            b_r2 = float(r2_score(y_true, pred)) if np.std(y_true) > 0 else float("nan")
            per_battery_rows.append({"noise_level": level["name"], "dataset": dataset,
                                      "battery_id": battery_id, "n": len(y_true),
                                      "rmse": b_rmse, "r2": b_r2})
            for cyc, yt, yp in zip(valid_idx, y_true, pred):
                per_cycle_rows.append({"noise_level": level["name"], "dataset": dataset,
                                        "battery_id": battery_id, "cycle_idx": cyc,
                                        "SOH_true": yt, "SOH_pred": float(yp)})

        y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
        rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
        mae = float(mean_absolute_error(y_true_all, y_pred_all))
        r2 = float(r2_score(y_true_all, y_pred_all))
        print(f"[stage4-noise] {level['name']}: n={len(y_true_all)} RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        level_results.append({"noise_level": level["name"], "sigma_V_mV": level["sigma_V"] * 1000,
                               "sigma_I_mA": level["sigma_I"] * 1000, "sigma_T_degC": level["sigma_T"],
                               "n": len(y_true_all), "rmse": rmse, "mae": mae, "r2": r2})

    results_df = pd.DataFrame(level_results)
    baseline = results_df.iloc[0]
    results_df["delta_rmse_vs_clean"] = results_df["rmse"] - baseline["rmse"]
    results_df["delta_r2_vs_clean"] = results_df["r2"] - baseline["r2"]
    print(f"\n[stage4-noise] === AGGREGATE (all 6 test batteries pooled) ===")
    print(results_df.to_string(index=False))
    print(f"\n[stage4-noise] prior (lean, pre-Stage-4) result for comparison: "
          f"clean R2=0.9750, 1x=0.9742, 2x=0.9736, 5x-stress=... "
          f"(session 26/47's monotonic-degradation, no-collapse finding)")

    per_batt_df = pd.DataFrame(per_battery_rows)
    print(f"\n[stage4-noise] === per-battery breakdown ===")
    pivot = per_batt_df.pivot(index=["dataset", "battery_id"], columns="noise_level", values="r2")
    pivot = pivot[[lvl["name"] for lvl in NOISE_LEVELS]]
    print(pivot.to_string())

    monotonic_degradation = all(
        results_df.iloc[i]["r2"] >= results_df.iloc[i + 1]["r2"] - 0.02  # small tolerance for noise
        for i in range(len(results_df) - 1)
    )
    print(f"\n[stage4-noise] monotonic degradation pattern (R2 non-increasing with noise level, "
          f"within tolerance): {monotonic_degradation}")

    results_df.to_csv(OUT_DIR / "stage4_noise_robustness_summary.csv", index=False)
    per_batt_df.to_csv(OUT_DIR / "stage4_noise_robustness_per_battery.csv", index=False)
    pd.DataFrame(per_cycle_rows).to_csv(PRED_DIR / "stage4_noise_robustness_per_cycle.csv", index=False)
    print(f"\n[stage4-noise] saved outputs/stage4_noise_robustness_{{summary,per_battery}}.csv")
    print("[stage4-noise] DONE")


if __name__ == "__main__":
    main()
