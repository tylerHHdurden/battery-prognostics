"""
Follow-up Part C: re-run second-life grading (session 25) and sensor-
noise robustness (session 26) using Stage 1's 1.1+1.5 XGBoost-fusion
model instead of the original lean pipeline - the actual current best
point predictor, never checked against either analysis.

Pool: the ORIGINAL 32-battery pool (battery_split.json) - matching
where both session 25/26 and Stage 1's own 1.1+1.5 model were
evaluated, so this is a clean like-for-like swap of the point predictor
only, nothing else.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from health_indicators import compute_health_indicators
from rul_labels import soh_per_cycle
from sequence_features import get_cycle_tensor, apply_channel_norm
from models.ica_encoder import ICAEncoder
from stage1_common import (
    canonical_feature_cols, DURATION_FEATURES, load_nasa_mit_pool,
    battery_split_masks, fit_xgb, fusion_cols, OUT_DIR,
)

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"

ICA_CHANNEL_SLICE = slice(3, 6)
SEED = 42
BASELINE_CYCLE = 10

NOISE_LEVELS = [
    {"name": "clean (baseline)", "sigma_V": 0.0, "sigma_I": 0.0, "sigma_T": 0.0},
    {"name": "1x BMS-grade", "sigma_V": 0.001, "sigma_I": 0.01, "sigma_T": 0.5},
    {"name": "2x BMS-grade", "sigma_V": 0.002, "sigma_I": 0.02, "sigma_T": 1.0},
    {"name": "5x BMS-grade (stress)", "sigma_V": 0.005, "sigma_I": 0.05, "sigma_T": 2.5},
]
TEST_BATTERIES = [("NASA", "B0018"), ("MIT", "b1c4"), ("MIT", "b2c24"),
                   ("MIT", "b3c0"), ("MIT", "b3c35"), ("MIT", "b4c38")]

GRADE_ORDER = ["Recycle only", "Second-life candidate", "Primary EV use"]
GRADE_ORDINAL = {g: i for i, g in enumerate(GRADE_ORDER)}


def grade(soh: float) -> str:
    if soh >= 80:
        return "Primary EV use"
    elif soh >= 50:
        return "Second-life candidate"
    else:
        return "Recycle only"


def load_cycles(dataset, battery_id, mit_subset):
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    entry = next(e for e in mit_subset if e["global_id"] == battery_id)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def add_noise(cycle, sigma_V, sigma_I, sigma_T, rng):
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
    print("[part-c] === building the Stage 1.1+1.5 model (canonical reformulated features + monotone_constraints) ===")
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    base_features = canonical_feature_cols(reformulated=True)
    feature_cols = base_features + ["cycle_idx"]
    n_fusion = len(fusion_cols())
    monotone = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)
    model, medians, cols = fit_xgb(merged, train_mask, feature_cols, xgb_extra_kwargs={"monotone_constraints": monotone})
    print(f"[part-c] model trained on {train_mask.sum()} rows, feature order: {cols}")

    # =========================== C.1/C.2: second-life grading ===========================
    print("\n[part-c] === C.1/C.2: second-life grading, Stage 1.1+1.5 model ===")
    test_df = merged[test_mask].copy()
    X_test = test_df[cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(medians, inds[1])
    test_df["y_pred"] = model.predict(X_test)

    print(f"[part-c] B0018 in test set: {'B0018' in test_df['battery_id'].unique()} (must be True for the "
          f"original 32-battery pool per session 33's finding that B0018 only moves to train in the EXPANDED split)")

    test_df["true_grade"] = test_df["SOH"].apply(grade)
    test_df["pred_grade"] = test_df["y_pred"].apply(grade)
    test_df["true_ord"] = test_df["true_grade"].map(GRADE_ORDINAL)
    test_df["pred_ord"] = test_df["pred_grade"].map(GRADE_ORDINAL)
    test_df["misgrade_direction"] = np.select(
        [test_df["pred_ord"] > test_df["true_ord"], test_df["pred_ord"] < test_df["true_ord"]],
        ["risky (predicted too optimistic)", "conservative (predicted too pessimistic)"], default="correct")

    agreement = (test_df["true_grade"] == test_df["pred_grade"]).mean()
    print(f"[part-c] grading agreement: {agreement*100:.2f}% of {len(test_df)} test cycles "
          f"(session 25's original lean-pipeline agreement - see second_life_grading_per_cycle.csv for exact recompute if needed)")
    misgrade_counts = test_df["misgrade_direction"].value_counts()
    for k in ["correct", "risky (predicted too optimistic)", "conservative (predicted too pessimistic)"]:
        n = misgrade_counts.get(k, 0)
        print(f"[part-c]   {k}: {n} ({n/len(test_df)*100:.2f}%)")

    last_cycle = test_df.sort_values("cycle_idx").groupby(["dataset", "battery_id"]).tail(1)
    last_cycle = last_cycle[["dataset", "battery_id", "cycle_idx", "SOH", "y_pred", "true_grade", "pred_grade"]].sort_values("battery_id")
    print("\n[part-c] === per-battery CURRENT-STATUS grade (Stage 1.1+1.5 model) ===")
    print(last_cycle.to_string(index=False))

    b0018_row = last_cycle[last_cycle.battery_id == "B0018"].iloc[0]
    print(f"\n[part-c] B0018 CURRENT-STATUS: true SOH={b0018_row.SOH:.2f}% (grade={b0018_row.true_grade}), "
          f"predicted SOH={b0018_row.y_pred:.2f}% (grade={b0018_row.pred_grade})")
    print(f"[part-c] ORIGINAL lean pipeline (session 25): true=72.76%, predicted=81.50% - a RISKY misgrade")
    fixed = b0018_row.true_grade == b0018_row.pred_grade
    b0018_direction = test_df.loc[
        (test_df.battery_id == "B0018") & (test_df.cycle_idx == int(b0018_row.cycle_idx)), "misgrade_direction"
    ].iloc[0]
    print(f"[part-c] B0018 misgrade under Stage 1.1+1.5: "
          f"{'FIXED - grades now match' if fixed else f'STILL a misgrade ({b0018_direction})'}")
    print(f"[part-c] B0018 predicted-SOH error: original lean pipeline +8.74pp (81.50 vs 72.76), "
          f"Stage 1.1+1.5 {b0018_row.y_pred - b0018_row.SOH:+.2f}pp ({b0018_row.y_pred:.2f} vs {b0018_row.SOH:.2f})")

    last_cycle.to_csv(OUT_DIR / "stage1_followup_partC_grading_current_status.csv", index=False)
    test_df[["dataset", "battery_id", "cycle_idx", "SOH", "y_pred", "true_grade", "pred_grade", "misgrade_direction"]].to_csv(
        PRED_DIR / "stage1_followup_partC_grading_per_cycle.csv", index=False)

    # =========================== C.3/C.4: sensor-noise robustness ===========================
    print("\n[part-c] === C.3/C.4: sensor-noise robustness, Stage 1.1+1.5 model ===")
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()

    print("[part-c] loading raw test-battery cycles once...")
    all_cycles = {}
    for dataset, battery_id in TEST_BATTERIES:
        cycles = load_cycles(dataset, battery_id, mit_subset)
        all_cycles[(dataset, battery_id)] = cycles
        print(f"[part-c]   {dataset}/{battery_id}: {len(cycles)} cycles")

    rng = np.random.default_rng(SEED)
    level_results = []
    per_battery_rows = []

    for level in NOISE_LEVELS:
        print(f"\n[part-c] === noise level: {level['name']} ===")
        y_true_all, y_pred_all, bid_all = [], [], []
        for dataset, battery_id in TEST_BATTERIES:
            cycles = all_cycles[(dataset, battery_id)]
            soh_map = soh_per_cycle(cycles)
            noisy_cycles = [add_noise(c, level["sigma_V"], level["sigma_I"], level["sigma_T"], rng) for c in cycles]

            hi_rows, tensors, y_true, valid_cyc = [], [], [], []
            for c in noisy_cycles:
                his = compute_health_indicators(c)
                his["cycle_idx"] = c["cycle_idx"]
                tensor = get_cycle_tensor(c)
                if tensor is None:
                    continue
                hi_rows.append(his)
                tensors.append(tensor)
                y_true.append(soh_map[c["cycle_idx"]])
                valid_cyc.append(c["cycle_idx"])

            hi_df_noisy = pd.DataFrame(hi_rows)
            # 1.1's reformulation, applied to THIS noisy trace's OWN cycle-10
            # reading (not the clean one) - the realistic deployment
            # behavior, since a real BMS never has access to a "clean"
            # reference reading either.
            base_row = hi_df_noisy[hi_df_noisy["cycle_idx"] == BASELINE_CYCLE]
            for feat in DURATION_FEATURES:
                rel_col = f"{feat}_rel"
                if len(base_row) > 0 and abs(float(base_row[feat].iloc[0])) > 1e-6:
                    base_val = float(base_row[feat].iloc[0])
                else:
                    base_val = float(hi_df_noisy[feat].median())
                    if not np.isfinite(base_val) or abs(base_val) < 1e-6:
                        base_val = 1.0  # last-resort guard, never hit in practice (checked in Stage 1.1)
                hi_df_noisy[rel_col] = hi_df_noisy[feat] / base_val
            # any remaining NaNs (e.g. a non-duration feature that failed
            # to compute for a given noisy cycle) are imputed below via
            # the SAME train-medians array used for the model's own
            # training, applied uniformly to X_pred right before predict()

            X_all = apply_channel_norm(np.stack(tensors).astype(np.float32), norm_stats)
            with torch.no_grad():
                emb = encoder.encode(torch.tensor(X_all[:, :, ICA_CHANNEL_SLICE])).numpy()

            feat_df = hi_df_noisy.reset_index(drop=True)
            for i in range(16):
                feat_df[f"fusion_{i}"] = emb[:, i]
            feat_df["cycle_idx"] = valid_cyc
            X_pred = feat_df[cols].to_numpy(dtype=float, copy=True)
            nan_inds = np.where(np.isnan(X_pred))
            if len(nan_inds[0]):
                X_pred[nan_inds] = np.take(medians, nan_inds[1])
            pred = model.predict(X_pred)

            y_true_all.extend(y_true)
            y_pred_all.extend(pred.tolist())
            bid_all.extend([battery_id] * len(y_true))

        y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
        r2 = float(r2_score(y_true_all, y_pred_all))
        print(f"[part-c] {level['name']}: n={len(y_true_all)} RMSE={rmse:.4f} R2={r2:.4f}")
        level_results.append({"noise_level": level["name"], "n": len(y_true_all), "rmse": rmse, "r2": r2})

        pb_df = pd.DataFrame({"battery_id": bid_all, "y_true": y_true_all, "y_pred": y_pred_all})
        for bid, g in pb_df.groupby("battery_id"):
            r2_b = float(r2_score(g.y_true, g.y_pred)) if g.y_true.std() > 0 else float("nan")
            rmse_b = float(np.sqrt(mean_squared_error(g.y_true, g.y_pred)))
            per_battery_rows.append({"noise_level": level["name"], "battery_id": bid, "r2": r2_b, "rmse": rmse_b})

    level_df = pd.DataFrame(level_results)
    print("\n[part-c] === pooled results across noise levels ===")
    print(level_df.to_string(index=False))

    pb_df = pd.DataFrame(per_battery_rows)
    piv = pb_df.pivot(index="battery_id", columns="noise_level", values="r2")
    print("\n[part-c] === per-battery R2 across noise levels (Stage 1.1+1.5 model) ===")
    print(piv.to_string())

    b0018_trend = piv.loc["B0018"] if "B0018" in piv.index else None
    print(f"\n[part-c] B0018 R2 trend across noise levels: {b0018_trend.to_dict() if b0018_trend is not None else 'N/A'}")
    if b0018_trend is not None:
        vals = b0018_trend[[l["name"] for l in NOISE_LEVELS]].to_numpy()
        monotonic_degrading = all(vals[i] >= vals[i+1] - 1e-6 for i in range(len(vals)-1))
        print(f"[part-c] B0018 degrades monotonically with noise level (session 26's original finding): {monotonic_degrading}")

    level_df.to_csv(OUT_DIR / "stage1_followup_partC_noise_pooled.csv", index=False)
    piv.reset_index().to_csv(OUT_DIR / "stage1_followup_partC_noise_perbattery.csv", index=False)
    print("\n[part-c] saved outputs/stage1_followup_partC_{grading_current_status,noise_pooled,noise_perbattery}.csv")
    print("[part-c] DONE")


if __name__ == "__main__":
    main()
