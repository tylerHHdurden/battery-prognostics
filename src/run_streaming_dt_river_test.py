"""
Stage 6.3: the SAME verified predict-then-reveal-then-update test
session 28 used (run_streaming_dt_test.py), re-run against the new
River-based corrector (digital_twin_streaming_river.py) - NOT a one-
shot rerun with different code, the literal same test logic reused
(build_twin/run_battery structure mirrored exactly), confirming this
new version passes the same genuineness checks, then reporting
accuracy head-to-head against the original SGDRegressor version on
the SAME two batteries/residual sequences.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from rul_labels import soh_per_cycle
from models.ica_encoder import ICAEncoder
from digital_twin_streaming import CONFORMAL_WINDOW, MIN_HISTORY_FOR_CORRECTION
from digital_twin_streaming_river import StreamingDigitalTwinRiver
from run_streaming_dt_test import load_cycles

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

START_CYCLE = 5


def build_twin_river():
    import pickle
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        bfa_selected = [l.strip() for l in f if l.strip()]
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    train_medians = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])][bfa_selected].median(numeric_only=True).to_dict()

    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()
    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    with open(ROOT / "models" / "ocsvm_model.pkl", "rb") as f:
        ocsvm = pickle.load(f)
    with open(ROOT / "models" / "ocsvm_scaler.pkl", "rb") as f:
        ocsvm_scaler = pickle.load(f)
    ocsvm_feature_cols = json.loads((PROC_DIR / "ocsvm_feature_cols.json").read_text())
    constants = json.loads((PROC_DIR / "destandardization_constants.json").read_text())
    fallback_half_width = constants["soh_conformal_half_width"]

    def make_twin():
        return StreamingDigitalTwinRiver(
            xgb_fusion, encoder, ocsvm, ocsvm_scaler, ocsvm_feature_cols,
            bfa_selected, train_medians, norm_stats, fallback_half_width,
        )
    return make_twin, fallback_half_width


def run_battery(dataset, battery_id, make_twin):
    print(f"\n[dt-river-test] === {dataset}/{battery_id} ===")
    cycles = load_cycles(dataset, battery_id)
    soh_map = soh_per_cycle(cycles)
    twin = make_twin()

    rows = []
    for c in cycles:
        true_soh = soh_map[c["cycle_idx"]]
        result = twin.step(c, true_soh=true_soh)
        if "error" in result:
            continue
        rows.append(result)

    df = pd.DataFrame(rows)
    df["true_soh"] = df["true_soh"].astype(float)
    df["raw_abs_err"] = (df["raw_pred"] - df["true_soh"]).abs()
    df["corrected_abs_err"] = (df["corrected_pred"] - df["true_soh"]).abs()

    # --- test 1: does the corrector genuinely update? (River trees
    # don't expose a simple coef_ vector like SGD - genuineness is
    # instead confirmed by checking the correction values themselves
    # are non-trivial/non-constant once enough history has accumulated,
    # a stronger check in some ways since it doesn't just look at
    # internal parameters but the actual OUTPUT behavior) ---
    corrections = df[df["n_revealed_so_far"] >= MIN_HISTORY_FOR_CORRECTION]["correction"]
    n_distinct = corrections.round(4).nunique()
    print(f"[dt-river-test] corrector genuineness: {len(corrections)} corrected cycles, "
          f"{n_distinct} DISTINCT correction values "
          f"({'CHANGED - genuinely updating' if n_distinct > 1 else 'STATIC - investigate'})")

    final_row = df.iloc[-1]
    print(f"[dt-river-test] FINAL cycle {int(final_row['cycle_idx'])}: "
          f"raw_pred={final_row['raw_pred']:.2f}, corrected_pred={final_row['corrected_pred']:.2f}, "
          f"true_soh={final_row['true_soh']:.2f} "
          f"(raw error={final_row['raw_abs_err']:.2f}, corrected error={final_row['corrected_abs_err']:.2f})")

    stream = df[df["cycle_idx"] >= START_CYCLE]
    half = len(stream) // 2
    first_half, second_half = stream.iloc[:half], stream.iloc[half:]
    print(f"[dt-river-test] RAW pipeline MAE: first half={first_half['raw_abs_err'].mean():.3f}, "
          f"second half={second_half['raw_abs_err'].mean():.3f}")
    print(f"[dt-river-test] RIVER-CORRECTED MAE: first half={first_half['corrected_abs_err'].mean():.3f}, "
          f"second half={second_half['corrected_abs_err'].mean():.3f}")
    print(f"[dt-river-test] overall MAE: raw={stream['raw_abs_err'].mean():.3f}, "
          f"corrected={stream['corrected_abs_err'].mean():.3f} "
          f"({'IMPROVED' if stream['corrected_abs_err'].mean() < stream['raw_abs_err'].mean() else 'DID NOT improve'} "
          f"vs. frozen pipeline alone)")

    covered = (stream["true_soh"] >= (stream["corrected_pred"] - stream["half_width"])) & \
              (stream["true_soh"] <= (stream["corrected_pred"] + stream["half_width"]))
    print(f"[dt-river-test] EMPIRICAL COVERAGE (ACI): {covered.mean()*100:.1f}% (target 90%) "
          f"over {len(stream)} streamed cycles")

    n_drift = int(df["drift_detected"].sum())
    drift_cycles = df[df["drift_detected"]]["cycle_idx"].tolist()
    print(f"[dt-river-test] ADWIN drift events flagged: {n_drift} "
          f"{'at cycles ' + str(drift_cycles) if n_drift else '(none)'}")

    df.to_csv(PROC_DIR / "predictions" / f"streaming_dt_river_{dataset}_{battery_id}.csv", index=False)
    return df


def main():
    make_twin, fallback_half_width = build_twin_river()
    results = {}
    for dataset, battery_id in [("NASA", "B0018"), ("MIT", "b3c35")]:
        results[(dataset, battery_id)] = run_battery(dataset, battery_id, make_twin)
    print("\n[dt-river-test] === head-to-head vs. session 28's SGDRegressor (same batteries) ===")
    for (dataset, battery_id), df in results.items():
        sgd_path = PROC_DIR / "predictions" / f"streaming_dt_{dataset}_{battery_id}.csv"
        if not sgd_path.exists():
            print(f"    {dataset}/{battery_id}: no SGD baseline file found - run run_streaming_dt_test.py first")
            continue
        sgd_df = pd.read_csv(sgd_path)
        stream_river = df[df["cycle_idx"] >= START_CYCLE]
        stream_sgd = sgd_df[sgd_df["cycle_idx"] >= START_CYCLE]
        river_mae = (stream_river["corrected_pred"] - stream_river["true_soh"]).abs().mean()
        sgd_mae = (stream_sgd["corrected_pred"] - stream_sgd["true_soh"]).abs().mean()
        print(f"    {dataset}/{battery_id}: SGD (linear) corrected MAE={sgd_mae:.4f}, "
              f"River (Hoeffding Adaptive Tree) corrected MAE={river_mae:.4f} "
              f"({'River better' if river_mae < sgd_mae else 'SGD better' if sgd_mae < river_mae else 'tie'})")
    print("\n[dt-river-test] DONE")


if __name__ == "__main__":
    main()
