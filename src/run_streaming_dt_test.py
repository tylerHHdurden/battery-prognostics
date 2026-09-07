"""
Session 28: CLI verification of StreamingDigitalTwin BEFORE wiring it
into the Streamlit app - confirms it genuinely updates (corrector
coefficients visibly change, predictions differ from the frozen
one-shot baseline) rather than silently replaying static numbers, and
answers the two honesty questions the task asked directly:
  1. Does the online-updating prediction converge to the same/similar
     answer as the existing one-shot batch prediction?
  2. Does accuracy improve as more cycles stream in?

Run on NASA/B0018 (session 27's identified weak point - the most
informative test case, since its systematic late-life bias is exactly
what an online per-battery corrector should be able to learn) and
MIT/b3c35 (a well-behaved contrast case) for comparison.
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
from digital_twin_streaming import StreamingDigitalTwin, CONFORMAL_WINDOW, MIN_HISTORY_FOR_CORRECTION

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

START_CYCLE = 5  # "start with only a battery's first ~5 cycles" per instruction


def load_cycles(dataset, battery_id):
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    entry = next(e for e in mit_subset if e["global_id"] == battery_id)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def build_twin():
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

    # fallback conformal half-width: the pipeline's existing global
    # constant, reused as-is (not recomputed) for consistency with the
    # one-shot dashboard's own interval before the twin has enough local
    # evidence of its own
    constants = json.loads((PROC_DIR / "destandardization_constants.json").read_text())
    fallback_half_width = constants["soh_conformal_half_width"]

    def make_twin():
        return StreamingDigitalTwin(
            xgb_fusion, encoder, ocsvm, ocsvm_scaler, ocsvm_feature_cols,
            bfa_selected, train_medians, norm_stats, fallback_half_width,
        )
    return make_twin, fallback_half_width


def run_battery(dataset, battery_id, make_twin, fallback_half_width):
    print(f"\n[dt-test] === {dataset}/{battery_id} ===")
    cycles = load_cycles(dataset, battery_id)
    soh_map = soh_per_cycle(cycles)
    twin = make_twin()

    rows = []
    for c in cycles:
        true_soh = soh_map[c["cycle_idx"]]
        # step() is called identically for every cycle from 1 onward -
        # there is no separate code path for an "initial batch" vs.
        # "streamed" cycle. The task's "start with only the first ~5
        # cycles" is a UI/narrative framing (those first cycles are
        # already-available history the twin warm-starts from, shown
        # without an artificial per-cycle delay; the app then replays
        # cycle 6 onward one at a time WITH a delay to emulate arrival)
        # - the underlying online-update logic doesn't need, and
        # deliberately doesn't have, an arbitrary discontinuity at
        # cycle 5.
        result = twin.step(c, true_soh=true_soh)
        if "error" in result:
            continue
        rows.append(result)

    df = pd.DataFrame(rows)
    df["true_soh"] = df["true_soh"].astype(float)
    df["raw_abs_err"] = (df["raw_pred"] - df["true_soh"]).abs()
    df["corrected_abs_err"] = (df["corrected_pred"] - df["true_soh"]).abs()

    # --- test 1: does the corrector genuinely update? ---
    coefs = [r["corrector_coef"] for r in rows if r["corrector_coef"] is not None]
    if len(coefs) >= 2:
        coef_start, coef_end = coefs[0], coefs[-1]
        print(f"[dt-test] corrector coefficients: first={coef_start} -> last={coef_end} "
              f"({'CHANGED - genuinely updating' if not np.allclose(coef_start, coef_end) else 'UNCHANGED - static, investigate'})")
    else:
        print("[dt-test] not enough revealed cycles to confirm corrector updates")

    # --- test 2: one-shot batch baseline vs. final streamed (corrected) prediction ---
    final_row = df.iloc[-1]
    print(f"[dt-test] FINAL cycle {int(final_row['cycle_idx'])}: "
          f"one-shot/frozen raw_pred={final_row['raw_pred']:.2f}, "
          f"online-corrected pred={final_row['corrected_pred']:.2f}, "
          f"true_soh={final_row['true_soh']:.2f} "
          f"(raw error={final_row['raw_abs_err']:.2f}, corrected error={final_row['corrected_abs_err']:.2f})")

    # --- test 3: does accuracy improve over the stream (first half vs second half)? ---
    stream = df[df["cycle_idx"] >= START_CYCLE]
    half = len(stream) // 2
    first_half, second_half = stream.iloc[:half], stream.iloc[half:]
    print(f"[dt-test] RAW pipeline MAE: first half={first_half['raw_abs_err'].mean():.3f}, "
          f"second half={second_half['raw_abs_err'].mean():.3f}")
    print(f"[dt-test] ONLINE-CORRECTED MAE: first half={first_half['corrected_abs_err'].mean():.3f}, "
          f"second half={second_half['corrected_abs_err'].mean():.3f}")
    print(f"[dt-test] overall MAE: raw={stream['raw_abs_err'].mean():.3f}, "
          f"corrected={stream['corrected_abs_err'].mean():.3f} "
          f"({'IMPROVED' if stream['corrected_abs_err'].mean() < stream['raw_abs_err'].mean() else 'DID NOT improve'} "
          f"vs. frozen pipeline alone)")

    # --- test 4 (session 29): ACI half-width/coverage, reported more
    # fully than session 28's first-vs-last snapshot (mean/std/range too,
    # since a single first-vs-last comparison is exactly what made
    # session 28's "5x widening" look more dramatic than the full
    # trajectory necessarily warrants) ---
    covered = (stream["true_soh"] >= (stream["corrected_pred"] - stream["half_width"])) & \
              (stream["true_soh"] <= (stream["corrected_pred"] + stream["half_width"]))
    empirical_coverage = covered.mean()
    print(f"[dt-test] ACI half-width: first={stream['half_width'].iloc[0]:.3f}, "
          f"last={stream['half_width'].iloc[-1]:.3f}, mean={stream['half_width'].mean():.3f}, "
          f"std={stream['half_width'].std():.3f}, min={stream['half_width'].min():.3f}, "
          f"max={stream['half_width'].max():.3f}")
    print(f"[dt-test] ACI alpha_t: first={stream['alpha_t'].iloc[0]:.4f}, "
          f"last={stream['alpha_t'].iloc[-1]:.4f} (target alpha=0.10, i.e. 90% coverage)")
    print(f"[dt-test] EMPIRICAL COVERAGE over the stream: {empirical_coverage*100:.1f}% "
          f"(target 90%) over {len(stream)} streamed cycles")

    # --- fair apples-to-apples comparison against session 28's ORIGINAL
    # fixed-alpha sliding-window mechanism: session 28's own write-up
    # only reported a first-vs-last snapshot, not the full trajectory,
    # so replay that exact mechanism (fixed 0.9 quantile, same window,
    # same MIN_HISTORY threshold) on this run's IDENTICAL residual
    # sequence (the SGDRegressor corrector is unchanged, so corrected_pred
    # - and therefore every residual - is bit-for-bit the same either
    # way) rather than comparing ACI's full stats against session 28's
    # partial ones. ---
    # IMPORTANT: replay over the FULL df from cycle 1 (not just `stream`,
    # i.e. cycle >= START_CYCLE) - the real twin's residual/score buffer
    # already accumulated cycles 1..START_CYCLE-1's history by the time
    # cycle START_CYCLE is reached, so replaying only `stream`'s
    # residuals would incorrectly reset that history to empty and
    # understate how much buffer was actually available early in the
    # reported window. Caught by comparing this replay's "first" value
    # against session 28's actually-recorded first-cycle number before
    # trusting it - they didn't match until this fix.
    all_residuals = (df["true_soh"] - df["corrected_pred"]).abs().tolist()
    old_half_widths_full, old_buffer = [], []
    for r in all_residuals:
        if len(old_buffer) >= MIN_HISTORY_FOR_CORRECTION:
            old_half_widths_full.append(float(np.quantile(old_buffer[-CONFORMAL_WINDOW:], 0.9)))
        else:
            old_half_widths_full.append(fallback_half_width)
        old_buffer.append(r)
    df["old_half_width"] = old_half_widths_full
    old_hw = df.loc[stream.index, "old_half_width"].to_numpy()
    old_covered = np.array([
        abs(t - p) <= h for t, p, h in zip(stream["true_soh"], stream["corrected_pred"], old_hw)
    ])
    print(f"[dt-test] SESSION-28-ORIGINAL (fixed alpha=0.10 sliding window), replayed on this "
          f"SAME residual sequence for a fair full-trajectory comparison:")
    print(f"    half-width: first={old_hw[0]:.3f}, last={old_hw[-1]:.3f}, mean={old_hw.mean():.3f}, "
          f"std={old_hw.std():.3f}, min={old_hw.min():.3f}, max={old_hw.max():.3f}")
    print(f"    empirical coverage: {old_covered.mean()*100:.1f}% (target 90%)")

    df["covered"] = covered
    df.to_csv(OUT_DIR.parent / "data" / "processed" / "predictions" / f"streaming_dt_{dataset}_{battery_id}.csv",
              index=False)
    return df


def main():
    make_twin, fallback_half_width = build_twin()
    for dataset, battery_id in [("NASA", "B0018"), ("MIT", "b3c35")]:
        run_battery(dataset, battery_id, make_twin, fallback_half_width)
    print("\n[dt-test] DONE")


if __name__ == "__main__":
    main()
