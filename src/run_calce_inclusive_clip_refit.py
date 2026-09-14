"""
Direct follow-up to the clip-saturation session: refit dQdV/dVdQ clip
bounds using the UNION of NASA+MIT training data and CALCE's own raw,
UNLABELED dQdV/dVdQ curves, then re-run the existing zero-retrain
CALCE evaluation with the SAME already-trained model (ica_encoder.pt
and the 1.1+1.5 XGBoost-fusion config, both unchanged in weights/
training procedure) to see whether R2 moves.

Zero-label-leakage discipline (same basis as MMD/CORAL's own use of
CALCE's unlabeled curves): only CALCE's raw INPUT FEATURE RANGES feed
into the new clip-bound computation - no SOH/RUL label from CALCE is
ever read here.

Scope, per instruction: ONLY channels 3 (dQdV) and 4 (dVdQ) - the two
channels confirmed to actually reach ICA_CHANNEL_SLICE / the fusion
embedding - are touched. V_t (0), I_t (1), T_t (2), dIdV (5) keep
their ORIGINAL channel_norm_stats.json bounds unchanged. The NASA+MIT
training-side fusion embeddings (fusion_embeddings.csv) and
ica_encoder.pt's weights are completely untouched - only CALCE's own
input tensor is renormalized with the new bounds before being passed
through the SAME, already-trained encoder. XGBoost-fusion is refit via
the exact same deterministic procedure (fixed random_state, same
NASA+MIT training data/embeddings) used throughout Stage 1-3 - not a
new training decision, the same reproducible step every comparison in
this project already relies on.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, CALCE_CELLS, ICA_CHANNEL_SLICE, OUT_DIR, PROC_DIR, ROOT,
)
from sequence_features import build_dataset_tensors, apply_channel_norm, CHANNEL_NAMES
from data_adapters import iterate_calce_cycles
from models.ica_encoder import ICAEncoder
from run_clip_saturation_sweep import per_battery_channel_stats
from train_deep_models import load_all_battery_tensors, make_xy

DQDV_IDX, DVDQ_IDX = 3, 4


def main():
    print("[calce-clip-refit] === Item 1: fix already re-verified against B0053 above this script ===")

    # -----------------------------------------------------------------
    # Item 2: refit dQdV/dVdQ clip bounds on the UNION of NASA+MIT
    # training data (X_fit, exactly as train_deep_models.py builds it)
    # and CALCE's raw, unlabeled dQdV/dVdQ values (all 3 cells).
    # -----------------------------------------------------------------
    print("\n[calce-clip-refit] === building NASA+MIT X_fit (same protocol as train_deep_models.py) ===")
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    X_fit, y_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    print(f"[calce-clip-refit] X_fit: {len(fit_ids)} fit batteries, {X_fit.shape[0]} cycles")

    original_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())

    print("\n[calce-clip-refit] === building CALCE raw (pre-clip) tensors, all 3 cells ===")
    calce_raw = {}
    all_calce_X = []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        calce_raw[cid] = X
        all_calce_X.append(X)
        print(f"[calce-clip-refit] CALCE/{cid}: {X.shape[0]} cycles")
    X_calce_all = np.concatenate(all_calce_X, axis=0)

    new_stats = [dict(s) for s in original_stats]  # copy; channels 0,1,2,5 stay untouched
    print("\n[calce-clip-refit] === new bounds vs. original (NASA+MIT-only) bounds ===")
    for idx, name in [(DQDV_IDX, "dQdV"), (DVDQ_IDX, "dVdQ")]:
        train_vals = X_fit[:, :, idx].flatten()
        calce_vals = X_calce_all[:, :, idx].flatten()
        union_vals = np.concatenate([train_vals, calce_vals])
        new_lo, new_hi = np.percentile(union_vals, [1.0, 99.0])
        old_lo, old_hi = original_stats[idx]["lo"], original_stats[idx]["hi"]

        # recompute mean/std from X_fit ONLY, clipped to the NEW (wider) bounds -
        # keeps the training-distribution-centered z-score basis unchanged,
        # only the clip RANGE widens (per instruction: "refit clip bounds",
        # not "refit the whole normalization on CALCE-inclusive data")
        train_clipped = np.clip(train_vals, new_lo, new_hi)
        new_mean, new_std = float(train_clipped.mean()), float(train_clipped.std() + 1e-8)
        new_stats[idx] = {"lo": float(new_lo), "hi": float(new_hi), "mean": new_mean, "std": new_std}

        calce_min, calce_max = float(calce_vals.min()), float(calce_vals.max())
        train_min, train_max = float(train_vals.min()), float(train_vals.max())
        print(f"[calce-clip-refit] {name}: OLD bounds [{old_lo:.4g}, {old_hi:.4g}] -> "
              f"NEW bounds [{new_lo:.4g}, {new_hi:.4g}]")
        print(f"[calce-clip-refit] {name}: NASA+MIT raw range [{train_min:.4g}, {train_max:.4g}], "
              f"CALCE raw range [{calce_min:.4g}, {calce_max:.4g}]")
        lo_shift_frac = abs(new_lo - old_lo) / (abs(old_lo) + 1e-9)
        hi_shift_frac = abs(new_hi - old_hi) / (abs(old_hi) + 1e-9)
        # is the new bound closer to CALCE's own extreme than to the old NASA+MIT bound?
        lo_pulled_by_calce = abs(new_lo - calce_min) < abs(new_lo - old_lo)
        hi_pulled_by_calce = abs(new_hi - calce_max) < abs(new_hi - old_hi)
        print(f"[calce-clip-refit] {name}: lo shifted {lo_shift_frac*100:.1f}%, hi shifted {hi_shift_frac*100:.1f}% "
              f"(lo pulled toward CALCE's own min: {lo_pulled_by_calce}, hi pulled toward CALCE's own max: {hi_pulled_by_calce})")

    print(f"\n[calce-clip-refit] === confirming V_t/I_t/T_t/dIdV bounds UNTOUCHED ===")
    for idx in [0, 1, 2, 5]:
        unchanged = (new_stats[idx]["lo"] == original_stats[idx]["lo"] and
                     new_stats[idx]["hi"] == original_stats[idx]["hi"])
        print(f"[calce-clip-refit] channel {idx} ({CHANNEL_NAMES[idx]}): unchanged={unchanged}")

    # -----------------------------------------------------------------
    # Item 3: recompute CALCE's fusion embedding under the NEW bounds,
    # using the SAME already-trained ica_encoder.pt (no retraining),
    # then re-run the SAME 1.1+1.5 XGBoost-fusion config.
    # -----------------------------------------------------------------
    print("\n[calce-clip-refit] === Item 3: re-evaluating CALCE with the NEW bounds (same trained encoder) ===")
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()

    def embed_calce(cid, X_raw, stats):
        X_norm = apply_channel_norm(X_raw, stats)
        with torch.no_grad():
            emb = encoder.encode(torch.tensor(X_norm[:, :, ICA_CHANNEL_SLICE])).numpy()
        return emb

    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    from stage1_common import add_reformulated_duration_features
    hi_full = add_reformulated_duration_features(hi_full)
    calce_hi = hi_full[hi_full["dataset"] == "CALCE"]

    def build_calce_merged_with_stats(stats):
        frames = []
        for cid in CALCE_CELLS:
            X_raw = calce_raw[cid]
            cycles = list(iterate_calce_cycles(cid))
            _, soh, rul, idxs, censored = build_dataset_tensors(cycles)
            emb = embed_calce(cid, X_raw, stats)
            seq_df = pd.DataFrame({"battery_id": cid, "cycle_idx": idxs})
            for i in range(16):
                seq_df[f"fusion_{i}"] = emb[:, i]
            frames.append(seq_df)
        seq_all = pd.concat(frames, ignore_index=True)
        return pd.merge(calce_hi, seq_all, on=["battery_id", "cycle_idx"], how="inner")

    print("[calce-clip-refit] building CALCE-merged frame under NEW (CALCE-inclusive) bounds...")
    calce_merged_new = build_calce_merged_with_stats(new_stats)
    print("[calce-clip-refit] building CALCE-merged frame under ORIGINAL bounds (re-verification)...")
    calce_merged_orig = build_calce_merged_with_stats(original_stats)

    # --- train the SAME 1.1+1.5 canonical model (deterministic, same
    # procedure every Stage 1-3 script uses; NASA+MIT training embeddings
    # untouched - only CALCE's own embeddings differ between the two runs) ---
    merged, hi_full2 = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split2 = battery_split_masks(merged)
    base_features = canonical_feature_cols(reformulated=True)
    feature_cols = base_features + ["cycle_idx"]
    from stage1_common import fusion_cols
    fcols = fusion_cols()
    monotone = tuple([0] * len(base_features) + [-1] + [0] * len(fcols))
    model, medians, cols = fit_xgb(merged, train_mask, feature_cols, xgb_extra_kwargs={"monotone_constraints": monotone})
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    print(f"[calce-clip-refit] in-domain R2={indomain['r2']:.4f} (sanity check, should match Stage 1's 0.9750)")

    def eval_calce_df(calce_merged, cols, medians):
        X = calce_merged[cols].to_numpy(dtype=float, copy=True)
        for j, col in enumerate(feature_cols):
            nan_mask = np.isnan(X[:, j])
            if nan_mask.any():
                X[nan_mask, j] = medians[j]
        y_true = calce_merged["SOH"].to_numpy()
        pred = model.predict(X)
        return y_true, pred, calce_merged["battery_id"].to_numpy()

    def metrics(y_true, y_pred):
        return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred))}

    y_orig, pred_orig, bid_orig = eval_calce_df(calce_merged_orig, cols, medians)
    y_new, pred_new, bid_new = eval_calce_df(calce_merged_new, cols, medians)

    m_orig_pooled = metrics(y_orig, pred_orig)
    m_new_pooled = metrics(y_new, pred_new)
    print(f"\n[calce-clip-refit] verification (should reproduce Stage 1's canonical R2=0.5672): "
          f"pooled R2={m_orig_pooled['r2']:.4f} RMSE={m_orig_pooled['rmse']:.4f}")
    reproduces = abs(m_orig_pooled['r2'] - 0.5672) < 0.01
    print(f"[calce-clip-refit] reproduces the canonical 1.1+1.5 baseline: {reproduces}")

    print(f"\n[calce-clip-refit] === POOLED RESULT: original bounds vs. CALCE-inclusive bounds ===")
    print(f"[calce-clip-refit] ORIGINAL bounds:        R2={m_orig_pooled['r2']:.4f} RMSE={m_orig_pooled['rmse']:.4f}")
    print(f"[calce-clip-refit] CALCE-inclusive bounds:  R2={m_new_pooled['r2']:.4f} RMSE={m_new_pooled['rmse']:.4f}")
    print(f"[calce-clip-refit] delta R2: {m_new_pooled['r2']-m_orig_pooled['r2']:+.4f}")

    print(f"\n[calce-clip-refit] === PER-CELL RESULT (Stage 2.5 convention: per-battery first) ===")
    per_cell_rows = []
    for cid in CALCE_CELLS:
        mo = np.array(bid_orig) == cid
        mn = np.array(bid_new) == cid
        m_o = metrics(y_orig[mo], pred_orig[mo])
        m_n = metrics(y_new[mn], pred_new[mn])
        print(f"[calce-clip-refit] {cid}: original R2={m_o['r2']:.4f} RMSE={m_o['rmse']:.4f}  |  "
              f"new R2={m_n['r2']:.4f} RMSE={m_n['rmse']:.4f}  |  delta R2={m_n['r2']-m_o['r2']:+.4f}")
        per_cell_rows.append({"cell": cid, "r2_original": m_o['r2'], "rmse_original": m_o['rmse'],
                               "r2_new": m_n['r2'], "rmse_new": m_n['rmse'],
                               "delta_r2": m_n['r2'] - m_o['r2']})

    # -----------------------------------------------------------------
    # Item 3.4: recompute saturation fractions under the NEW bounds
    # -----------------------------------------------------------------
    print(f"\n[calce-clip-refit] === saturation fractions, dQdV/dVdQ, OLD vs NEW bounds, per cell ===")
    sat_rows = []
    for cid in CALCE_CELLS:
        X_raw = calce_raw[cid]
        for idx, name in [(DQDV_IDX, "dQdV"), (DVDQ_IDX, "dVdQ")]:
            old_lo, old_hi = original_stats[idx]["lo"], original_stats[idx]["hi"]
            new_lo, new_hi = new_stats[idx]["lo"], new_stats[idx]["hi"]
            sat_old, clip_old = per_battery_channel_stats(X_raw[:, :, idx], old_lo, old_hi)
            sat_new, clip_new = per_battery_channel_stats(X_raw[:, :, idx], new_lo, new_hi)
            print(f"[calce-clip-refit] {cid}/{name}: frac_clipped OLD={clip_old*100:.1f}% -> NEW={clip_new*100:.1f}% "
                  f"(fully-saturated cycles OLD={sat_old*100:.1f}% -> NEW={sat_new*100:.1f}%)")
            sat_rows.append({"cell": cid, "channel": name, "frac_clipped_old": clip_old,
                              "frac_clipped_new": clip_new, "frac_sat_old": sat_old, "frac_sat_new": sat_new})

    # -----------------------------------------------------------------
    # Item 4: honest interpretation
    # -----------------------------------------------------------------
    delta = m_new_pooled['r2'] - m_orig_pooled['r2']
    gap_to_indomain_orig = indomain['r2'] - m_orig_pooled['r2']
    frac_gap_closed = delta / gap_to_indomain_orig if gap_to_indomain_orig != 0 else float("nan")
    print(f"\n[calce-clip-refit] === HONEST INTERPRETATION ===")
    print(f"[calce-clip-refit] baseline compared against: 1.1+1.5 canonical CALCE R2=0.5672 "
          f"(the CURRENT, most rigorously-established Stage 1 canonical number, same exact "
          f"model/feature config as this follow-up - the only variable that changes here is "
          f"the CALCE clip-bound preprocessing step; the older pre-Stage-1 R2~0.31 baseline "
          f"used a different feature/model config entirely and is not the right comparator here)")
    print(f"[calce-clip-refit] delta R2 = {delta:+.4f} ({delta/m_orig_pooled['r2']*100:+.1f}% relative to the 0.5672 baseline)")
    print(f"[calce-clip-refit] in-domain R2 = {indomain['r2']:.4f}; original CALCE gap = {gap_to_indomain_orig:.4f}; "
          f"this fix closes {frac_gap_closed*100:.1f}% of that gap" if not np.isnan(frac_gap_closed) else "")

    if delta > 0.02:
        outcome = "(a) R2 improves meaningfully - a real, quantifiable fraction of the CALCE collapse was avoidable preprocessing loss."
    elif delta < -0.02:
        outcome = "(c) R2 moves in an unexpected direction (WORSE) - reported honestly, investigated below rather than dismissed."
    else:
        outcome = "(b) R2 barely moves - the clip-bound confound, while real and worth disclosing, is NOT a major driver of the CALCE collapse."
    print(f"[calce-clip-refit] OUTCOME: {outcome}")

    pd.DataFrame(per_cell_rows).to_csv(OUT_DIR / "calce_inclusive_clip_refit_percell.csv", index=False)
    pd.DataFrame(sat_rows).to_csv(OUT_DIR / "calce_inclusive_clip_refit_saturation.csv", index=False)
    pd.DataFrame([
        {"variant": "original_bounds", "r2": m_orig_pooled['r2'], "rmse": m_orig_pooled['rmse']},
        {"variant": "calce_inclusive_bounds", "r2": m_new_pooled['r2'], "rmse": m_new_pooled['rmse']},
    ]).to_csv(OUT_DIR / "calce_inclusive_clip_refit_pooled.csv", index=False)
    print("\n[calce-clip-refit] saved outputs/calce_inclusive_clip_refit_{pooled,percell,saturation}.csv")
    print("[calce-clip-refit] DONE")


if __name__ == "__main__":
    main()
