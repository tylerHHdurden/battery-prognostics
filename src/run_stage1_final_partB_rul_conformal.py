"""
Stage 1 final closeout, Part B: RUL conformal recalibration against
session 41's new, substantially more accurate JointSOHRULModelFusion
(RUL R2 0.432->0.666) - every RUL coverage number on record (session
6's 83.3%, session 11's corrected 93.0%, Stage 1.6's 99.6-99.9%) was
calibrated against the OLD, much weaker joint model. Pure inference/
calibration work - JointSOHRULModelFusion is NOT retrained here.

Split convention: reuses Stage 1.6's own 3-way battery-level split
(this project's most recent established practice for RUL specifically)
- the joint model's own early-stopping VALIDATION battery set (never
gradient-updated) as the calibration group, genuinely distinct from
both fit and test batteries.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from train_deep_models import load_all_battery_tensors, make_xy
from sequence_features import apply_channel_norm
from stage1_common import canonical_feature_cols, add_reformulated_duration_features
from run_stage1_followup_partB_joint_rul import JointSOHRULModelFusion, build_hi_features
from run_conformal import split_conformal

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

ALPHA = 0.1
CV_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def evaluate_interval(y_true, pred, lo, hi):
    covered = (y_true >= lo) & (y_true <= hi)
    return {"coverage": float(covered.mean()), "avg_width": float(np.mean(hi - lo)), "n": len(y_true)}


def main():
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]  # calibration group (never gradient-updated)
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[rul-conformal] fit={len(fit_ids)} calib(val)={len(val_ids)} test={len(test_ids)} batteries")
    print(f"[rul-conformal] calibration batteries: {val_ids}")

    X_fit, soh_fit, rul_fit, _, bid_fit, cyc_fit = make_xy(battery_data, fit_ids)
    X_calib, soh_calib, rul_calib, _, bid_calib, cyc_calib = make_xy(battery_data, val_ids)
    X_test, soh_test, rul_test, _, bid_test, cyc_test = make_xy(battery_data, test_ids)

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_calib = apply_channel_norm(X_calib, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_reformulated = add_reformulated_duration_features(hi_full)
    feature_cols = canonical_feature_cols(reformulated=True)

    fit_key_df = pd.DataFrame({"battery_id": bid_fit, "cycle_idx": cyc_fit}).merge(
        hi_reformulated[["battery_id", "cycle_idx"] + feature_cols], on=["battery_id", "cycle_idx"], how="left")
    train_medians = fit_key_df[feature_cols].median(numeric_only=True).to_numpy()

    HI_fit = build_hi_features(bid_fit, cyc_fit, hi_reformulated, feature_cols, train_medians)
    HI_calib = build_hi_features(bid_calib, cyc_calib, hi_reformulated, feature_cols, train_medians)
    HI_test = build_hi_features(bid_test, cyc_test, hi_reformulated, feature_cols, train_medians)
    hi_mean, hi_std = HI_fit.mean(axis=0), HI_fit.std(axis=0) + 1e-8
    HI_calib = (HI_calib - hi_mean) / hi_std
    HI_test = (HI_test - hi_mean) / hi_std

    model = JointSOHRULModelFusion(n_hi_features=len(feature_cols))
    model.load_state_dict(torch.load(ROOT / "models" / "joint_adaptive_fusion_canonical.pt"))
    model.eval()

    # recover the EXACT rul_mean/std/soh_mean/std the model was trained
    # with - standardization stats are fit-set-derived and not saved in
    # the .pt state_dict, so recompute identically from fit_ids (same
    # convention run_conformal.py's own RUL section uses).
    rul_mean, rul_std = float(rul_fit.mean()), float(rul_fit.std() + 1e-8)

    with torch.no_grad():
        _, rul_pred_calib_z = model(torch.tensor(X_calib), torch.tensor(HI_calib))
        _, rul_pred_test_z = model(torch.tensor(X_test), torch.tensor(HI_test))
    rul_pred_calib = rul_pred_calib_z.squeeze(-1).numpy() * rul_std + rul_mean
    rul_pred_test = rul_pred_test_z.squeeze(-1).numpy() * rul_std + rul_mean

    from sklearn.metrics import r2_score, mean_squared_error
    print(f"[rul-conformal] sanity check - test RUL R2 with this inference path: "
          f"{r2_score(rul_test, rul_pred_test):.4f} (session 41 Part B's own reported test R2: 0.6657)")

    # --- plain split-conformal, 3-way split (this project's current RUL convention) ---
    _, lo, hi, method = split_conformal(rul_pred_calib, rul_calib, rul_pred_test, ALPHA)
    result = evaluate_interval(rul_test, rul_pred_test, lo, hi)
    print(f"\n[rul-conformal] === split-conformal (3-way split) on JointSOHRULModelFusion ===")
    print(f"[rul-conformal] method={method} coverage={result['coverage']:.4f} avg_width={result['avg_width']:.3f} "
          f"(n_test={result['n']}, n_calib={len(rul_calib)})")

    print(f"\n[rul-conformal] === COMPARISON vs. most recent prior number on record ===")
    print(f"[rul-conformal] Stage 1.6 (OLD joint-adaptive model, 3-way split): coverage=99.6%-99.9% (across seeds)")
    print(f"[rul-conformal] session 11 corrected: 93.0%  |  session 6 original: 83.3%")
    print(f"[rul-conformal] JointSOHRULModelFusion (this run, new model): coverage={result['coverage']*100:.1f}% "
          f"avg_width={result['avg_width']:.1f} cycles")

    if result['coverage'] >= 0.90:
        outcome = "(a) coverage remains AT or ABOVE the 90% target despite the large accuracy gain."
    elif result['coverage'] >= 0.80:
        outcome = ("(b) coverage is BELOW the 90% target and meaningfully lower than the prior 99.6-99.9% "
                  "number - the same accuracy-improves/coverage-gets-worse mechanism session 38 Part A found for SOH.")
    else:
        outcome = ("(b) coverage COLLAPSES well below the 90% target and far below the prior 99.6-99.9% number - "
                  "a severe version of the same mechanism session 38 Part A found for SOH.")
    print(f"\n[rul-conformal] OUTCOME: {outcome}")

    # --- optional: cheap Jackknife+/CV+ via a linear recalibration layer over the RUL point predictions ---
    print(f"\n[rul-conformal] === OPTIONAL: Jackknife+/CV+ via a cheap Ridge recalibration layer ===")
    print("[rul-conformal] NOTE: no established precedent for this exists for RUL in this project (Stage 1.6 "
          "scoped RUL down to plain split-conformal only, citing deep-model retrain cost) - this is a NEW "
          "construction for this pass, analogous to SOH's own Ridge-over-fixed-predictions trick, stated "
          "explicitly rather than presented as reusing prior established practice.")
    calib_groups = np.array(bid_calib)
    n_calib_batteries = len(set(calib_groups))
    X_calib_ridge = np.column_stack([rul_pred_calib, cyc_calib])
    X_test_ridge = np.column_stack([rul_pred_test, cyc_test])

    from mapie.regression import CrossConformalRegressor
    cv_loo = GroupKFold(n_splits=n_calib_batteries)
    mapie_jk = CrossConformalRegressor(estimator=Ridge(), confidence_level=1 - ALPHA, method="plus", cv=cv_loo)
    mapie_jk.fit_conformalize(X_calib_ridge, rul_calib, groups=calib_groups)
    pred_jk, interval_jk = mapie_jk.predict_interval(X_test_ridge)
    lo_jk, hi_jk = interval_jk[:, 0, 0], interval_jk[:, 1, 0]
    jk_result = evaluate_interval(rul_test, pred_jk, lo_jk, hi_jk)
    print(f"[rul-conformal] Jackknife+ (K={n_calib_batteries}, leave-one-battery-out): "
          f"coverage={jk_result['coverage']:.4f} avg_width={jk_result['avg_width']:.3f}")
    print(f"[rul-conformal] vs. plain split-conformal: coverage={result['coverage']:.4f} avg_width={result['avg_width']:.3f}")

    pd.DataFrame([
        {"method": "split_conformal_3way", "coverage": result["coverage"], "avg_width": result["avg_width"]},
        {"method": "jackknife_plus_recalibration", "coverage": jk_result["coverage"], "avg_width": jk_result["avg_width"]},
    ]).to_csv(OUT_DIR / "stage1_final_partB_rul_conformal.csv", index=False)
    print("\n[rul-conformal] saved outputs/stage1_final_partB_rul_conformal.csv")
    print("[rul-conformal] DONE")


if __name__ == "__main__":
    main()
