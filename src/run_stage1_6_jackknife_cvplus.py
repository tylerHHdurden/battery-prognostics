"""
Stage 1, Item 1.6: Jackknife+/CV+ with a genuine three-way battery-level
split, replacing the existing 2-way calibration/evaluation split that
session 11 found produces 64.7%-99.6% coverage swings purely from which
3 of 6 test batteries land in calibration vs. eval.

SOH (genuine Jackknife+/CV+, cost-free): reuses Stage 0 Check 0.1's
already-computed GENUINELY out-of-fold base-learner predictions
(oof_stacking_check_meta_features.csv, 26 original-pool TRAIN batteries,
each one's predictions coming from a model that never saw it during
fitting) as the CALIBRATION pool - satisfying "held out from training
batteries, never used to fit any base learner" without any new
retraining. The RIDGE META-LEARNER is the piece MAPIE's CV+/Jackknife+
actually cross-validates (a Ridge refit is near-instant, so K-fold /
leave-one-battery-out refitting is cheap) - this is honest, explicit
scoping: retraining all 4 DEEP base learners K times over would repeat
Check 0.1's ~2.7h cost K times, well outside this item's budget.
Evaluation is the existing, already-legitimately-out-of-sample 6-battery
TEST set (unchanged).

RUL (SCOPED DOWN, stated explicitly - NOT genuine Jackknife+/CV+): the
RUL point predictor is Phase 4's joint-adaptive deep model. Genuine
Jackknife+/CV+ would require retraining this deep model K times (the
same cost class Check 0.1 already flagged as out of budget) - not
attempted here. Instead, this item applies the STRUCTURAL fix (a
calibration group distinct from the test set) using plain split-
conformal: the joint model's own VALIDATION battery split (5 batteries,
used only for early-stopping EPOCH SELECTION, never for a gradient
update - see train_deep_models.py) is repurposed as the calibration
group, genuinely distinct from both the fit batteries (gradient-trained
on) and the test batteries (final eval, unchanged). This is honestly a
partial implementation of this item for RUL - the structural 3-way-
split fix is real, the Jackknife+/CV+ refitting-based method is not -
stated plainly here and again in the final report, not glossed over.
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
from mapie.regression import CrossConformalRegressor
from train_ensemble import load_merged
from train_deep_models import load_all_battery_tensors, make_xy
from sequence_features import apply_channel_norm
from models.joint_model import JointSOHRULModel
from run_conformal import split_conformal

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

ALPHA = 0.1
BASE_COLS = ["pred_XGBoost", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
CV_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def evaluate_interval(y_true, pred, lo, hi):
    covered = (y_true >= lo) & (y_true <= hi)
    return {"coverage": float(covered.mean()), "avg_width": float(np.mean(hi - lo)), "n": len(y_true)}


def soh_jackknife_cvplus():
    print("\n[stage1.6] === SOH: genuine Jackknife+/CV+ via MAPIE, calibration pool = Check 0.1's OOF predictions ===")
    oof = pd.read_csv(PRED_DIR / "oof_stacking_check_meta_features.csv")
    X_calib = oof[BASE_COLS].to_numpy()
    y_calib = oof["SOH"].to_numpy()
    groups = oof["battery_id"].to_numpy()
    n_batteries = len(set(groups))
    print(f"[stage1.6] SOH calibration pool: {len(oof)} rows, {n_batteries} batteries (genuinely OOF w.r.t. base learners)")

    test_df = load_merged("test")
    X_test = test_df[BASE_COLS].to_numpy()
    y_test = test_df["SOH"].to_numpy()
    print(f"[stage1.6] SOH test set (unchanged, already out-of-sample): {len(test_df)} rows, "
          f"{test_df['battery_id'].nunique()} batteries")

    # --- ORIGINAL method, for reference: plain split-conformal, calib/eval both from the SAME small test pool ---
    from run_conformal import calib_eval_battery_split
    with open(ROOT / "models" / "ridge_meta.pkl", "rb") as f:
        import pickle
        ridge_orig = pickle.load(f)
    test_df["pred_orig"] = ridge_orig.predict(X_test)
    orig_calib_ids, orig_eval_ids = calib_eval_battery_split(test_df["battery_id"].unique().tolist())
    orig_calib = test_df[test_df.battery_id.isin(orig_calib_ids)]
    orig_eval = test_df[test_df.battery_id.isin(orig_eval_ids)]
    _, lo, hi, _ = split_conformal(orig_calib["pred_orig"].to_numpy(), orig_calib["SOH"].to_numpy(),
                                    orig_eval["pred_orig"].to_numpy(), ALPHA)
    orig_result = evaluate_interval(orig_eval["SOH"].to_numpy(), orig_eval["pred_orig"].to_numpy(), lo, hi)
    print(f"[stage1.6] ORIGINAL method (calib/eval both carved from the 6-battery test pool): "
          f"coverage={orig_result['coverage']:.3f} avg_width={orig_result['avg_width']:.3f} (n={orig_result['n']})")

    # --- Jackknife+ : leave-one-battery-out (K = n_batteries), deterministic, no seed dependence ---
    cv_loo = GroupKFold(n_splits=n_batteries)
    mapie_jk = CrossConformalRegressor(estimator=Ridge(), confidence_level=1 - ALPHA, method="plus", cv=cv_loo)
    mapie_jk.fit_conformalize(X_calib, y_calib, groups=groups)
    pred_jk, interval_jk = mapie_jk.predict_interval(X_test)
    lo_jk, hi_jk = interval_jk[:, 0, 0], interval_jk[:, 1, 0]
    jk_result = evaluate_interval(y_test, pred_jk, lo_jk, hi_jk)
    print(f"[stage1.6] JACKKNIFE+ (leave-one-battery-out, K={n_batteries}, deterministic): "
          f"coverage={jk_result['coverage']:.3f} avg_width={jk_result['avg_width']:.3f}")

    # --- CV+ : K=5 battery-grouped folds, repeated across several seeds for stability ---
    cvplus_coverages = []
    for seed in CV_SEEDS:
        cv5 = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
        mapie_cv = CrossConformalRegressor(estimator=Ridge(), confidence_level=1 - ALPHA, method="plus", cv=cv5)
        mapie_cv.fit_conformalize(X_calib, y_calib, groups=groups)
        pred_cv, interval_cv = mapie_cv.predict_interval(X_test)
        lo_cv, hi_cv = interval_cv[:, 0, 0], interval_cv[:, 1, 0]
        r = evaluate_interval(y_test, pred_cv, lo_cv, hi_cv)
        cvplus_coverages.append(r["coverage"])
        print(f"[stage1.6] CV+ (K=5, seed={seed}): coverage={r['coverage']:.3f} avg_width={r['avg_width']:.3f}")

    cv_min, cv_max = min(cvplus_coverages), max(cvplus_coverages)
    print(f"\n[stage1.6] SOH CV+ coverage range across {len(CV_SEEDS)} seeds: "
          f"[{cv_min:.3f}, {cv_max:.3f}] (spread={cv_max-cv_min:.3f})")
    print(f"[stage1.6] SOH Jackknife+ coverage (deterministic, K=n_batteries, no seed dependence): {jk_result['coverage']:.3f}")
    print(f"[stage1.6] SOH ORIGINAL method coverage (session-11-style 2-way split): {orig_result['coverage']:.3f} "
          f"(single number - session 11's ORIGINAL 64.7%-99.6% swing was itself measured by varying which "
          f"3-of-6 test batteries land in calib vs eval; reproduced conceptually by this same instability class)")

    return {
        "original_coverage": orig_result["coverage"], "original_width": orig_result["avg_width"],
        "jackknife_plus_coverage": jk_result["coverage"], "jackknife_plus_width": jk_result["avg_width"],
        "cvplus_coverage_min": cv_min, "cvplus_coverage_max": cv_max,
        "cvplus_coverage_spread": cv_max - cv_min, "cvplus_coverages_by_seed": cvplus_coverages,
    }


def rul_three_way_split():
    print("\n[stage1.6] === RUL: three-way split fix (calibration != test), PLAIN split-conformal (see docstring for scope) ===")
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    battery_data = load_all_battery_tensors()
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]   # joint model's own early-stopping val battery set
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[stage1.6] RUL fit batteries (gradient-trained on, unchanged): {len(fit_ids)}")
    print(f"[stage1.6] RUL calibration batteries (NEW - joint model's own val set, never gradient-updated): "
          f"{len(val_ids)} {val_ids}")
    print(f"[stage1.6] RUL test batteries (unchanged): {len(test_ids)} {test_ids}")

    X_train_full, rul_train_full, _, _, _, _ = (None,) * 6
    Xf, sf, rulf, _, _, _ = make_xy(battery_data, fit_ids)
    Xc, sc, rulc, _, bidc, _ = make_xy(battery_data, val_ids)
    Xt, st, rult, _, bidt, _ = make_xy(battery_data, test_ids)

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    Xf = apply_channel_norm(Xf, norm_stats)
    Xc = apply_channel_norm(Xc, norm_stats)
    Xt = apply_channel_norm(Xt, norm_stats)

    joint = JointSOHRULModel()
    joint.load_state_dict(torch.load(ROOT / "models" / "joint_adaptive.pt"))
    joint.eval()
    rul_mean, rul_std = float(rulf.mean()), float(rulf.std() + 1e-8)
    with torch.no_grad():
        _, rul_pred_calib_z = joint(torch.tensor(Xc))
        _, rul_pred_test_z = joint(torch.tensor(Xt))
    rul_pred_calib = rul_pred_calib_z.squeeze(-1).numpy() * rul_std + rul_mean
    rul_pred_test = rul_pred_test_z.squeeze(-1).numpy() * rul_std + rul_mean

    _, lo, hi, method = split_conformal(rul_pred_calib, rulc, rul_pred_test, ALPHA)
    result = evaluate_interval(rult, rul_pred_test, lo, hi)
    print(f"[stage1.6] RUL new 3-way-split coverage: method={method} coverage={result['coverage']:.3f} "
          f"avg_width={result['avg_width']:.3f} (n_test={result['n']})")

    # stability check: since only 5 candidate calibration batteries exist (val_ids)
    # without retraining, vary the SPECIFIC subset drawn from these 5 across seeds -
    # a real but small-sample stability check, caveat stated explicitly.
    rng_coverages = []
    bidc_arr = np.array(bidc)
    for seed in CV_SEEDS:
        rng = np.random.RandomState(seed)
        n_draw = max(1, len(val_ids) - 1)
        drawn = rng.choice(val_ids, size=n_draw, replace=False)
        mask = np.isin(bidc_arr, drawn)
        if mask.sum() < 5:
            continue
        _, lo_s, hi_s, _ = split_conformal(rul_pred_calib[mask], rulc[mask], rul_pred_test, ALPHA)
        r = evaluate_interval(rult, rul_pred_test, lo_s, hi_s)
        rng_coverages.append(r["coverage"])
        print(f"[stage1.6] RUL seed={seed} (calib batteries drawn: {sorted(drawn)}): coverage={r['coverage']:.3f}")

    rng_min, rng_max = (min(rng_coverages), max(rng_coverages)) if rng_coverages else (result["coverage"], result["coverage"])
    print(f"\n[stage1.6] RUL 3-way-split coverage range across {len(rng_coverages)} seeds "
          f"(SMALL-SAMPLE CAVEAT: only {len(val_ids)} candidate calibration batteries exist without "
          f"retraining the deep RUL model - read this range as indicative, not precise): "
          f"[{rng_min:.3f}, {rng_max:.3f}] (spread={rng_max-rng_min:.3f})")

    return {"new_split_coverage": result["coverage"], "new_split_width": result["avg_width"],
            "coverage_range_min": rng_min, "coverage_range_max": rng_max,
            "coverage_range_spread": rng_max - rng_min, "n_calib_candidates": len(val_ids)}


def main():
    soh_res = soh_jackknife_cvplus()
    rul_res = rul_three_way_split()

    print("\n[stage1.6] === FINAL COMPARISON vs. session 11's original 64.7%-99.6% coverage-swing finding ===")
    print(f"[stage1.6] SESSION 11 ORIGINAL (RUL, 2-way split, calib/eval both from the 6-battery test pool): "
          f"coverage swung 64.7% to 99.6% (spread=34.9pp) purely by which 3-of-6 test batteries land where.")
    print(f"[stage1.6] SOH — Jackknife+ (deterministic): {soh_res['jackknife_plus_coverage']:.1%} coverage, "
          f"NO seed-dependence at all (structurally eliminates this specific instability for SOH).")
    print(f"[stage1.6] SOH — CV+ (K=5, {len(CV_SEEDS)} seeds): coverage range "
          f"[{soh_res['cvplus_coverage_min']:.1%}, {soh_res['cvplus_coverage_max']:.1%}] "
          f"(spread={soh_res['cvplus_coverage_spread']*100:.1f}pp)")
    print(f"[stage1.6] RUL — new 3-way split (plain split-conformal, NOT genuine Jackknife+/CV+, small-sample "
          f"caveat applies): coverage range [{rul_res['coverage_range_min']:.1%}, {rul_res['coverage_range_max']:.1%}] "
          f"(spread={rul_res['coverage_range_spread']*100:.1f}pp) vs. session 11's 34.9pp spread")

    verdict_soh = ("NARROWS/CLOSES the instability for SOH" if soh_res['cvplus_coverage_spread'] < 0.10
                   else "does NOT meaningfully narrow the instability for SOH")
    verdict_rul = ("NARROWS the instability for RUL" if rul_res['coverage_range_spread'] < 0.349
                   else "does NOT narrow the instability for RUL relative to session 11's original spread")
    print(f"\n[stage1.6] VERDICT: {verdict_soh}; for RUL (scoped-down method), this {verdict_rul}.")

    pd.DataFrame([{**{"target": "SOH"}, **soh_res}]).to_csv(OUT_DIR / "stage1_6_soh_results.csv", index=False)
    pd.DataFrame([{**{"target": "RUL"}, **rul_res}]).to_csv(OUT_DIR / "stage1_6_rul_results.csv", index=False)
    print("\n[stage1.6] saved outputs/stage1_6_{soh,rul}_results.csv")
    print("[stage1.6] DONE")


if __name__ == "__main__":
    main()
