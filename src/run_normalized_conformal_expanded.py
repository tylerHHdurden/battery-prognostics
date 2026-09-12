"""
Targeted improvement pass, Part 4 (the main event): does input-adaptive
conformal prediction fix the CALCE coverage collapse that fixed-width
split-conformal cannot?

BACKGROUND. run_calce_zero_retrain_eval_expanded.py's split-conformal
interval has ONE global width, set entirely by the in-domain calibration
residual quantile - identical half-width in-domain (1.154) and on CALCE
(1.154), which is mechanically why CALCE coverage collapses (7.4% vs.
a 90% target): the interval simply never adapts to how much harder a
given input is to predict. Neither of this project's two prior
CALCE-focused attempts fixed this specific mechanism - ACI (session 29)
adapts a threshold OVER TIME in a data stream, not by input difficulty;
weighted conformal (session ~31, domain_shift_conformal) reweights by a
domain-classifier density ratio and broke at small sample size (3-vs-3
CALCE cells). This script implements the two standard, well-established
methods that DO make width vary per-input:

  1. NORMALIZED (locally-weighted) conformal prediction (Papadopoulos
     et al. 2002): a secondary model sigma(x) predicts expected
     residual MAGNITUDE from the same 8 BFA-selected features used
     everywhere else in this pipeline. Nonconformity score becomes
     |y - f(x)| / sigma(x); the calibration quantile q of this
     NORMALIZED score is found once, then every point's interval is
     f(x) +/- q * sigma(x) - genuinely input-adaptive width.
  2. Conformalized Quantile Regression / CQR (Romano, Patterson,
     Candes 2019): two quantile-regression models predict the
     (alpha/2, 1-alpha/2) conditional quantiles of y directly (same 8
     BFA features). The calibration quantile of the CQR nonconformity
     score (how far outside [q_lo(x), q_hi(x)] the true y falls)
     widens/narrows those raw quantile predictions into a valid
     interval - a second, INDEPENDENT way to get input-adaptive width.

Both sigma(x) and the CQR quantile models are fit ONLY on TRAIN rows
(genuinely disjoint from the calibration/eval battery split AND from
CALCE) - the same exchangeability discipline session 4's original
conformal fix already established for the point predictor itself,
applied here to the new adaptivity models too, to avoid a new form of
the same leakage bug.

Reused UNCHANGED: the existing Stacking-Ridge-fusion-expanded point
predictions (this script does NOT retrain the ensemble - normalized CP
and CQR are both built ON TOP of an existing point predictor, that is
the whole point of the method), the same calib_eval_battery_split and
finite-sample quantile correction as run_conformal.py, the same CALCE
zero-retrain methodology (same calibration, applied out-of-domain) as
run_calce_zero_retrain_eval_expanded.py.

HONESTY COMMITMENT (per instruction): this script reports whichever of
three outcomes actually happens - (a) width varies AND CALCE coverage
measurably improves, (b) width varies but CALCE coverage does not
improve (sigma(x)/quantile models don't generalize to CALCE's inputs
any better than the point predictor did), or (c) the documented
literature failure mode - intervals become "adaptive" in form but
inflated/uninformative, or coverage gets WORSE. Whichever it is gets
reported as the real number, not smoothed into a success narrative.
CALCE is only 3 cells (2,941 cycles) - any result here is flagged
against the small-sample-noise risk this project already hit once
(session 19's domain classifier), not presented as definitive on its
own.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ensemble_fusion_expanded import load_merged_fusion_expanded
from run_conformal import calib_eval_battery_split

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

ALPHA = 0.1  # 90% target coverage, same as every prior conformal result in this project
SIGMA_FLOOR = 0.15  # SOH percentage points - avoids division blow-up on a near-zero residual prediction


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Identical correction to run_conformal.py's manual fallback -
    the standard split-conformal finite-sample quantile, applied here
    to whichever nonconformity score (raw or normalized) is passed in."""
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def load_bfa_features(bfa_selected: list[str]) -> pd.DataFrame:
    hi = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    keep = ["dataset", "battery_id", "cycle_idx"] + bfa_selected
    return hi[keep].copy()


def impute_with_train_medians(X: np.ndarray, train_medians: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[np.isinf(X)] = np.nan
    inds = np.where(np.isnan(X))
    X[inds] = np.take(train_medians, inds[1])
    return X


def main():
    with open(PROC_DIR / "bfa_selected_features_expanded.txt") as f:
        bfa_selected = [l.strip() for l in f if l.strip()]
    print(f"[norm-conformal] BFA-selected features used for sigma(x)/CQR: {bfa_selected}")

    bfa_df = load_bfa_features(bfa_selected)

    print("\n[norm-conformal] === Building TRAIN-set ensemble predictions "
          "(for fitting sigma(x)/CQR, disjoint from calib/eval/CALCE) ===")
    train_df, fusion_cols = load_merged_fusion_expanded("train")
    base_cols = ["pred_XGBoost_fusion", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
    meta_cols = base_cols + fusion_cols
    import pickle
    with open(ROOT / "models" / "ridge_meta_fusion_expanded.pkl", "rb") as f:
        ridge = pickle.load(f)
    train_df["pred_ensemble"] = ridge.predict(train_df[meta_cols].to_numpy())
    train_df = pd.merge(train_df, bfa_df, on=["dataset", "battery_id", "cycle_idx"], how="left")

    X_train_raw = train_df[bfa_selected].to_numpy(dtype=float, copy=True)
    X_train_raw[np.isinf(X_train_raw)] = np.nan
    train_medians = np.nanmedian(X_train_raw, axis=0)
    X_train = impute_with_train_medians(X_train_raw, train_medians)
    y_train = train_df["SOH"].to_numpy()
    resid_train = np.abs(y_train - train_df["pred_ensemble"].to_numpy())
    print(f"[norm-conformal] train rows={len(train_df)}, mean|residual|={resid_train.mean():.4f}, "
          f"NaN BFA values imputed with train medians (same convention as every other "
          f"script in this pipeline)")

    print("\n[norm-conformal] === Fitting sigma(x): GBR on log1p(|residual|) ===")
    sigma_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
    )
    sigma_model.fit(X_train, np.log1p(resid_train))

    def predict_sigma(X):
        raw = np.expm1(sigma_model.predict(X))
        return np.clip(raw, SIGMA_FLOOR, None)

    sigma_train_pred = predict_sigma(X_train)
    print(f"[norm-conformal] sigma(x) on TRAIN: predicted range [{sigma_train_pred.min():.3f}, "
          f"{sigma_train_pred.max():.3f}], mean={sigma_train_pred.mean():.3f} "
          f"(actual mean|residual|={resid_train.mean():.3f}) - sigma(x) should track this "
          f"roughly, not exactly (it's a smoothed model of a noisy target)")

    print("\n[norm-conformal] === Fitting CQR quantile models "
          f"(alpha/2={ALPHA/2:.3f}, 1-alpha/2={1-ALPHA/2:.3f}) ===")
    q_lo_model = GradientBoostingRegressor(
        loss="quantile", alpha=ALPHA / 2, n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8, random_state=42,
    )
    q_hi_model = GradientBoostingRegressor(
        loss="quantile", alpha=1 - ALPHA / 2, n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8, random_state=42,
    )
    q_lo_model.fit(X_train, y_train)
    q_hi_model.fit(X_train, y_train)
    q_lo_train = q_lo_model.predict(X_train)
    q_hi_train = q_hi_model.predict(X_train)
    crossed = (q_lo_train > q_hi_train).sum()
    print(f"[norm-conformal] CQR raw quantile predictions on TRAIN: "
          f"mean width={np.mean(q_hi_train - q_lo_train):.3f}, "
          f"{crossed}/{len(q_lo_train)} rows have crossed quantiles (lo>hi, a known GBR "
          f"quantile-regression failure mode - CQR's calibration step corrects the final "
          f"coverage regardless, but this is worth flagging as-is, not silently clipped)")

    # ---------------- in-domain (NASA+MIT) calib/eval ----------------
    print("\n[norm-conformal] === Loading NASA/MIT test-set data (same calib/eval "
          "battery split as the existing CALCE zero-retrain script) ===")
    test_df = pd.read_csv(PRED_DIR / "ensemble_fusion_expanded_test_preds.csv")
    test_df = pd.merge(test_df, bfa_df, on=["dataset", "battery_id", "cycle_idx"], how="left")
    calib_ids, eval_ids = calib_eval_battery_split(test_df["battery_id"].unique().tolist())
    calib_df = test_df[test_df["battery_id"].isin(calib_ids)].reset_index(drop=True)
    eval_df = test_df[test_df["battery_id"].isin(eval_ids)].reset_index(drop=True)
    print(f"[norm-conformal] calib batteries={len(calib_ids)} ({len(calib_df)} rows), "
          f"eval batteries={len(eval_ids)} ({len(eval_df)} rows)")

    def features_of(df):
        X = df[bfa_selected].to_numpy(dtype=float, copy=True)
        return impute_with_train_medians(X, train_medians)

    X_calib, X_eval = features_of(calib_df), features_of(eval_df)
    pred_calib = calib_df["pred_Stacking_Ridge_fusion_expanded"].to_numpy()
    pred_eval = eval_df["pred_Stacking_Ridge_fusion_expanded"].to_numpy()
    y_calib = calib_df["SOH"].to_numpy()
    y_eval = eval_df["SOH"].to_numpy()

    # ---------------- CALCE (out-of-domain, zero-retrain) ----------------
    print("\n[norm-conformal] === Loading CALCE zero-retrain data "
          "(reusing calce_zero_retrain_expanded_preds.csv, unchanged point predictions) ===")
    calce_df = pd.read_csv(PRED_DIR / "calce_zero_retrain_expanded_preds.csv")
    calce_df = pd.merge(calce_df, bfa_df, on=["dataset", "battery_id", "cycle_idx"], how="left")
    X_calce = features_of(calce_df)
    pred_calce = calce_df["pred_Stacking_Ridge_fusion_expanded"].to_numpy()
    y_calce = calce_df["SOH"].to_numpy()
    print(f"[norm-conformal] CALCE rows={len(calce_df)} (3 cells) - small-sample-noise "
          f"caveat applies to everything reported below for this domain.")

    def report_baseline():
        """Fixed-width split-conformal, recomputed here identically to
        run_calce_zero_retrain_eval_expanded.py, purely as the in-script
        reference point every method below is compared against."""
        resid_calib = np.abs(y_calib - pred_calib)
        q = finite_sample_quantile(resid_calib, ALPHA)
        for name, pred, y in [("in-domain (eval)", pred_eval, y_eval), ("CALCE", pred_calce, y_calce)]:
            lo, hi = pred - q, pred + q
            cov = float(((y >= lo) & (y <= hi)).mean())
            width = float(np.mean(hi - lo))
            print(f"[norm-conformal] BASELINE fixed-width split-conformal, {name}: "
                  f"coverage={cov:.3f} width={width:.3f} (constant, by construction)")
        return q

    print("\n[norm-conformal] === Method 0: baseline fixed-width split-conformal (reference) ===")
    report_baseline()

    results = []

    def eval_method(method_name, lo_calib, hi_calib, lo_eval_fn, hi_eval_fn):
        """lo_eval_fn/hi_eval_fn: callables taking (X, pred) -> lo/hi arrays,
        already calibrated (q or Q baked in via closure)."""
        for domain, X, pred, y in [
            ("in-domain (eval)", X_eval, pred_eval, y_eval),
            ("CALCE (out-of-domain, zero-retrain)", X_calce, pred_calce, y_calce),
        ]:
            lo, hi = lo_eval_fn(X, pred), hi_eval_fn(X, pred)
            cov = float(((y >= lo) & (y <= hi)).mean())
            width = hi - lo
            print(f"[norm-conformal] {method_name}, {domain}: coverage={cov:.4f} "
                  f"mean_width={width.mean():.3f} width_std={width.std():.3f} "
                  f"width_range=[{width.min():.3f},{width.max():.3f}] n={len(y)}")
            results.append({
                "method": method_name, "domain": domain, "coverage": cov,
                "target_coverage": 1 - ALPHA, "mean_width": float(width.mean()),
                "width_std": float(width.std()), "width_min": float(width.min()),
                "width_max": float(width.max()), "n": len(y),
            })

    print("\n[norm-conformal] === Method 1: Normalized (locally-weighted) conformal prediction ===")
    sigma_calib = predict_sigma(X_calib)
    norm_scores_calib = np.abs(y_calib - pred_calib) / sigma_calib
    q_norm = finite_sample_quantile(norm_scores_calib, ALPHA)
    print(f"[norm-conformal] normalized calibration quantile q={q_norm:.4f} "
          f"(interval half-width for a point x is q * sigma(x) - genuinely varies per point, "
          f"unlike the baseline's constant q)")

    def norm_lo(X, pred):
        return pred - q_norm * predict_sigma(X)

    def norm_hi(X, pred):
        return pred + q_norm * predict_sigma(X)

    eval_method("Normalized CP", None, None, norm_lo, norm_hi)

    print("\n[norm-conformal] === Method 2: Conformalized Quantile Regression (CQR) ===")
    q_lo_calib = q_lo_model.predict(X_calib)
    q_hi_calib = q_hi_model.predict(X_calib)
    cqr_scores_calib = np.maximum(q_lo_calib - y_calib, y_calib - q_hi_calib)
    Q_cqr = finite_sample_quantile(cqr_scores_calib, ALPHA)
    print(f"[norm-conformal] CQR calibration correction Q={Q_cqr:.4f} "
          f"(final interval = [q_lo(x)-Q, q_hi(x)+Q])")

    def cqr_lo(X, pred):
        return q_lo_model.predict(X) - Q_cqr

    def cqr_hi(X, pred):
        return q_hi_model.predict(X) + Q_cqr

    eval_method("CQR", None, None, cqr_lo, cqr_hi)

    # ---------------- honest verdict ----------------
    print("\n[norm-conformal] === VERDICT (reporting the real outcome, not assuming success) ===")
    base_q = report_baseline()
    df_res = pd.DataFrame(results)
    for method in ["Normalized CP", "CQR"]:
        calce_row = df_res[(df_res.method == method) & (df_res.domain.str.startswith("CALCE"))].iloc[0]
        indomain_row = df_res[(df_res.method == method) & (df_res.domain == "in-domain (eval)")].iloc[0]
        width_varies = calce_row["width_std"] > 1e-6 or indomain_row["width_std"] > 1e-6
        calce_improved = calce_row["coverage"] > 0.074 + 0.03  # meaningfully above the fixed-width baseline's 7.4%
        calce_worse = calce_row["coverage"] < 0.074 - 0.01
        if width_varies and calce_improved:
            outcome = "(a) width varies AND CALCE coverage measurably improved"
        elif width_varies and calce_worse:
            outcome = "(c) documented failure mode: adaptive in form, coverage got WORSE"
        elif width_varies and not calce_improved:
            outcome = "(b) width varies but CALCE coverage did NOT meaningfully improve"
        else:
            outcome = "width did not vary in a meaningful way - inconclusive"
        print(f"[norm-conformal] {method}: CALCE coverage {calce_row['coverage']:.1%} "
              f"(baseline was 7.4%) | width std={calce_row['width_std']:.3f} (baseline=0, "
              f"constant) -> OUTCOME: {outcome}")

    pd.DataFrame(results).to_csv(OUT_DIR / "normalized_conformal_expanded_results.csv", index=False)
    print(f"\n[norm-conformal] Saved outputs/normalized_conformal_expanded_results.csv "
          f"({len(results)} rows)")
    print("[norm-conformal] DONE")


if __name__ == "__main__":
    main()
