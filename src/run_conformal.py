"""
Phase 6: split-conformal prediction via MAPIE, wrapped around the
ensemble output, for both SOH and RUL. Reports empirical coverage on the
NASA/MIT test set.

Point estimates conformalized:
  - SOH: the Phase 3 Stacking-Ridge meta-learner prediction.
  - RUL: Phase 2/3 never trained an RUL predictor (all 4 base learners +
    both meta-learners are SOH-only regressors) - RUL point estimates
    only exist from Phase 4's joint model. This script uses the Phase 4
    "adaptive" variant's RUL head (the one variant that isn't collapsed
    on RUL, per the Phase 4 ablation) as the RUL point estimate to
    conformalize. This is logged explicitly because it means SOH and RUL
    conformal intervals in this script come from two DIFFERENT underlying
    models (Stacking-Ridge vs. Phase 4 joint-adaptive), not the same
    ensemble wearing two hats.

Calibration split — IMPORTANT, this was wrong in the first draft and
fixed after actually checking the numbers: split-conformal's coverage
guarantee requires the calibration residuals to be exchangeable with the
test residuals. The first version of this script calibrated on the
TRAIN split's own residuals - but the Ridge/XGBoost meta-learners (and
XGBoost itself) were FIT directly on those exact rows, so their "train"
residuals are in-sample and artificially tiny (median abs residual
0.115 SOH%) compared to genuine held-out residuals (test median 0.777,
~6.7x larger). Calibrating on in-sample residuals produced an interval
so narrow it only covered 27% of test points against a 90% target - not
a MAPIE bug (a manual from-scratch implementation gave the identical
27% coverage), a genuine violation of conformal prediction's
exchangeability assumption.

Fix: the 6 held-out TEST batteries are split in half - one half used
purely for conformal CALIBRATION, the other half for final coverage
EVALUATION. Both halves are genuinely unseen by every model in the
pipeline (never used for fitting any base learner or meta-learner), so
they're exchangeable with each other, which is what the method actually
needs. Cost: the final reported coverage is measured on only ~half the
original test set (3 batteries instead of 6) - a real reduction in
statistical power, logged rather than hidden. One further limitation:
with only 6 test batteries total (1 NASA + 5 MIT) and 1 NASA battery
in the whole set, both halves can't simultaneously contain a NASA
battery; NASA's B0018 ends up in the calibration half only, so the
final coverage number is only directly validated on MIT cells - noted
here explicitly rather than glossed over.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.1  # 90% target coverage


class PrefitLookup:
    """A .predict()/.fit()-compatible wrapper around a precomputed prediction
    array, indexed by row position - lets MAPIE treat an already-fixed
    point estimate as a 'prefit' sklearn estimator (its intended use case
    for split-conformal calibration)."""

    def __init__(self, lookup: np.ndarray):
        self.lookup = lookup
        self.__sklearn_is_fitted__ = lambda: True

    def predict(self, X):
        idx = X[:, 0].astype(int)
        return self.lookup[idx]

    def fit(self, X, y):
        return self


def split_conformal(calib_pred, calib_y, test_pred, alpha):
    """
    Tries MAPIE's modern then legacy API; falls back to a manual
    split-conformal absolute-residual-quantile implementation (the exact
    method MAPIE itself implements for a prefit 1D regressor) if neither
    API matches this environment's installed MAPIE version.
    """
    n_calib = len(calib_pred)
    combined = np.concatenate([calib_pred, test_pred])
    estimator = PrefitLookup(combined)
    X_calib = np.arange(n_calib).reshape(-1, 1)
    X_test = (np.arange(len(test_pred)) + n_calib).reshape(-1, 1)

    try:
        from mapie.regression import SplitConformalRegressor
        mapie_est = SplitConformalRegressor(estimator=estimator, confidence_level=1 - alpha, prefit=True)
        mapie_est.conformalize(X_calib, calib_y)
        y_pred, y_interval = mapie_est.predict_interval(X_test)
        return y_pred, y_interval[:, 0, 0], y_interval[:, 1, 0], "MAPIE.SplitConformalRegressor"
    except Exception as e1:
        try:
            from mapie.regression import MapieRegressor
            mapie_est = MapieRegressor(estimator=estimator, cv="prefit")
            mapie_est.fit(X_calib, calib_y)
            y_pred, y_interval = mapie_est.predict(X_test, alpha=alpha)
            return y_pred, y_interval[:, 0, 0], y_interval[:, 1, 0], "MAPIE.MapieRegressor(prefit)"
        except Exception as e2:
            print(f"[conformal] Both MAPIE APIs failed "
                  f"({type(e1).__name__}: {e1} | {type(e2).__name__}: {e2}); "
                  f"falling back to manual split-conformal.")
            resid = np.abs(calib_y - calib_pred)
            n = len(resid)
            q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            q = np.quantile(resid, q_level, method="higher")
            return test_pred, test_pred - q, test_pred + q, "manual_split_conformal"


def evaluate_coverage(target_name, calib_pred, calib_y, test_pred, test_y, alpha=ALPHA):
    _, lo, hi, method = split_conformal(calib_pred, calib_y, test_pred, alpha)
    covered = (test_y >= lo) & (test_y <= hi)
    coverage = float(covered.mean())
    avg_width = float(np.mean(hi - lo))
    print(f"[conformal] {target_name}: method={method} target_coverage={1-alpha:.0%} "
          f"empirical_coverage={coverage:.3f} avg_interval_width={avg_width:.3f} n_test={len(test_y)}")
    return {
        "target": target_name, "method": method, "target_coverage": 1 - alpha,
        "empirical_coverage": coverage, "avg_interval_width": avg_width, "n_test": len(test_y),
    }, lo, hi


def calib_eval_battery_split(test_ids: list[str]):
    """Even/odd split (by sorted position) of the held-out test batteries
    into a calibration half and an evaluation half - both genuinely
    unseen by every fitted model, so exchangeable with each other."""
    ids_sorted = sorted(test_ids)
    calib_ids = ids_sorted[0::2]
    eval_ids = ids_sorted[1::2]
    return calib_ids, eval_ids


def main():
    from train_ensemble import load_merged
    test_df = load_merged("test")
    base_cols = ["pred_XGBoost", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]

    with open(ROOT / "models" / "ridge_meta.pkl", "rb") as f:
        ridge = pickle.load(f)
    test_df["pred_Stacking_Ridge"] = ridge.predict(test_df[base_cols].to_numpy())

    calib_ids, eval_ids = calib_eval_battery_split(test_df["battery_id"].unique().tolist())
    print(f"[conformal] calibration batteries ({len(calib_ids)}): {calib_ids}")
    print(f"[conformal] evaluation batteries ({len(eval_ids)}): {eval_ids}")

    calib_mask = test_df["battery_id"].isin(calib_ids)
    eval_mask = test_df["battery_id"].isin(eval_ids)
    calib_df, eval_df = test_df[calib_mask], test_df[eval_mask]

    results = []

    # --- SOH conformal (Stacking-Ridge) ---
    r, lo, hi = evaluate_coverage(
        "SOH (Stacking-Ridge)",
        calib_df["pred_Stacking_Ridge"].to_numpy(), calib_df["SOH"].to_numpy(),
        eval_df["pred_Stacking_Ridge"].to_numpy(), eval_df["SOH"].to_numpy(),
    )
    results.append(r)
    eval_out = eval_df.copy()
    eval_out["SOH_pred"] = eval_out["pred_Stacking_Ridge"]
    eval_out["SOH_lo90"], eval_out["SOH_hi90"] = lo, hi
    eval_out.to_csv(PRED_DIR / "conformal_soh_eval_preds.csv", index=False)

    # --- RUL conformal (Phase 4 joint-adaptive RUL head) ---
    from train_deep_models import load_all_battery_tensors, make_xy
    from sequence_features import apply_channel_norm
    from models.joint_model import JointSOHRULModel

    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]

    X_train_full, soh_train_full, rul_train_full, _, _, _ = make_xy(battery_data, train_ids)
    X_calib, soh_calib, rul_calib, ds_c, bid_c, cyc_c = make_xy(battery_data, calib_ids)
    X_eval, soh_eval, rul_eval, ds_e, bid_e, cyc_e = make_xy(battery_data, eval_ids)

    # joint_adaptive.pt was trained on channel-normalized input (see
    # sequence_features.compute_channel_norm_stats) - apply the same
    # saved transform here, or every prediction below is garbage.
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_train_full = apply_channel_norm(X_train_full, norm_stats)
    X_calib = apply_channel_norm(X_calib, norm_stats)
    X_eval = apply_channel_norm(X_eval, norm_stats)

    joint = JointSOHRULModel()
    joint.load_state_dict(torch.load(ROOT / "models" / "joint_adaptive.pt"))
    joint.eval()

    # the joint model was trained on standardized RUL; recover its
    # training-set mean/std the same way train_joint_adaptive.py did, so
    # predictions are de-standardized consistently. (train_ids here, NOT
    # calib/eval, since that's the actual normalization the model learned.)
    rul_mean, rul_std = float(rul_train_full.mean()), float(rul_train_full.std() + 1e-8)
    with torch.no_grad():
        _, rul_pred_calib_z = joint(torch.tensor(X_calib))
        _, rul_pred_eval_z = joint(torch.tensor(X_eval))
    rul_pred_calib = rul_pred_calib_z.squeeze(-1).numpy() * rul_std + rul_mean
    rul_pred_eval = rul_pred_eval_z.squeeze(-1).numpy() * rul_std + rul_mean

    r, lo, hi = evaluate_coverage(
        "RUL (joint-adaptive)", rul_pred_calib, rul_calib, rul_pred_eval, rul_eval,
    )
    results.append(r)

    rul_out = pd.DataFrame({
        "dataset": ds_e, "battery_id": bid_e, "cycle_idx": cyc_e,
        "RUL": rul_eval, "RUL_pred": rul_pred_eval, "RUL_lo90": lo, "RUL_hi90": hi,
    })
    rul_out.to_csv(PRED_DIR / "conformal_rul_eval_preds.csv", index=False)

    pd.DataFrame(results).to_csv(OUT_DIR / "conformal_coverage.csv", index=False)
    print("[conformal] DONE (see outputs/conformal_coverage.csv)")


if __name__ == "__main__":
    main()
