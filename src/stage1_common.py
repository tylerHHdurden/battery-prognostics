"""
Stage 1 shared utilities. Reused (not duplicated) across 1.1/1.2/1.5's
XGBoost-fusion variants so all three retrains are trained/evaluated by
IDENTICAL code paths and differ only in the one thing each item changes
(features, sample weights, or monotone_constraints respectively) - the
same discipline as run_calce_temp_confound_check.py (Stage 0, Check 0.2),
which this module generalizes.

CANONICAL FEATURE SET DECISION (per Stage 1's own instruction): uses the
Check 0.3 CORRECTED (NASA+MIT-only) BFA feature set from this point
forward - the only one of this project's three historical feature sets
without the CALCE-visibility leak. See DEVELOPMENT_LOG.md Stage 1 entry
for the full reasoning; this is now canonical for everything in Stage 1.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters import iterate_calce_cycles
from sequence_features import build_dataset_tensors, apply_channel_norm
from models.ica_encoder import ICAEncoder
from run_conformal import calib_eval_battery_split, split_conformal

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]
ICA_CHANNEL_SLICE = slice(3, 6)

with open(PROC_DIR / "bfa_selected_features_nasa_mit_only.txt") as _f:
    CANONICAL_FEATURES = [l.strip() for l in _f if l.strip()]

# Raw wall-clock/elapsed-time HIs in the canonical set - confirmed by
# direct reading of health_indicators.py's own docstrings: ICHV ("time
# spent with charge voltage above 90% of peak"), TEVD ("time elapsed
# ... until voltage first falls to 50%"), TEVI ("duration spent
# traversing the 80%->20% band") are all raw seconds. SCV/VDEDT/VIECT/
# MATD/MET are NOT raw durations (dQ/dV slope, dV/dt rate, a voltage
# value, a temperature, an energy respectively) - confirmed by the same
# docstrings, not assumed.
DURATION_FEATURES = ["ICHV", "TEVD", "TEVI"]
BASELINE_CYCLE = 10
ALPHA = 0.1  # 90% target coverage, same convention as every other conformal script


def add_reformulated_duration_features(hi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `{feat}_rel` = {feat}(cycle_n) / {feat}(battery's own cycle 10)
    for every feature in DURATION_FEATURES, reformulating a raw wall-
    clock duration into a protocol-invariant, per-battery-normalized
    ratio (session 27's root cause: NASA's slow-cycling protocol makes
    ICHV/TEVI durations orders of magnitude longer than MIT's fast-
    charging protocol in absolute terms - dividing by the SAME
    battery's own early-cycle value removes the protocol-scale
    constant, leaving only the within-battery relative change).

    Divide-by-zero/near-zero guard (explicit, not silent): if a
    battery's cycle-10 baseline for a feature is smaller in magnitude
    than 1e-6, that battery's baseline is replaced with the median of
    the feature over ALL of that battery's own cycles instead (still a
    per-battery, protocol-invariant reference, just a more robust one
    for the rare battery whose cycle 10 happens to be degenerate) - and
    this fallback is logged per battery/feature it fires for, not
    silently swapped in. Checked against the real data before writing
    this: none of the 35 batteries in this project's pool have a
    near-zero cycle-10 baseline for ICHV/TEVD/TEVI (min observed:
    ICHV=24.3, TEVD=13.2, TEVI=13.5) - so this guard does not fire on
    the actual dataset, but is real code, not a documented-but-untested
    assumption.
    """
    hi_df = hi_df.copy()
    baseline_rows = hi_df[hi_df["cycle_idx"] == BASELINE_CYCLE]
    baseline_map = {}  # (battery_id, feature) -> baseline value
    missing_baseline_batteries = set(hi_df["battery_id"].unique()) - set(baseline_rows["battery_id"].unique())
    if missing_baseline_batteries:
        print(f"[stage1-duration] WARNING: {len(missing_baseline_batteries)} batteries have no "
              f"cycle_idx=={BASELINE_CYCLE} row at all: {sorted(missing_baseline_batteries)} - "
              f"falling back to each such battery's own first available cycle as baseline.")

    for feat in DURATION_FEATURES:
        rel_col = f"{feat}_rel"
        rel_values = np.full(len(hi_df), np.nan, dtype=float)
        n_fallback = 0
        for bid, g in hi_df.groupby("battery_id"):
            base_row = baseline_rows[baseline_rows["battery_id"] == bid]
            if len(base_row) > 0:
                base_val = float(base_row[feat].iloc[0])
            else:
                base_val = float(g.sort_values("cycle_idx")[feat].iloc[0])
            if not np.isfinite(base_val) or abs(base_val) < 1e-6:
                fallback = float(g[feat].median())
                print(f"[stage1-duration] near-zero/non-finite cycle-{BASELINE_CYCLE} baseline "
                      f"for {bid}/{feat} ({base_val}) - falling back to this battery's own "
                      f"median {feat} ({fallback:.4f}) as the denominator.")
                base_val = fallback
                n_fallback += 1
            if not np.isfinite(base_val) or abs(base_val) < 1e-6:
                # even the median is degenerate - cannot form a ratio; leave NaN and log loudly.
                print(f"[stage1-duration] ERROR: {bid}/{feat} has a degenerate median baseline "
                      f"too ({base_val}) - {rel_col} will be NaN for this battery, not silently 0/inf.")
                continue
            idx = g.index
            rel_values[idx] = hi_df.loc[idx, feat].to_numpy(dtype=float) / base_val
        n_nonfinite = int((~np.isfinite(rel_values)).sum())
        hi_df[rel_col] = rel_values
        print(f"[stage1-duration] {feat} -> {rel_col}: {n_fallback} batteries used median-fallback "
              f"baseline, {n_nonfinite} non-finite output values (of {len(hi_df)} rows)")
    return hi_df


def canonical_feature_cols(reformulated: bool) -> list[str]:
    if not reformulated:
        return list(CANONICAL_FEATURES)
    return [f"{f}_rel" if f in DURATION_FEATURES else f for f in CANONICAL_FEATURES]


def load_nasa_mit_pool(reformulated: bool = False) -> pd.DataFrame:
    """NASA+MIT hi_table rows merged with fusion embeddings - the exact
    32-battery-pool training pool used throughout Stage 0/1's XGBoost-
    fusion variants. If reformulated=True, adds the *_rel duration
    columns to the FULL hi_table first (so CALCE gets its own reformulated
    columns too, from the SAME function, before any NASA+MIT/CALCE
    split - avoids two different reformulation code paths silently
    drifting apart)."""
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    if reformulated:
        hi_df = add_reformulated_duration_features(hi_df)
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    nasa_mit = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])]
    merged = pd.merge(nasa_mit, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    return merged, hi_df


def fusion_cols() -> list[str]:
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    return [c for c in fusion.columns if c.startswith("fusion_")]


def fit_xgb(merged: pd.DataFrame, train_mask: np.ndarray, feature_cols: list[str],
            sample_weight: np.ndarray = None, xgb_extra_kwargs: dict = None):
    """Median-impute (train-only medians) + fit XGBRegressor - identical
    hyperparameters to every other XGBoost-fusion script in this project
    (train_xgboost.py, run_calce_temp_confound_check.py) unless overridden
    via xgb_extra_kwargs (used by 1.3's pseudohuber objective and 1.5's
    monotone_constraints)."""
    cols = feature_cols + fusion_cols()
    X = merged[cols].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X[train_mask], axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = merged["SOH"].to_numpy(dtype=float)

    kwargs = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  n_jobs=-1, reg_lambda=1.0)
    if xgb_extra_kwargs:
        kwargs.update(xgb_extra_kwargs)
    model = XGBRegressor(**kwargs)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight[train_mask]
    model.fit(X[train_mask], y[train_mask], **fit_kwargs)
    return model, col_medians, cols


def eval_indomain(model, medians, cols, merged: pd.DataFrame, test_mask: np.ndarray) -> dict:
    X = merged[cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    y = merged["SOH"].to_numpy(dtype=float)
    pred = model.predict(X[test_mask])
    y_true = y[test_mask]
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "pred": pred, "y_true": y_true,
        "battery_id": merged.loc[test_mask, "battery_id"].to_numpy(),
    }


_CALCE_TENSOR_CACHE = {}


def build_calce_tensors():
    if "X" in _CALCE_TENSOR_CACHE:
        return _CALCE_TENSOR_CACHE["X"], _CALCE_TENSOR_CACHE["soh"], \
            _CALCE_TENSOR_CACHE["bid"], _CALCE_TENSOR_CACHE["cyc"]
    all_X, all_soh, all_bid, all_cyc = [], [], [], []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        all_X.append(X.astype(np.float32))
        all_soh.append(soh.astype(np.float32))
        all_bid += [cid] * len(soh)
        all_cyc += list(idxs)
    X_all = np.concatenate(all_X)
    soh_all = np.concatenate(all_soh)
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_all = apply_channel_norm(X_all, norm_stats)
    _CALCE_TENSOR_CACHE.update(X=X_all, soh=soh_all, bid=all_bid, cyc=all_cyc)
    return X_all, soh_all, all_bid, all_cyc


_CALCE_FUSION_CACHE = {}


def build_calce_merged(hi_df_full: pd.DataFrame) -> pd.DataFrame:
    """CALCE rows merged with (unchanged) fusion embeddings from the
    original ica_encoder.pt - reused identically from
    run_calce_temp_confound_check.py's build_calce_tensors, cached across
    calls within one process since it's the same expensive step every
    Stage 1 variant needs."""
    if "df" in _CALCE_FUSION_CACHE:
        return _CALCE_FUSION_CACHE["df"]
    X_calce, soh_calce, bid_calce, cyc_calce = build_calce_tensors()
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()
    with torch.no_grad():
        fusion_emb = encoder.encode(torch.tensor(X_calce[:, :, ICA_CHANNEL_SLICE])).numpy()

    calce_hi = hi_df_full[hi_df_full["dataset"] == "CALCE"]
    seq_df = pd.DataFrame({"battery_id": bid_calce, "cycle_idx": cyc_calce})
    for i in range(16):
        seq_df[f"fusion_{i}"] = fusion_emb[:, i]
    calce_merged = pd.merge(calce_hi, seq_df, on=["battery_id", "cycle_idx"], how="inner")
    _CALCE_FUSION_CACHE["df"] = calce_merged
    return calce_merged


def eval_calce(model, medians, feature_cols, calce_merged: pd.DataFrame) -> dict:
    cols = feature_cols + fusion_cols()
    X = calce_merged[cols].to_numpy(dtype=float, copy=True)
    for j, col in enumerate(feature_cols):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = medians[j]
    y_true = calce_merged["SOH"].to_numpy()
    pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "pred": pred, "y_true": y_true,
        "battery_id": calce_merged["battery_id"].to_numpy(),
    }


def calce_coverage(indomain_result: dict, calce_result: dict, alpha: float = ALPHA) -> dict:
    """Standard (fixed-width) split-conformal coverage on CALCE, reusing
    run_conformal.py's own split_conformal utility unchanged: calibrate
    on ONE HALF of the in-domain (NASA+MIT) test battery set (the
    project's existing calib_eval_battery_split convention), evaluate
    coverage on CALCE. This is the same class of baseline number as the
    project's other headline CALCE coverage figures (e.g. the 7.4%
    expanded-pool baseline from session 35 Part 4), computed fresh here
    for THIS exact model/feature-set/pool combination so 1.1/1.2/1.5 can
    be compared to a self-consistent Stage-1 reference point rather than
    to a different pool's number."""
    test_bids = indomain_result["battery_id"]
    unique_test_ids = sorted(set(test_bids.tolist()))
    calib_ids, eval_ids = calib_eval_battery_split(unique_test_ids)
    calib_mask = np.isin(test_bids, calib_ids)
    calib_pred = indomain_result["pred"][calib_mask]
    calib_y = indomain_result["y_true"][calib_mask]

    _, lo, hi, method = split_conformal(calib_pred, calib_y, calce_result["pred"], alpha)
    covered = (calce_result["y_true"] >= lo) & (calce_result["y_true"] <= hi)
    coverage = float(covered.mean())
    avg_width = float(np.mean(hi - lo))
    return {"method": method, "target_coverage": 1 - alpha, "empirical_coverage": coverage,
            "avg_interval_width": avg_width, "n_calib": int(calib_mask.sum()), "n_calce": len(calce_result["pred"])}


def battery_split_masks(merged: pd.DataFrame):
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_mask = merged["battery_id"].isin(split["train_ids"]).to_numpy()
    test_mask = ~train_mask
    return train_mask, test_mask, split
