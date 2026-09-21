"""
Research pass 2, item 5: statistical consolidation pass. Builds one
consolidated table with proper PAIRED significance tests (not just
point-estimate R2/RMSE comparisons) for three head-to-head claims this
project has already made, reusing already-saved data/models wherever
possible - NO new model architecture or methodology is trained.

Two DISCLOSED exceptions to "no new training," both pure bookkeeping,
not new methodology (stated plainly, not hidden):
  (A) reformulated-vs-original: the Stage 5 extended-reformulation
      XGBoost model (models/_experimental_xgb_soh_fusion_extended_
      reformulation.json) was only ever scored for AGGREGATE R2/RMSE
      (outputs/stage5_extended_reformulation_eval.csv) - no per-row
      predictions were ever saved. Both this model and the deployed
      model are re-SCORED (pure inference on an already-trained,
      already-saved model, the exact same "no retraining" convention
      already established by run_save_percycle_predictions.py) on the
      identical rows so their errors are genuinely paired.
  (B) Severson/Attia baselines: outputs/stage6_1_severson_attia_
      results.csv is also aggregate-only - the ElasticNetCV model
      objects themselves were never saved. The already-cached Severson
      feature parquets (severson_features_*.parquet, built once,
      reused here unchanged) are used to refit the SAME tiny 1-4-
      feature ElasticNetCV baseline (identical code/hyperparameters to
      run_stage6_1_severson_attia_baselines.py) purely to recover
      per-row predictions for pairing - not a new baseline or method.

TWO paired tests reported for every comparison, matching this project's
own established bootstrap convention (run_bootstrap_significance_
expanded.py's battery_bootstrap_row_indices: battery-level resampling
WITH replacement, N=2000, seed=42, 95% CI via percentiles):
  - paired battery-level bootstrap on the R2 delta (works at any
    battery count)
  - paired Wilcoxon signed-rank test on PER-BATTERY mean |error| deltas
    (the natural independent-sample unit - raw per-CYCLE rows within a
    battery are not independent draws, so Wilcoxon is run on one
    number per battery, not one number per cycle; disclosed as having
    limited power when a dataset has few batteries, not hidden)
Effect sizes (mean R2/RMSE delta, and Cliff's delta for the Wilcoxon
comparison) are reported alongside p-values, not p-values alone.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols,
    build_calce_merged, OUT_DIR, PROC_DIR, ROOT,
)
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols

SEED = 42
N_BOOTSTRAP = 2000
CI = (2.5, 97.5)
VARIANCE_FEATS = ["log_var_dq"]
RICH_FEATS = ["log_var_dq", "min_dq", "skew_dq", "early_fade_slope"]


def battery_bootstrap_delta_ci(battery_ids, err_a, err_b, metric_fn, y_true, n_boot=N_BOOTSTRAP, seed=SEED):
    """metric_fn(y_true, pred) -> scalar; err_a/err_b are actually PRED
    arrays here (named for clarity at call sites: a=baseline, b=candidate).
    Resamples battery IDs with replacement (project's own established
    convention), computes metric(candidate) - metric(baseline) per draw."""
    rng = np.random.default_rng(seed)
    battery_ids = np.asarray(battery_ids)
    unique_b = np.array(sorted(set(battery_ids)))
    row_idx = {b: np.where(battery_ids == b)[0] for b in unique_b}
    deltas = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_b, size=len(unique_b), replace=True)
        idx = np.concatenate([row_idx[b] for b in chosen])
        m_a = metric_fn(y_true[idx], err_a[idx])
        m_b = metric_fn(y_true[idx], err_b[idx])
        deltas.append(m_b - m_a)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, CI)
    mean = float(np.mean(deltas))
    significant = not (lo <= 0 <= hi)
    return {"mean_delta_r2": mean, "ci_lo": float(lo), "ci_hi": float(hi), "significant": bool(significant)}


def cliffs_delta(x, y):
    """Simple O(n*m) Cliff's delta - fine at per-battery sample sizes."""
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > yj) for xi in x for yj in y)
    lt = sum((xi < yj) for xi in x for yj in y)
    return (gt - lt) / (len(x) * len(y))


def paired_wilcoxon_per_battery(battery_ids, y_true, pred_a, pred_b, label_a, label_b):
    """Per-battery mean |error| for each method, then a paired Wilcoxon
    signed-rank test across batteries (the natural independent unit)."""
    df = pd.DataFrame({"battery_id": battery_ids, "y_true": y_true, "pred_a": pred_a, "pred_b": pred_b})
    df["abs_err_a"] = (df["y_true"] - df["pred_a"]).abs()
    df["abs_err_b"] = (df["y_true"] - df["pred_b"]).abs()
    per_batt = df.groupby("battery_id")[["abs_err_a", "abs_err_b"]].mean()
    n_batt = len(per_batt)
    if n_batt < 3:
        return {"n_batteries": n_batt, "wilcoxon_p": None, "cliffs_delta": None,
                "note": "too few batteries for a meaningful Wilcoxon test"}
    try:
        stat, p = wilcoxon(per_batt["abs_err_a"], per_batt["abs_err_b"])
    except ValueError as e:
        return {"n_batteries": n_batt, "wilcoxon_p": None, "cliffs_delta": None, "note": f"wilcoxon failed: {e}"}
    delta = cliffs_delta(per_batt["abs_err_a"].to_numpy(), per_batt["abs_err_b"].to_numpy())
    return {"n_batteries": n_batt, "wilcoxon_p": float(p), "cliffs_delta": float(delta),
            "mean_abs_err_a": float(per_batt["abs_err_a"].mean()), "mean_abs_err_b": float(per_batt["abs_err_b"].mean()),
            "note": f"lower abs_err is better ({label_a} vs {label_b})"}


def r2_metric(y, p):
    return r2_score(y, p)


def run_comparison(name, battery_ids, y_true, pred_baseline, pred_candidate, label_baseline, label_candidate):
    print(f"\n--- {name} ---")
    r2_base = r2_score(y_true, pred_baseline)
    r2_cand = r2_score(y_true, pred_candidate)
    rmse_base = float(np.sqrt(mean_squared_error(y_true, pred_baseline)))
    rmse_cand = float(np.sqrt(mean_squared_error(y_true, pred_candidate)))
    print(f"[stat-consolidation] point estimates: {label_baseline} R2={r2_base:.4f} RMSE={rmse_base:.4f} | "
          f"{label_candidate} R2={r2_cand:.4f} RMSE={rmse_cand:.4f}")

    boot = battery_bootstrap_delta_ci(battery_ids, pred_baseline, pred_candidate, r2_metric, y_true)
    print(f"[stat-consolidation] battery-bootstrap R2 delta ({label_candidate} - {label_baseline}): "
          f"mean={boot['mean_delta_r2']:+.4f} 95% CI=[{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}] -> "
          f"{'SIGNIFICANT' if boot['significant'] else 'NOT significant'}")

    wil = paired_wilcoxon_per_battery(battery_ids, y_true, pred_baseline, pred_candidate, label_baseline, label_candidate)
    if wil.get("wilcoxon_p") is not None:
        print(f"[stat-consolidation] paired Wilcoxon (per-battery mean |err|, n={wil['n_batteries']} batteries): "
              f"p={wil['wilcoxon_p']:.4f} Cliff's delta={wil['cliffs_delta']:+.3f} "
              f"({label_baseline} mean|err|={wil['mean_abs_err_a']:.4f} vs {label_candidate} mean|err|={wil['mean_abs_err_b']:.4f})")
    else:
        print(f"[stat-consolidation] paired Wilcoxon: {wil['note']} (n_batteries={wil['n_batteries']})")

    return {"comparison": name, "label_baseline": label_baseline, "label_candidate": label_candidate,
            "r2_baseline": r2_base, "r2_candidate": r2_cand, "rmse_baseline": rmse_base, "rmse_candidate": rmse_cand,
            "bootstrap_mean_delta_r2": boot["mean_delta_r2"], "bootstrap_ci_lo": boot["ci_lo"],
            "bootstrap_ci_hi": boot["ci_hi"], "bootstrap_significant": boot["significant"],
            "wilcoxon_n_batteries": wil["n_batteries"], "wilcoxon_p": wil.get("wilcoxon_p"),
            "cliffs_delta": wil.get("cliffs_delta"), "wilcoxon_note": wil.get("note")}


def impute(X, medians):
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    return X


def main():
    t0 = time.time()
    print("=== Research pass 2, item 5: statistical consolidation pass ===")
    results = []

    # ---- Comparison 1: ensemble vs. standalone XGBoost (expanded pool, already-saved paired preds) ----
    ens = pd.read_csv(PROC_DIR / "predictions" / "ensemble_fusion_expanded_test_preds.csv")
    results.append(run_comparison(
        "Ensemble (Stacking-Ridge-fusion) vs. standalone XGBoost-fusion (expanded pool, in-domain TEST)",
        ens["battery_id"].to_numpy(), ens["SOH"].to_numpy(),
        ens["pred_XGBoost_fusion"].to_numpy(), ens["pred_Stacking_Ridge_fusion_expanded"].to_numpy(),
        "standalone XGBoost", "Stacking-Ridge ensemble"))

    # ---- Comparison 2: reformulated (Stage 5 extended) vs. original (deployed, Stage 1.1-only) features ----
    print("\n[stat-consolidation] Comparison 2: scoring both models (pure inference, no training) on "
          "identical rows for genuine pairing...")
    merged_nm, hi_full = load_nasa_mit_pool(reformulated=True)
    hi_full_ext = add_scv_matd_viect_reformulated(hi_full)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    nasa_mit_ext = hi_full_ext[hi_full_ext["dataset"].isin(["NASA", "MIT"])]
    merged_ext = pd.merge(nasa_mit_ext, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    train_mask_ext, test_mask_ext, _ = battery_split_masks(merged_ext)

    base_cols = canonical_feature_cols(reformulated=True)
    extended_cols = extended_canonical_feature_cols(base_cols)
    orig_feature_cols = base_cols + ["cycle_idx"]
    ext_feature_cols = extended_cols + ["cycle_idx"]
    fcols = fusion_cols()

    deployed_model = XGBRegressor(); deployed_model.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    ext_model = XGBRegressor(); ext_model.load_model(str(ROOT / "models" / "_experimental_xgb_soh_fusion_extended_reformulation.json"))

    X_orig_train = merged_ext.loc[train_mask_ext, orig_feature_cols + fcols].to_numpy(dtype=float, copy=True)
    orig_medians = np.nanmedian(np.where(np.isinf(X_orig_train), np.nan, X_orig_train), axis=0)
    X_ext_train = merged_ext.loc[train_mask_ext, ext_feature_cols + fcols].to_numpy(dtype=float, copy=True)
    ext_medians = np.nanmedian(np.where(np.isinf(X_ext_train), np.nan, X_ext_train), axis=0)

    calce_ext = add_scv_matd_viect_reformulated(build_calce_merged(hi_full))
    oxford_ext = add_scv_matd_viect_reformulated(pd.read_parquet(PROC_DIR / "stage5_1_oxford_merged.parquet"))
    hust_ext = add_scv_matd_viect_reformulated(pd.read_parquet(PROC_DIR / "stage5_1_hust_merged.parquet"))
    xjtu_ext = add_scv_matd_viect_reformulated(pd.read_parquet(PROC_DIR / "stage5_1_xjtu_merged.parquet"))

    for name, df in [("in-domain (TEST)", merged_ext[test_mask_ext]), ("CALCE", calce_ext),
                      ("Oxford", oxford_ext), ("HUST", hust_ext), ("XJTU", xjtu_ext)]:
        X_orig = impute(df[orig_feature_cols + fcols].to_numpy(dtype=float, copy=True), orig_medians)
        X_ext = impute(df[ext_feature_cols + fcols].to_numpy(dtype=float, copy=True), ext_medians)
        pred_orig = deployed_model.predict(X_orig)
        pred_ext = ext_model.predict(X_ext)
        y_true = df["SOH"].to_numpy(dtype=float)
        results.append(run_comparison(
            f"Reformulated (Stage 5 extended SCV/MATD/VIECT) vs. original (deployed, Stage 1.1) features - {name}",
            df["battery_id"].to_numpy(), y_true, pred_orig, pred_ext,
            "original (deployed) features", "Stage 5 extended-reformulated features"))

    # ---- Comparison 3: deployed model vs. Severson/Attia baselines ----
    print("\n[stat-consolidation] Comparison 3: refitting tiny Severson/Attia ElasticNetCV baselines "
          "(disclosed exception - pure reproduction of Stage 6.1's own already-published methodology, "
          "on already-cached features, solely to recover per-row predictions for pairing) ...")
    pool_df = pd.read_parquet(PROC_DIR / "severson_features_42pool.parquet")
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    sev_train = pool_df[pool_df["battery_id"].isin(train_ids)]
    sev_test = pool_df[pool_df["battery_id"].isin(test_ids)]

    # Held-out feature frames already loaded in Comparison 2 (calce_ext/
    # oxford_ext/hust_ext/xjtu_ext) carry battery_id+cycle_idx - used
    # here to merge-by-KEY with Severson's own (battery_id, cycle_idx)
    # rows and score the deployed model fresh on the matched rows,
    # rather than risk a fragile positional join against a separately-
    # generated percycle_*_deployed.csv with no cycle_idx column at all.
    held_feature_frames = {"CALCE": calce_ext, "Oxford": oxford_ext, "HUST": hust_ext, "XJTU": xjtu_ext}
    held_severson = {
        "CALCE": pd.read_parquet(PROC_DIR / "severson_features_calce.parquet"),
        "Oxford": pd.read_parquet(PROC_DIR / "severson_features_oxford.parquet"),
        "HUST": pd.read_parquet(PROC_DIR / "severson_features_hust.parquet"),
        "XJTU": pd.read_parquet(PROC_DIR / "severson_features_xjtu.parquet"),
    }

    for label, feat_cols in [("Severson variance model (1 feat)", VARIANCE_FEATS),
                              ("Attia-style rich model (4 feat)", RICH_FEATS)]:
        X_tr = sev_train[feat_cols].to_numpy(dtype=float)
        y_tr = sev_train["SOH"].to_numpy(dtype=float)
        med = np.nanmedian(X_tr, axis=0)
        X_tr = impute(X_tr.copy(), med)
        scaler = StandardScaler().fit(X_tr)
        m = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=5000, random_state=SEED)
        m.fit(scaler.transform(X_tr), y_tr)

        X_te = impute(sev_test[feat_cols].to_numpy(dtype=float).copy(), med)
        pred_sev_indomain = m.predict(scaler.transform(X_te))
        deployed_indomain = pd.read_csv(PROC_DIR / "predictions" / "xgb_fusion_preds.csv")
        deployed_indomain = deployed_indomain[deployed_indomain["split"] == "test"]
        merged_indomain = pd.merge(
            sev_test[["battery_id", "cycle_idx", "SOH"]].assign(pred_severson=pred_sev_indomain),
            deployed_indomain[["battery_id", "cycle_idx", "y_pred_soh_fusion"]],
            on=["battery_id", "cycle_idx"], how="inner")
        if len(merged_indomain) > 0:
            results.append(run_comparison(
                f"Deployed model vs. {label} - in-domain (TEST)",
                merged_indomain["battery_id"].to_numpy(), merged_indomain["SOH"].to_numpy(),
                merged_indomain["pred_severson"].to_numpy(), merged_indomain["y_pred_soh_fusion"].to_numpy(),
                label, "deployed XGBoost-fusion"))
        else:
            print(f"[stat-consolidation] WARNING: {label} in-domain merge produced 0 rows "
                  f"(cycle_idx convention mismatch between severson features and xgb_fusion_preds.csv) - skipped")

        for name, sev_df in held_severson.items():
            X_h = impute(sev_df[feat_cols].to_numpy(dtype=float).copy(), med)
            pred_sev_h = m.predict(scaler.transform(X_h))
            sev_with_pred = sev_df[["battery_id", "cycle_idx", "SOH"]].assign(pred_severson=pred_sev_h)

            feat_df = held_feature_frames[name]
            X_dep = impute(feat_df[orig_feature_cols + fcols].to_numpy(dtype=float, copy=True), orig_medians)
            pred_dep_all = deployed_model.predict(X_dep)
            dep_with_pred = feat_df[["battery_id", "cycle_idx"]].assign(pred_deployed=pred_dep_all)

            merged_h = pd.merge(sev_with_pred, dep_with_pred, on=["battery_id", "cycle_idx"], how="inner")
            if len(merged_h) == 0:
                print(f"[stat-consolidation] WARNING: {label} / {name}: key-based merge (battery_id, "
                      f"cycle_idx) produced 0 matching rows between severson features and the deployed "
                      f"model's own feature frame - skipped")
                continue
            results.append(run_comparison(
                f"Deployed model vs. {label} - {name}",
                merged_h["battery_id"].to_numpy(), merged_h["SOH"].to_numpy(),
                merged_h["pred_severson"].to_numpy(), merged_h["pred_deployed"].to_numpy(),
                label, "deployed XGBoost-fusion"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass2_item5_statistical_consolidation.csv", index=False)
    print("\n=== CONSOLIDATED SIGNIFICANCE TABLE ===")
    print(results_df[["comparison", "r2_baseline", "r2_candidate", "bootstrap_mean_delta_r2",
                       "bootstrap_significant", "wilcoxon_p", "cliffs_delta"]].to_string(index=False))
    print(f"\n[stat-consolidation] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
