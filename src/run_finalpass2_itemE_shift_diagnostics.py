"""
Final experiment pass 2, item E: shift diagnostics.

Part 1: Spearman + Pearson correlation, across the 13 held-out
datasets, of the domain-classifier AUC (item 1's own diagnostic, using
whichever feature representation the deployed system ACTUALLY routes
each dataset to) against R2, MAE, and conformal coverage (this pass's
own already-computed routed numbers: item 5a's 5-seed aggregate for
R2/MAE, item 3/5e's k=0 coverage) - with bootstrap CIs (resample the 13
datasets with replacement, n=2000).

Part 2: a per-BATTERY OOD score (mean predict_proba of a source-vs-
target LogisticRegression domain classifier, same feature space as
item 1's AUC diagnostic) and a risk-coverage curve (MAE of retained
batteries as the abstention rate sweeps 0->50%, retaining the
LOWEST-OOD-score batteries first - standard selective-prediction
protocol). A SEPARATE, single reference threshold is calibrated using
ONLY source (in-domain) data - the in-domain calib/eval battery split's
own OOD-score distribution under an analogous classifier - and applied
fixed to each target dataset, reporting the abstention rate/MAE that
ONE source-only threshold actually produces there (not swept against
target data).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    load_all_heldout_base, load_all_heldout_extended,
    base_feature_cols, extended_feature_cols, fit_medians,
    load_base_model, load_extended_model, build_X, OUT_DIR, PROC_DIR,
)
from run_conformal import calib_eval_battery_split
from stage1_common import fusion_cols

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
SEED = 42
N_BOOT = 2000
ABSTENTION_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def unsup_cols(feature_cols_with_cycle_idx):
    hi_cols = [c for c in feature_cols_with_cycle_idx if c != "cycle_idx"]
    return hi_cols + fusion_cols()


def clean(X, ref_medians=None):
    X = np.where(np.isinf(X), np.nan, X)
    col_medians = ref_medians if ref_medians is not None else np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    X = np.where(np.isnan(X), 0.0, X)
    return X


def bootstrap_corr(x, y, method, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
            continue
        r = spearmanr(xs, ys).correlation if method == "spearman" else pearsonr(xs, ys)[0]
        if np.isfinite(r):
            vals.append(r)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    print("=== Final pass 2, item E: shift diagnostics (AUC vs. R2/MAE/coverage correlations + risk-coverage) ===")

    item1_df = pd.read_csv(OUT_DIR / "finalpass_item1_labelfree_routing.csv")
    item5a_df = pd.read_csv(OUT_DIR / "finalpass_item5a_5seed_aggregate.csv")
    coverage_df = pd.read_csv(OUT_DIR / "finalpass_item5e_conformal_consolidated.csv")

    rows = []
    for _, r in item1_df.iterrows():
        name = r["dataset"]
        routed_ext = name in EXTENDED_ROUTED_DATASETS
        auc = r["auc_ext_repr"] if routed_ext else r["auc_base_repr"]
        r2_row = item5a_df[item5a_df["dataset"] == name]
        cov_row = coverage_df[coverage_df["dataset"] == name]
        if len(r2_row) == 0 or len(cov_row) == 0:
            continue
        rows.append({
            "dataset": name, "auc": auc,
            "r2": float(r2_row["r2_mean"].iloc[0]), "mae": float(r2_row["mae_mean"].iloc[0]),
            "coverage": float(cov_row["coverage"].iloc[0]),
        })
    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(OUT_DIR / "finalpass2_itemE_auc_vs_metrics.csv", index=False)
    print(f"\n[itemE] Part 1 input table (n={len(diag_df)} datasets):")
    print(diag_df.to_string(index=False))

    print("\n=== Part 1: AUC correlation with R2/MAE/coverage (n=13, bootstrap 95% CI) ===")
    corr_rows = []
    for target_col in ["r2", "mae", "coverage"]:
        x = diag_df["auc"].to_numpy()
        y = diag_df[target_col].to_numpy()
        sp = spearmanr(x, y)
        pe = pearsonr(x, y)
        sp_lo, sp_hi = bootstrap_corr(x, y, "spearman")
        pe_lo, pe_hi = bootstrap_corr(x, y, "pearson")
        print(f"[itemE] AUC vs {target_col}: Spearman rho={sp.correlation:.3f} [95% CI {sp_lo:.3f},{sp_hi:.3f}], "
              f"Pearson r={pe[0]:.3f} [95% CI {pe_lo:.3f},{pe_hi:.3f}]")
        corr_rows.append({"target": target_col, "spearman_rho": sp.correlation, "spearman_ci_lo": sp_lo,
                          "spearman_ci_hi": sp_hi, "pearson_r": pe[0], "pearson_ci_lo": pe_lo, "pearson_ci_hi": pe_hi})
    pd.DataFrame(corr_rows).to_csv(OUT_DIR / "finalpass2_itemE_correlations.csv", index=False)

    print("\n=== Part 2: per-battery OOD score + risk-coverage curve ===")
    merged_base, hi_full_base, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_ext, hi_full_ext, hi_full_raw, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_cols = base_feature_cols()
    ext_cols = extended_feature_cols()
    base_medians = fit_medians(merged_base, train_mask_b, base_cols)
    ext_medians = fit_medians(merged_ext, train_mask_e, ext_cols)
    base_model = load_base_model()
    ext_model = load_extended_model()
    base_unsup = unsup_cols(base_cols)
    ext_unsup = unsup_cols(ext_cols)

    X_train_base = clean(merged_base.loc[train_mask_b, base_unsup].to_numpy(dtype=float, copy=True))
    X_train_ext = clean(merged_ext.loc[train_mask_e, ext_unsup].to_numpy(dtype=float, copy=True))
    train_base_medians = np.nanmedian(np.where(np.isinf(
        merged_base.loc[train_mask_b, base_unsup].to_numpy(dtype=float, copy=True)), np.nan,
        merged_base.loc[train_mask_b, base_unsup].to_numpy(dtype=float, copy=True)), axis=0)
    train_ext_medians = np.nanmedian(np.where(np.isinf(
        merged_ext.loc[train_mask_e, ext_unsup].to_numpy(dtype=float, copy=True)), np.nan,
        merged_ext.loc[train_mask_e, ext_unsup].to_numpy(dtype=float, copy=True)), axis=0)

    # source-only threshold reference: split the SAME train pool's own
    # calib/eval battery halves (in-domain, no real target ever involved),
    # fit a classifier calib-vs-eval, score the eval half's OWN batteries -
    # this is the "in-domain, no-drift" OOD-score distribution to calibrate
    # a threshold from, using ONLY source information.
    test_bids_b = merged_base.loc[test_mask_b, "battery_id"].to_numpy()
    unique_test_ids = sorted(set(test_bids_b.tolist()))
    calib_ids, eval_ids = calib_eval_battery_split(unique_test_ids)
    calib_mask = np.isin(test_bids_b, calib_ids)
    eval_mask = np.isin(test_bids_b, eval_ids)
    X_calib_src = clean(merged_base.loc[test_mask_b, base_unsup].to_numpy(dtype=float, copy=True)[calib_mask], train_base_medians)
    X_eval_src = clean(merged_base.loc[test_mask_b, base_unsup].to_numpy(dtype=float, copy=True)[eval_mask], train_base_medians)
    Xs = np.vstack([X_calib_src, X_eval_src])
    ys = np.concatenate([np.zeros(len(X_calib_src)), np.ones(len(X_eval_src))])
    scaler_src = StandardScaler().fit(Xs)
    clf_src = LogisticRegression(max_iter=1000, random_state=SEED).fit(scaler_src.transform(Xs), ys)
    eval_probs = clf_src.predict_proba(scaler_src.transform(X_eval_src))[:, 1]
    eval_bids = merged_base.loc[test_mask_b, "battery_id"].to_numpy()[eval_mask]
    eval_batt_scores = pd.Series(eval_probs).groupby(eval_bids).mean()
    source_only_threshold = float(np.percentile(eval_batt_scores, 90))
    print(f"[itemE] source-only threshold (90th pctl of in-domain eval-half battery OOD scores, "
          f"using an in-domain calib-vs-eval classifier, never touching real target data): "
          f"{source_only_threshold:.4f}")

    heldout_base = load_all_heldout_base(hi_full_base)
    heldout_ext = load_all_heldout_extended(hi_full_raw)

    def get_target(name):
        if name in {"CALCE", "Oxford", "HUST", "XJTU"}:
            routed = name in EXTENDED_ROUTED_DATASETS
            df = heldout_ext[name] if routed else heldout_base[name]
            cols, medians, model, unsup, train_X, train_med = (
                (ext_cols, ext_medians, ext_model, ext_unsup, X_train_ext, train_ext_medians) if routed
                else (base_cols, base_medians, base_model, base_unsup, X_train_base, train_base_medians))
        else:
            df = pd.read_parquet(PROC_DIR / f"batterylife_{name}_merged.parquet")
            cols, medians, model, unsup, train_X, train_med = base_cols, base_medians, base_model, base_unsup, X_train_base, train_base_medians
        return df, cols, medians, model, unsup, train_X, train_med

    rc_rows, thresh_rows = [], []
    all_datasets = ["CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_SOURCES
    for name in all_datasets:
        if name in BATTERYLIFE_SOURCES and not (PROC_DIR / f"batterylife_{name}_merged.parquet").exists():
            continue
        df, cols, medians, model, unsup, train_X, train_med = get_target(name)
        X_pred = build_X(df, cols, medians)
        pred = model.predict(X_pred)
        y_true = df["SOH"].to_numpy()
        bids = df["battery_id"].to_numpy()

        X_target_unsup = clean(df[unsup].to_numpy(dtype=float, copy=True), train_med)
        Xp = np.vstack([train_X, X_target_unsup])
        yp = np.concatenate([np.zeros(len(train_X)), np.ones(len(X_target_unsup))])
        scaler = StandardScaler().fit(Xp)
        clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(scaler.transform(Xp), yp)
        target_probs = clf.predict_proba(scaler.transform(X_target_unsup))[:, 1]

        battery_ood = pd.Series(target_probs).groupby(bids).mean()
        battery_mae = pd.DataFrame({"battery_id": bids, "abs_err": np.abs(y_true - pred)}).groupby("battery_id")["abs_err"].mean()
        battery_ood, battery_mae = battery_ood.align(battery_mae)
        order = battery_ood.sort_values().index  # lowest OOD (most in-domain-like) first

        for rate in ABSTENTION_RATES:
            n_keep = max(1, int(round(len(order) * (1 - rate))))
            kept = order[:n_keep]
            mae_kept = float(battery_mae.loc[kept].mean())
            rc_rows.append({"dataset": name, "abstention_rate": rate, "n_batteries_kept": n_keep,
                            "n_batteries_total": len(order), "mae_retained": mae_kept})

        n_abstain_at_source_thresh = int((battery_ood > source_only_threshold).sum())
        rate_at_source_thresh = n_abstain_at_source_thresh / len(battery_ood)
        kept_at_thresh = battery_ood[battery_ood <= source_only_threshold].index
        mae_at_thresh = float(battery_mae.loc[kept_at_thresh].mean()) if len(kept_at_thresh) else np.nan
        thresh_rows.append({"dataset": name, "source_only_threshold": source_only_threshold,
                            "resulting_abstention_rate": rate_at_source_thresh,
                            "n_abstained": n_abstain_at_source_thresh, "n_total": len(battery_ood),
                            "mae_retained_at_threshold": mae_at_thresh})
        print(f"[itemE] {name}: source-only threshold ({source_only_threshold:.3f}) abstains on "
              f"{n_abstain_at_source_thresh}/{len(battery_ood)} batteries "
              f"({rate_at_source_thresh:.1%}), MAE of retained={mae_at_thresh:.3f}")

    rc_df = pd.DataFrame(rc_rows)
    rc_df.to_csv(OUT_DIR / "finalpass2_itemE_risk_coverage.csv", index=False)
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(OUT_DIR / "finalpass2_itemE_source_threshold.csv", index=False)

    print("\n=== Risk-coverage summary (mean MAE-retained across all datasets, by abstention rate) ===")
    print(rc_df.groupby("abstention_rate")["mae_retained"].mean().to_string())


if __name__ == "__main__":
    main()
