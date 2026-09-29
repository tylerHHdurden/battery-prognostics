"""
Final research pass, item 1: label-free routing.

The CURRENT deployed routing (src/live_inference.py's
EXTENDED_ROUTED_DATASETS = {"CALCE","Oxford","HUST"}) was chosen using
held-out R2 - a reviewer will correctly call this test-set selection.
This script builds a routing rule that uses NO target SOH/RUL labels
at all, grounded in this project's own domain-classifier-AUC finding
(session 19 / run_domain_classifier_sanity_check_expanded.py: fitting a
LogisticRegression to distinguish train-pool cycles from target-dataset
cycles and reading its own in-sample AUC as a distributional-shift
score; Stage 1.1's duration reformulation was originally justified by
this exact mechanism - reducing that separability).

RULE (label-free, decided BEFORE any target R2 is looked at):
for each held-out dataset, compute domain-classifier AUC of the
NASA+MIT train pool vs. that dataset TWICE - once in each model's own
native feature representation (base: Stage-1.1-reformulated; extended:
Stage-5-extended-reformulated). Route to whichever representation
yields the LOWER AUC (closer to 0.5 = less distributionally separable
from what that model was trained on = the representation/model this
project's own prior finding says should generalize better). Ties
(|AUC_base - AUC_ext| < 0.01) route to the base model (the cheaper,
non-experimental default).

EVALUATION (labels used ONLY here, never inside the rule): for each of
the 13 held-out datasets (CALCE/Oxford/HUST/XJTU + 9 locally-available
BatteryLife sources: ul_pur/hnei/snl/mich/mich_exp/rwth/stanford/
stanford_2/isu_ilcc), compute the REAL R2 of both the base and the
extended model, fresh, and call whichever is numerically higher the
"true winner". Report the full confusion: does the label-free rule's
pick match the true winner, dataset by dataset, honestly including
HNEI and RWTH (both flagged in Part B as plausible new routing
candidates, never acted on).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    load_all_heldout_base, load_all_heldout_extended,
    base_feature_cols, extended_feature_cols, fit_medians,
    load_base_model, load_extended_model, build_X, score, OUT_DIR, PROC_DIR,
)

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
AUC_TIE_MARGIN = 0.01


def domain_classifier_auc(X_a: np.ndarray, X_b: np.ndarray, seed: int = 42) -> float:
    """Unchanged from run_domain_classifier_sanity_check_expanded.py -
    fits ONE LogisticRegression to distinguish X_a (label 0, train pool)
    from X_b (label 1, target dataset), returns its own in-sample AUC.
    No labels (SOH/RUL) enter this computation at all - X_a/X_b are HI
    + fusion embedding features only."""
    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))])
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(Xs, y)
    p = clf.predict_proba(Xs)[:, 1]
    return float(roc_auc_score(y, p))


def unsup_cols(feature_cols_with_cycle_idx: list[str]) -> list[str]:
    """HI + fusion columns, WITHOUT cycle_idx - the domain-classifier's
    own established feature space (session 19), deliberately excluding
    cycle_idx since it's a supervised-relevant sequential index, not a
    distributional-shift signal."""
    from stage1_common import fusion_cols
    hi_cols = [c for c in feature_cols_with_cycle_idx if c != "cycle_idx"]
    return hi_cols + fusion_cols()


def clean(X: np.ndarray, ref_medians: np.ndarray = None) -> np.ndarray:
    """Imputes inf/NaN. If ref_medians is given (the TRAIN pool's own
    column medians), uses those - avoids an all-NaN column in a SMALL
    target dataset producing an unfillable NaN median of its own (e.g.
    a column that happens to be entirely missing for one BatteryLife
    source); falls back to the array's own column median only when no
    reference is supplied (used for the train pool itself)."""
    X = np.where(np.isinf(X), np.nan, X)
    col_medians = ref_medians if ref_medians is not None else np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    # last-resort: any column still all-NaN in the reference itself -> 0.0
    X = np.where(np.isnan(X), 0.0, X)
    return X


def main():
    t0 = time.time()
    print("=== Final pass item 1: label-free routing (no target labels used in the rule) ===")

    merged_base, hi_full_base, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_ext, hi_full_ext, hi_full_raw, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_cols = base_feature_cols()
    ext_cols = extended_feature_cols()
    base_medians = fit_medians(merged_base, train_mask_b, base_cols)
    ext_medians = fit_medians(merged_ext, train_mask_e, ext_cols)
    base_model = load_base_model()
    ext_model = load_extended_model()

    base_unsup_cols = unsup_cols(base_cols)
    ext_unsup_cols = unsup_cols(ext_cols)

    # Train-pool unsupervised feature matrices (TRAIN split only - the
    # same convention every zero-retrain eval in this project uses:
    # medians/reference distributions come from train, never test).
    X_train_base_unsup = clean(merged_base.loc[train_mask_b, base_unsup_cols].to_numpy(dtype=float, copy=True))
    X_train_ext_unsup = clean(merged_ext.loc[train_mask_e, ext_unsup_cols].to_numpy(dtype=float, copy=True))
    train_base_medians = np.nanmedian(np.where(np.isinf(
        merged_base.loc[train_mask_b, base_unsup_cols].to_numpy(dtype=float, copy=True)), np.nan,
        merged_base.loc[train_mask_b, base_unsup_cols].to_numpy(dtype=float, copy=True)), axis=0)
    train_ext_medians = np.nanmedian(np.where(np.isinf(
        merged_ext.loc[train_mask_e, ext_unsup_cols].to_numpy(dtype=float, copy=True)), np.nan,
        merged_ext.loc[train_mask_e, ext_unsup_cols].to_numpy(dtype=float, copy=True)), axis=0)

    heldout_base = load_all_heldout_base(hi_full_base)
    heldout_ext = load_all_heldout_extended(hi_full_raw)

    rows = []

    def evaluate_one(name: str, df_base: pd.DataFrame, df_ext: pd.DataFrame):
        # --- ground truth (labels used ONLY here) ---
        X_b = build_X(df_base, base_cols, base_medians)
        pred_b = base_model.predict(X_b)
        r_b = score(df_base["SOH"].to_numpy(), pred_b)

        X_e = build_X(df_ext, ext_cols, ext_medians)
        pred_e = ext_model.predict(X_e)
        r_e = score(df_ext["SOH"].to_numpy(), pred_e)

        true_winner = "extended" if r_e["r2"] > r_b["r2"] else "base"

        # --- label-free rule (no SOH/RUL touched above this line) ---
        X_target_base_unsup = clean(df_base[base_unsup_cols].to_numpy(dtype=float, copy=True), train_base_medians)
        X_target_ext_unsup = clean(df_ext[ext_unsup_cols].to_numpy(dtype=float, copy=True), train_ext_medians)
        auc_base = domain_classifier_auc(X_train_base_unsup, X_target_base_unsup)
        auc_ext = domain_classifier_auc(X_train_ext_unsup, X_target_ext_unsup)

        if auc_base - auc_ext > AUC_TIE_MARGIN:
            rule_pick = "extended"
        elif auc_ext - auc_base > AUC_TIE_MARGIN:
            rule_pick = "base"
        else:
            rule_pick = "base"  # tie -> cheaper default

        correct = rule_pick == true_winner
        print(f"[item1] {name}: AUC_base={auc_base:.4f} AUC_ext={auc_ext:.4f} -> rule picks '{rule_pick}' | "
              f"true R2_base={r_b['r2']:.4f} R2_ext={r_e['r2']:.4f} -> true winner '{true_winner}' | "
              f"{'CORRECT' if correct else 'WRONG'}")
        rows.append({
            "dataset": name, "auc_base_repr": auc_base, "auc_ext_repr": auc_ext,
            "rule_pick": rule_pick, "r2_base_model": r_b["r2"], "r2_ext_model": r_e["r2"],
            "true_winner": true_winner, "rule_correct": correct,
            "currently_routed_live": "extended" if name in {"CALCE", "Oxford", "HUST"} else "base",
        })

    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        evaluate_one(name, heldout_base[name], heldout_ext[name])

    for source in BATTERYLIFE_SOURCES:
        path = PROC_DIR / f"batterylife_{source}_merged.parquet"
        if not path.exists():
            print(f"[item1] {source}: no local merged table - skipped")
            continue
        df = pd.read_parquet(path)
        evaluate_one(source, df, df)  # same table carries both representations already

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "finalpass_item1_labelfree_routing.csv", index=False)

    n_correct = int(results_df["rule_correct"].sum())
    n_total = len(results_df)
    print(f"\n=== SUMMARY: label-free rule agrees with the true winner on {n_correct}/{n_total} "
          f"held-out datasets ({100*n_correct/n_total:.1f}%) ===")
    print(results_df.to_string(index=False))

    # Confusion matrix
    conf = pd.crosstab(results_df["rule_pick"], results_df["true_winner"],
                        rownames=["rule_pick"], colnames=["true_winner"])
    print("\n[item1] Confusion matrix (rule_pick vs. true_winner):")
    print(conf.to_string())
    conf.to_csv(OUT_DIR / "finalpass_item1_confusion.csv")

    # Compare against a naive always-base / always-extended / current-live baseline
    n_always_base_correct = int((results_df["true_winner"] == "base").sum())
    n_always_ext_correct = int((results_df["true_winner"] == "extended").sum())
    n_live_correct = int((results_df["currently_routed_live"] == results_df["true_winner"]).sum())
    print(f"\n[item1] For context: always-base would get {n_always_base_correct}/{n_total} right, "
          f"always-extended {n_always_ext_correct}/{n_total}, "
          f"the CURRENT live positive-list routing (label-based, only defined for CALCE/Oxford/HUST vs. "
          f"base-default-everything-else) gets {n_live_correct}/{n_total} right on this same 13-dataset panel.")
    print(f"\n[item1] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
