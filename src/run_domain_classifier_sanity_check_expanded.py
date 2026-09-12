"""
Dataset Expansion Phase 1, Step 4: re-runs session 19's IN-DOMAIN
domain-classifier sanity check on the expanded pool's test set - NOT
the full weighted-conformal-on-CALCE experiment (which depends on the
session-13 MMD-aligned encoder, out of scope for this additive dataset-
expansion pass), just the specific diagnostic that matters for the
"more batteries -> less spurious separability" hypothesis test:

Session 19's original finding: splitting the (then only 6) NASA+MIT
test batteries into a calib half and an eval half and fitting a
logistic-regression domain classifier on their (7 BFA HI + 16 fusion
embedding) features gave AUC=0.902 - should be ~0.5 for two halves of
the SAME in-domain distribution, but individual-battery idiosyncrasies
dominated at that small a battery count.

Same method, same feature space (7 expanded-BFA-selected HIs + 16
expanded fusion dims), same calib/eval battery split logic
(run_conformal.calib_eval_battery_split, imported unchanged) - applied
to the expanded test set, whatever size battery_level_split produced.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conformal import calib_eval_battery_split

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"


def domain_classifier_auc(X_a: np.ndarray, X_b: np.ndarray, seed: int = 42) -> float:
    """Fits ONE LogisticRegression to distinguish X_a (label 0) from
    X_b (label 1), returns its OWN in-sample AUC on the same data it was
    fit on - matching session 19's exact diagnostic (this is checking
    whether the classifier CAN separate the two groups at all, not doing
    a held-out generalization test, since the point is to see how
    separable the raw feature distributions are)."""
    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(Xs, y)
    p = clf.predict_proba(Xs)[:, 1]
    return float(roc_auc_score(y, p))


def main():
    with open(PROC_DIR / "bfa_selected_features_expanded.txt") as f:
        selected = [l.strip() for l in f if l.strip()]
    fusion_cols = [f"fusion_{i}" for i in range(16)]
    feature_cols = selected + fusion_cols

    hi_df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    hi_df = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings_expanded.csv")
    merged = pd.merge(hi_df, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")

    X = merged[feature_cols].to_numpy(dtype=float, copy=True)
    X[np.isinf(X)] = np.nan  # see run_bfa_expanded.py's comment: one MIT/b2c30 cycle's VDEDT is inf
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    merged_clean = merged.copy()
    for i, c in enumerate(feature_cols):
        merged_clean[c] = X[:, i]

    ens_test = pd.read_csv(PRED_DIR / "ensemble_fusion_expanded_test_preds.csv")
    test_battery_ids = ens_test["battery_id"].unique().tolist()
    print(f"[domain-sanity-exp] expanded test set: {len(test_battery_ids)} batteries "
          f"(original session 19 run had only 6)")

    calib_ids, eval_ids = calib_eval_battery_split(test_battery_ids)
    print(f"[domain-sanity-exp] calib batteries: {len(calib_ids)}, eval batteries: {len(eval_ids)}")

    calib_rows = merged_clean[merged_clean["battery_id"].isin(calib_ids)]
    eval_rows = merged_clean[merged_clean["battery_id"].isin(eval_ids)]
    print(f"[domain-sanity-exp] calib cycles: {len(calib_rows)}, eval cycles: {len(eval_rows)}")

    X_calib_full = calib_rows[feature_cols].to_numpy()
    X_eval_full = eval_rows[feature_cols].to_numpy()
    auc_full = domain_classifier_auc(X_calib_full, X_eval_full)

    X_calib_fusiononly = calib_rows[fusion_cols].to_numpy()
    X_eval_fusiononly = eval_rows[fusion_cols].to_numpy()
    auc_fusiononly = domain_classifier_auc(X_calib_fusiononly, X_eval_fusiononly)

    print(f"\n[domain-sanity-exp] IN-DOMAIN AUC (calib-half vs. eval-half, "
          f"should be ~0.5 for a genuinely in-domain split):")
    print(f"    full feature space (7 HI + 16 fusion): AUC={auc_full:.4f}")
    print(f"    fusion-only (16-dim):                  AUC={auc_fusiononly:.4f}")
    print(f"\n[domain-sanity-exp] ORIGINAL 6-battery-test-set reference: "
          f"AUC=0.9021 (full feature space)")

    delta = 0.9021 - auc_full
    closer = "YES, moved closer to 0.5" if abs(auc_full - 0.5) < abs(0.9021 - 0.5) else "NO, did not move closer to 0.5"
    print(f"[domain-sanity-exp] Moved closer to 0.5 with more test batteries? {closer} "
          f"(|AUC-0.5| original=0.4021, expanded={abs(auc_full-0.5):.4f})")

    pd.DataFrame([
        {"feature_space": "full_7HI_16fusion", "n_calib_batteries": len(calib_ids),
         "n_eval_batteries": len(eval_ids), "n_calib_cycles": len(calib_rows),
         "n_eval_cycles": len(eval_rows), "auc": auc_full,
         "original_6battery_auc": 0.9021},
        {"feature_space": "fusion_only_16dim", "n_calib_batteries": len(calib_ids),
         "n_eval_batteries": len(eval_ids), "n_calib_cycles": len(calib_rows),
         "n_eval_cycles": len(eval_rows), "auc": auc_fusiononly,
         "original_6battery_auc": np.nan},
    ]).to_csv(OUT_DIR / "domain_classifier_sanity_check_expanded.csv", index=False)

    print("\n[domain-sanity-exp] DONE")


if __name__ == "__main__":
    main()
