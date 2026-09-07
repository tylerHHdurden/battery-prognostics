"""
Phase 5 follow-up: LIME (Local Interpretable Model-agnostic Explanations)
as a SECOND, INDEPENDENT explainability method alongside the existing
TreeSHAP analysis (run_shap_analysis.py — untouched, read fully before
writing this, not modified by this script).

Read-only comparison, additive: does not change or remove anything in
run_shap_analysis.py or its output files (shap_xgboost_base_ranking.csv,
shap_meta_ranking.csv, shap_deep_models_summary.csv). This script
re-explains the SAME per-instance predictions run_shap_analysis.py
already covers for the two XGBoost models (base learner: 7 BFA-selected
HIs; Stacking-XGBoost meta-learner: 4 base-learner predictions) with
LIME's tabular explainer, and reports whether LIME's top-3 features
agree with TreeSHAP's top-3 for each instance. LIME is NOT run against
the 3 deep sequence models (VLSTM/CNN-LSTM/PiFormer) — LimeTabularExplainer
is for tabular feature vectors, not the raw (200-timestep, 6-channel)
sequence input those models take; that's exactly why the task scoped
this to "XGBoost/meta-learner predictions", the two places TreeSHAP
already explains a genuinely tabular feature vector.

The cross-validation (do two independently-derived methods, TreeSHAP's
exact game-theoretic attribution vs. LIME's local-linear-surrogate
approximation, agree on which features drive a SPECIFIC prediction) is
the point of this script, not LIME's ranking in isolation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_EXPLAIN = 5   # "a handful", per instruction
TOP_K = 3
SEED = 42


def top_k_by_abs(values, feature_names, k=TOP_K):
    order = np.argsort(-np.abs(values))[:k]
    return [feature_names[i] for i in order]


def top_k_lime(exp, k=TOP_K):
    # LimeTabularExplainer.as_list() is already sorted by |weight|
    # descending; discretize_continuous=False (see explain_instances()
    # below) makes each entry a plain feature name, not a binned rule
    # string, so no string-parsing is needed to recover feature identity.
    return [name for name, _ in exp.as_list()[:k]]


def explain_instances(model_name, model, X_background, X_explain, feature_names, instance_meta):
    """
    X_background: (n_bg, d) — TRAINING-distribution rows, used only to
    give LIME's local perturbation sampler realistic feature statistics
    (mean/std per feature). TreeSHAP needs no such background (it reads
    the fitted tree structure directly), so this is LIME-only setup, not
    an extra advantage handed to either method.
    X_explain: (n_explain, d) — the SAME rows explained by both methods,
    one row per test-set prediction being cross-validated.
    """
    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(X_explain)

    lime_explainer = LimeTabularExplainer(
        X_background, feature_names=feature_names, mode="regression",
        discretize_continuous=False, random_state=SEED,
    )

    rows = []
    for i in range(len(X_explain)):
        shap_top3 = top_k_by_abs(shap_values[i], feature_names)
        exp = lime_explainer.explain_instance(
            X_explain[i], model.predict, num_features=len(feature_names)
        )
        lime_top3 = top_k_lime(exp)
        overlap = len(set(shap_top3) & set(lime_top3))
        row = {
            "model": model_name, **instance_meta[i],
            "shap_top3": "|".join(shap_top3), "lime_top3": "|".join(lime_top3),
            "overlap_count": overlap, "match_fraction": overlap / TOP_K,
        }
        rows.append(row)
        print(f"[lime] {model_name} {instance_meta[i]}: "
              f"SHAP_top3={shap_top3}  LIME_top3={lime_top3}  overlap={overlap}/{TOP_K}")
    return rows


def xgboost_base_learner_comparison():
    """Mirrors run_shap_analysis.tree_shap_xgboost's data/model exactly
    (same hi_table.parquet rows, same BFA-selected features, same
    xgb_soh.json), but restricted to held-out TEST battery rows (the
    original TreeSHAP function samples from the full NASA+MIT table, not
    test-only — this script explains genuine test-set PREDICTIONS, per
    instruction)."""
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        selected = [l.strip() for l in f if l.strip()]

    df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    df = df[df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)

    X = df[selected].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])  # same median-imputation as tree_shap_xgboost

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    test_mask = df["battery_id"].isin(split["test_ids"]).to_numpy()
    train_mask = ~test_mask
    X_train, X_test = X[train_mask], X[test_mask]
    test_meta_df = df.loc[test_mask, ["dataset", "battery_id", "cycle_idx"]].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False)
    pick = np.sort(pick)
    X_pick = X_test[pick]
    meta_pick = [test_meta_df.iloc[j].to_dict() for j in pick]

    model = XGBRegressor()
    model.load_model(str(ROOT / "models" / "xgb_soh.json"))

    print(f"\n[lime] === XGBoost base learner (7 BFA-selected HIs), "
          f"{len(X_pick)} held-out test instances ===")
    return explain_instances("XGBoost-base", model, X_train, X_pick, selected, meta_pick)


def xgboost_meta_learner_comparison():
    """Mirrors run_shap_analysis.tree_shap_meta's data/model exactly
    (same ensemble_test_preds.csv rows, same 4 base-learner-prediction
    features, same xgb_meta.json). Background distribution for LIME uses
    the TRAIN-split meta-features (train_ensemble.load_merged("train")),
    the exact same features xgb_meta.json was fit on — not the test rows
    being explained."""
    from train_ensemble import load_merged

    base_cols = ["pred_XGBoost", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
    train_df = load_merged("train")
    test_df = pd.read_csv(PRED_DIR / "ensemble_test_preds.csv")

    X_train = train_df[base_cols].to_numpy(dtype=float)
    X_test = test_df[base_cols].to_numpy(dtype=float)
    test_meta_df = test_df[["dataset", "battery_id", "cycle_idx"]].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False)
    pick = np.sort(pick)
    X_pick = X_test[pick]
    meta_pick = [test_meta_df.iloc[j].to_dict() for j in pick]

    meta = XGBRegressor()
    meta.load_model(str(ROOT / "models" / "xgb_meta.json"))

    print(f"\n[lime] === Stacking-XGBoost meta-learner (4 base-learner predictions), "
          f"{len(X_pick)} held-out test instances ===")
    return explain_instances("XGBoost-meta", meta, X_train, X_pick, base_cols, meta_pick)


def main():
    rows = []
    rows += xgboost_base_learner_comparison()
    rows += xgboost_meta_learner_comparison()

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "lime_shap_comparison.csv", index=False)

    print("\n[lime] === SHAP vs LIME top-3 agreement summary ===")
    summary = out.groupby("model")["match_fraction"].agg(["mean", "count"])
    print(summary.to_string())
    overall_mean = out["match_fraction"].mean()
    full_match_frac = (out["overlap_count"] == TOP_K).mean()
    print(f"\n[lime] overall mean top-3 overlap fraction: {overall_mean:.3f}")
    print(f"[lime] instances with FULL 3/3 top-3 agreement: {full_match_frac:.1%} "
          f"({(out['overlap_count'] == TOP_K).sum()}/{len(out)})")
    print(f"[lime] saved per-instance comparison to outputs/lime_shap_comparison.csv")
    print("[lime] DONE")


if __name__ == "__main__":
    main()
