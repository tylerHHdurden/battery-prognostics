"""
Stage 2, Item 2.4 (parts 1-2): re-run Phase 5's SHAP top-feature
validation and session 15's LIME/SHAP agreement check against Stage
1's current canonical feature set (1.1's reformulated features + 1.5's
monotone_constraints on cycle_idx), scoped to the XGBoost BASE learner
specifically - the piece Stage 1 actually changed. The meta-learner
(4 base-learner predictions -> Ridge/XGBoost-meta) is untouched by
Stage 1 and out of this item's scope; its original 86.7% LIME/SHAP
figure stands as-is, not re-run here.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, fusion_cols, OUT_DIR,
)

N_EXPLAIN = 5
TOP_K = 3
SEED = 42


def top_k_by_abs(values, feature_names, k=TOP_K):
    order = np.argsort(-np.abs(values))[:k]
    return [feature_names[i] for i in order]


def main():
    print("[revalidate] === building the Stage 1.1+1.5 canonical XGBoost-fusion model ===")
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    base_features = canonical_feature_cols(reformulated=True)
    feature_cols = base_features + ["cycle_idx"]
    n_fusion = len(fusion_cols())
    monotone = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)
    model, medians, cols = fit_xgb(merged, train_mask, feature_cols, xgb_extra_kwargs={"monotone_constraints": monotone})
    print(f"[revalidate] feature order: {cols}")

    X_all = merged[cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X_all))
    X_all[inds] = np.take(medians, inds[1])
    X_train, X_test = X_all[train_mask], X_all[test_mask]

    # --- Part 1: TreeSHAP top-feature ranking, current canonical model ---
    print("\n[revalidate] === Part 1: TreeSHAP ranking, Stage 1.1+1.5 model (HI features + cycle_idx only, "
          "excluding the 16 fusion-embedding dims - matching Phase 5's original HI-only ranking scope) ===")
    explainer = shap.TreeExplainer(model)
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(X_all), size=min(2000, len(X_all)), replace=False)
    shap_values = explainer.shap_values(X_all[sample_idx])
    hi_idx = [cols.index(c) for c in feature_cols]  # HI features + cycle_idx, excluding fusion dims
    mean_abs = np.abs(shap_values[:, hi_idx]).mean(axis=0)
    ranking = sorted(zip(feature_cols, mean_abs), key=lambda t: -t[1])
    print("[revalidate] NEW TreeSHAP ranking (canonical, reformulated feature set):")
    for name, val in ranking:
        print(f"    {name}: {val:.4f}")

    original = pd.read_csv("outputs/shap_xgboost_base_ranking.csv")
    print("\n[revalidate] ORIGINAL Phase 5 ranking (leaked 7-feature set), for direct comparison:")
    print(original.to_string(index=False))

    new_top3 = [n for n, _ in ranking[:3]]
    orig_top3 = original["feature"].tolist()[:3]
    print(f"\n[revalidate] NEW top-3: {new_top3}")
    print(f"[revalidate] ORIGINAL top-3: {orig_top3}")
    # map reformulated names back to their raw equivalents for a fair comparison
    def base_name(f):
        return f[:-4] if f.endswith("_rel") else f
    new_top3_base = [base_name(f) for f in new_top3]
    overlap = len(set(new_top3_base) & set(orig_top3))
    print(f"[revalidate] top-3 overlap (mapping _rel features back to their raw name): {overlap}/3")

    pd.DataFrame(ranking, columns=["feature", "mean_abs_shap"]).to_csv(
        OUT_DIR / "stage2_shap_canonical_ranking.csv", index=False)

    # --- Part 2: LIME vs SHAP agreement, current canonical model, HI-only feature space ---
    print("\n[revalidate] === Part 2: LIME vs SHAP top-3 agreement, Stage 1.1+1.5 model ===")
    X_train_hi = X_train[:, hi_idx]
    X_test_hi = X_test[:, hi_idx]
    test_meta = merged.loc[test_mask, ["dataset", "battery_id", "cycle_idx"]].reset_index(drop=True)

    pick = np.sort(rng.choice(len(X_test_hi), size=min(N_EXPLAIN, len(X_test_hi)), replace=False))

    def predict_hi_only(X_hi_rows):
        full = np.tile(medians, (len(X_hi_rows), 1))
        full[:, :len(hi_idx)] = X_hi_rows  # hi_idx are the first len(feature_cols) columns of `cols` by construction
        return model.predict(full)

    tree_explainer_hi = shap.TreeExplainer(model)
    lime_explainer = LimeTabularExplainer(X_train_hi, feature_names=feature_cols, mode="regression",
                                           discretize_continuous=False, random_state=SEED)

    rows = []
    for j in pick:
        x_full = X_test[j]
        shap_vals_full = tree_explainer_hi.shap_values(x_full.reshape(1, -1))[0]
        shap_top3 = top_k_by_abs(shap_vals_full[hi_idx], feature_cols)

        exp = lime_explainer.explain_instance(X_test_hi[j], predict_hi_only, num_features=len(feature_cols))
        lime_top3 = [name for name, _ in exp.as_list()[:TOP_K]]

        overlap = len(set(shap_top3) & set(lime_top3))
        meta = test_meta.iloc[j].to_dict()
        print(f"[revalidate] {meta}: SHAP_top3={shap_top3}  LIME_top3={lime_top3}  overlap={overlap}/{TOP_K}")
        rows.append({**meta, "shap_top3": "|".join(shap_top3), "lime_top3": "|".join(lime_top3),
                    "overlap_count": overlap, "match_fraction": overlap / TOP_K})

    agree_df = pd.DataFrame(rows)
    new_agreement = agree_df["match_fraction"].mean()
    n_full_match = (agree_df["overlap_count"] == TOP_K).sum()
    print(f"\n[revalidate] NEW mean top-3 overlap (XGBoost-base, canonical model): {new_agreement*100:.1f}% "
          f"({n_full_match}/{len(agree_df)} instances at full 3/3)")
    print(f"[revalidate] ORIGINAL (session 15, leaked feature set): base learner=100%, meta=86.7%, overall mean=93.3%")

    agree_df.to_csv(OUT_DIR / "stage2_lime_shap_agreement_canonical.csv", index=False)
    print("\n[revalidate] saved outputs/stage2_{shap_canonical_ranking,lime_shap_agreement_canonical}.csv")
    print("[revalidate] DONE")


if __name__ == "__main__":
    main()
