"""
Research pass Group B, item 6: symbolic regression retry, three
combined fixes over Stage 6.5's original catastrophically-degenerate
result (equations 1400-1868 characters long, R2=-1.92/-1.95 against
both the deployed model's own predictions and true SOH - worse than
predicting the mean, and unreadable).

(a) SISSO-INSPIRED, disclosed substitution: genuine SISSO (Sure
Independence Screening + Sparsifying Operator) needs the original
Fortran/MPI SISSO binary via its Python wrapper (pysisso) - confirmed
NOT installable in this environment (its build chain is broken here,
tries to rebuild pandas from source under a missing pkg_resources).
Reimplements the CORE algorithmic idea instead, disclosed as an
approximation not a port: (1) SIS - build a large pool of candidate
features from simple, named univariate transforms of the base HI
features (no unconstrained recursive composition, unlike gplearn's own
search), rank by |correlation| with the target, keep the top-K; (2) SO
- fit a SPARSE linear model (Lasso) on the screened pool, which by
construction produces a SHORT formula (a handful of named terms with
real coefficients), not an uncontrolled nested tree.

(b) CONSTRAINED, DOMAIN-APPROPRIATE OPERATORS: the candidate pool is
{x, x^2, sqrt(|x|), log(|x|+eps), 1/(x+eps)} applied to each canonical
HI feature UNIVARIATELY, plus cycle_idx cross-terms with the 2 features
this project's own SHAP analysis has repeatedly found dominant
(SCV_rel, VIECT_rel) - motivated by battery literature's own standard
capacity-fade shapes (power-law/exponential decay in cycle count), not
gplearn's generic +/-/*// composition search.

(c) GREY-BOX RESIDUAL FRAMING: the sparse Lasso formula is reported on
its OWN merits first (does symbolic regression alone explain the smooth
global trend?), then a small residual model (shallow decision tree) is
fit on (true_SOH - formula_prediction) using the full feature set, to
see how much of the REMAINING nonlinearity a small residual step can
recover - reported as a second, separate number, not conflated with
the formula's own standalone result.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols, OUT_DIR, ROOT
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols

SEED = 42
N_TRAIN_SAMPLE = 2000
TOP_K_SCREEN = 15


def build_candidate_pool(df: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    eps = 1e-6
    candidates = {}
    for feat in base_features:
        x = df[feat].to_numpy(dtype=float)
        candidates[feat] = x
        candidates[f"{feat}^2"] = x ** 2
        candidates[f"sqrt|{feat}|"] = np.sqrt(np.abs(x))
        candidates[f"log|{feat}|"] = np.log(np.abs(x) + eps)
        candidates[f"1/({feat})"] = 1.0 / (x + np.sign(x + eps) * eps)
    if "cycle_idx" in df.columns:
        c = df["cycle_idx"].to_numpy(dtype=float)
        for feat in ["SCV_rel", "VIECT_rel"]:
            if feat in df.columns:
                x = df[feat].to_numpy(dtype=float)
                candidates[f"cycle_idx*{feat}"] = c * x
                candidates[f"cycle_idx/({feat})"] = c / (x + np.sign(x + eps) * eps)
    return pd.DataFrame(candidates, index=df.index)


def main():
    t0 = time.time()
    print("=== Research pass Group B, item 6: symbolic regression retry (SISSO-inspired) ===")
    merged_nm, hi_full = load_nasa_mit_pool(reformulated=True)
    hi_full_ext = add_scv_matd_viect_reformulated(hi_full)
    fusion_df = pd.read_csv(Path(ROOT) / "data" / "processed" / "fusion_embeddings.csv")
    nasa_mit_ext = hi_full_ext[hi_full_ext["dataset"].isin(["NASA", "MIT"])]
    merged = pd.merge(nasa_mit_ext, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    train_mask, test_mask, split = battery_split_masks(merged)

    base_cols = canonical_feature_cols(reformulated=True)
    extended_cols = extended_canonical_feature_cols(base_cols)
    feature_cols = extended_cols + ["cycle_idx"]
    fcols = fusion_cols()

    model = XGBRegressor()
    model.load_model(str(Path(ROOT) / "models" / "_experimental_xgb_soh_fusion_extended_reformulation.json"))

    train_df = merged.loc[train_mask].copy()
    rng = np.random.default_rng(SEED)
    if len(train_df) > N_TRAIN_SAMPLE:
        train_df = train_df.sample(n=N_TRAIN_SAMPLE, random_state=SEED)
    test_df = merged.loc[test_mask].copy()
    print(f"[symbolic-v2] train subsample: {len(train_df)} rows, test: {len(test_df)} rows")

    X_full_train = train_df[feature_cols + fcols].to_numpy(dtype=float)
    X_full_train = np.where(np.isinf(X_full_train), np.nan, X_full_train)
    medians = np.nanmedian(X_full_train, axis=0)
    inds = np.where(np.isnan(X_full_train))
    X_full_train[inds] = np.take(medians, inds[1])
    model_pred_train = model.predict(X_full_train)
    true_soh_train = train_df["SOH"].to_numpy(dtype=float)

    X_full_test = test_df[feature_cols + fcols].to_numpy(dtype=float)
    X_full_test = np.where(np.isinf(X_full_test), np.nan, X_full_test)
    inds_t = np.where(np.isnan(X_full_test))
    X_full_test[inds_t] = np.take(medians, inds_t[1])
    true_soh_test = test_df["SOH"].to_numpy(dtype=float)

    # Impute base-feature NaNs on the interpretable subset for the candidate pool
    base_feat_df_train = train_df[feature_cols].copy()
    base_feat_df_test = test_df[feature_cols].copy()
    for i, c in enumerate(feature_cols):
        base_feat_df_train[c] = base_feat_df_train[c].fillna(medians[i])
        base_feat_df_test[c] = base_feat_df_test[c].fillna(medians[i])

    for target_name, y_train, y_test in [
        ("true SOH", true_soh_train, true_soh_test),
        ("deployed model's own predictions", model_pred_train, model.predict(X_full_test)),
    ]:
        print(f"\n--- target: {target_name} ---")
        pool_train = build_candidate_pool(base_feat_df_train, feature_cols)
        pool_test = build_candidate_pool(base_feat_df_test, feature_cols)
        pool_train = pool_train.replace([np.inf, -np.inf], np.nan).fillna(pool_train.median(numeric_only=True))
        pool_test = pool_test.replace([np.inf, -np.inf], np.nan).fillna(pool_train.median(numeric_only=True))

        # SIS: rank candidates by |correlation| with target, keep top-K
        corrs = pool_train.apply(lambda col: abs(np.corrcoef(col, y_train)[0, 1]) if col.std() > 1e-9 else 0.0)
        top_features = corrs.sort_values(ascending=False).head(TOP_K_SCREEN).index.tolist()
        print(f"[symbolic-v2] SIS: top-{TOP_K_SCREEN} screened candidates (by |corr|): {top_features[:6]}...")

        # SO: sparse Lasso on the screened pool
        X_screen_train = pool_train[top_features].to_numpy()
        X_screen_test = pool_test[top_features].to_numpy()
        lasso = LassoCV(cv=5, random_state=SEED, max_iter=10000).fit(X_screen_train, y_train)
        nonzero = [(f, c) for f, c in zip(top_features, lasso.coef_) if abs(c) > 1e-8]
        formula_terms = " + ".join(f"{c:+.4f}*{f}" for f, c in nonzero)
        formula = f"{lasso.intercept_:.4f} {formula_terms}"
        print(f"[symbolic-v2] Sparsifying Operator (Lasso) formula ({len(nonzero)} non-zero terms):")
        print(f"    {formula}")

        pred_train = lasso.predict(X_screen_train)
        pred_test = lasso.predict(X_screen_test)
        r2_formula_train = r2_score(y_train, pred_train)
        r2_formula_test = r2_score(y_test, pred_test)
        print(f"[symbolic-v2] Formula-ONLY R2: train={r2_formula_train:.4f} test={r2_formula_test:.4f} "
              f"(Stage 6.5 baseline was R2=-1.92/-1.95, degenerate)")

        # Grey-box residual: small decision tree on the remaining nonlinearity
        residual_train = y_train - pred_train
        tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=SEED)
        tree.fit(base_feat_df_train.to_numpy(), residual_train)
        residual_pred_test = tree.predict(base_feat_df_test.to_numpy())
        combined_pred_test = pred_test + residual_pred_test
        r2_combined_test = r2_score(y_test, combined_pred_test)
        print(f"[symbolic-v2] Formula + small residual-tree (grey-box) R2 on test: {r2_combined_test:.4f}")

        yield_row = {
            "target": target_name, "n_nonzero_terms": len(nonzero), "formula": formula,
            "formula_only_r2_train": r2_formula_train, "formula_only_r2_test": r2_formula_test,
            "grey_box_r2_test": r2_combined_test, "stage6_5_baseline_r2": -1.92 if target_name == "deployed model's own predictions" else -1.95,
        }
        globals().setdefault("_results", []).append(yield_row)

    results_df = pd.DataFrame(globals()["_results"])
    results_df.to_csv(OUT_DIR / "researchpass_groupB6_symbolic_regression_v2.csv", index=False)
    print("\n=== SUMMARY ===")
    print(results_df[["target", "n_nonzero_terms", "formula_only_r2_test", "grey_box_r2_test", "stage6_5_baseline_r2"]].to_string(index=False))
    for r in globals()["_results"]:
        verdict = "genuinely non-degenerate, positive R2" if r["formula_only_r2_test"] > 0 else "still degenerate (R2<=0)"
        print(f"\n[symbolic-v2] {r['target']}: formula-only test R2={r['formula_only_r2_test']:.4f} -> {verdict}")

    print(f"\n[symbolic-v2] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
