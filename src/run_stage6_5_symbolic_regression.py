"""
Stage 6.5: symbolic regression (gplearn) against the SOH prediction
task, using the canonical feature set, to test whether a short,
interpretable closed-form equation approximates (a) the deployed
model's own behavior, and (b) true SOH directly - a formula, not just
a feature-importance attribution, as a genuinely different kind of
explainability artifact than SHAP/LIME.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols, OUT_DIR, ROOT
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols

N_TRAIN_SAMPLE = 2000
SEED = 42


def main():
    t0 = time.time()
    print("=== Stage 6.5: symbolic regression ===")
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
    print(f"[symbolic] training subsample: {len(train_df)} rows, {train_df['battery_id'].nunique()} batteries")

    X_full = train_df[feature_cols + fcols].to_numpy(dtype=float)
    X_full = np.where(np.isinf(X_full), np.nan, X_full)
    medians = np.nanmedian(X_full, axis=0)
    inds = np.where(np.isnan(X_full))
    X_full[inds] = np.take(medians, inds[1])
    model_pred = model.predict(X_full)
    true_soh = train_df["SOH"].to_numpy(dtype=float)

    # symbolic regression only on the 9 INTERPRETABLE features (not the
    # 16 opaque fusion embeddings - defeats the point of an interpretable
    # equation if half its terms are learned, uninterpretable numbers)
    X_interp = train_df[feature_cols].to_numpy(dtype=float)
    X_interp = np.where(np.isinf(X_interp), np.nan, X_interp)
    interp_medians = np.nanmedian(X_interp, axis=0)
    inds2 = np.where(np.isnan(X_interp))
    X_interp[inds2] = np.take(interp_medians, inds2[1])

    results = []
    for target_name, y in [("deployed model's own predictions", model_pred), ("true SOH", true_soh)]:
        print(f"\n[symbolic] fitting SymbolicRegressor against: {target_name}")
        sr = SymbolicRegressor(
            population_size=2000, generations=25, stopping_criteria=0.01,
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs"),
            parsimony_coefficient=0.001, max_samples=0.9, verbose=1,
            feature_names=feature_cols, random_state=SEED, n_jobs=-1,
        )
        sr.fit(X_interp, y)
        pred = sr.predict(X_interp)
        r2_vs_target = r2_score(y, pred)
        r2_vs_soh = r2_score(true_soh, pred)
        program_str = str(sr._program)
        program_len = len(program_str)
        print(f"[symbolic] discovered equation (vs. {target_name}):")
        print(f"    {program_str}")
        print(f"[symbolic] R2 vs. {target_name}: {r2_vs_target:.4f}  |  R2 vs. true SOH: {r2_vs_soh:.4f}  "
              f"|  expression length: {program_len} chars")
        results.append({"target": target_name, "equation": program_str, "r2_vs_target": r2_vs_target,
                         "r2_vs_true_soh": r2_vs_soh, "expression_length_chars": program_len})

    pd.DataFrame(results).to_csv(OUT_DIR / "stage6_5_symbolic_regression_results.csv", index=False)
    print(f"\n[symbolic] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
