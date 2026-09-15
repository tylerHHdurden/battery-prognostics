"""
Stage 6.6: counterfactual explanations (DiCE) for the SOH prediction
task - a genuinely different third lens beyond SHAP/LIME: not "what
mattered" (attribution) but "what minimal feature change would flip
this prediction meaningfully" (a concrete, actionable alternative
scenario). Uses the same deployed (canonical, extended-reformulation)
XGBoost-fusion model as everywhere else in this stage.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import dice_ml
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols, OUT_DIR, ROOT
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols

SEED = 42
DESIRED_SHIFT = -15.0  # target: a meaningful SOH drop (an "early-retirement" counterfactual)


def main():
    t0 = time.time()
    print("=== Stage 6.6: counterfactual explanations (DiCE) ===")
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
    all_cols = feature_cols + fcols

    model = XGBRegressor()
    model.load_model(str(Path(ROOT) / "models" / "_experimental_xgb_soh_fusion_extended_reformulation.json"))

    train_df = merged.loc[train_mask].copy()
    X_train = train_df[all_cols].to_numpy(dtype=float)
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    medians = np.nanmedian(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(medians, inds[1])
    train_df_clean = train_df.copy()
    train_df_clean[all_cols] = X_train
    train_df_clean["SOH"] = train_df["SOH"].to_numpy(dtype=float)

    # DiCE only varies the 8 INTERPRETABLE HI features + cycle_idx -
    # not the 16 opaque fusion embeddings, which have no physical
    # meaning to "change" in a counterfactual sense.
    dice_data = dice_ml.Data(dataframe=train_df_clean[feature_cols + ["SOH"]],
                              continuous_features=feature_cols, outcome_name="SOH")

    def predict_fn(X_partial: pd.DataFrame) -> np.ndarray:
        full = np.tile(np.nanmedian(X_train[:, len(feature_cols):], axis=0), (len(X_partial), 1))
        X_full = np.concatenate([X_partial[feature_cols].to_numpy(dtype=float), full], axis=1)
        return model.predict(X_full)

    # dice_ml needs an sklearn-like object with .predict; wrap the XGBoost
    # model + fixed fusion-embedding medians in a tiny adapter class.
    class _WrappedModel:
        def predict(self, X):
            return predict_fn(pd.DataFrame(X, columns=feature_cols))
    dice_model = dice_ml.Model(model=_WrappedModel(), backend="sklearn", model_type="regressor")

    exp = dice_ml.Dice(dice_data, dice_model, method="random")

    test_df = merged.loc[test_mask].copy()
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(test_df), size=min(5, len(test_df)), replace=False)
    query_cases = test_df.iloc[sample_idx]

    results = []
    for _, row in query_cases.iterrows():
        query = row[feature_cols].to_frame().T.astype(float)
        current_pred = float(predict_fn(query)[0])
        target = current_pred + DESIRED_SHIFT
        print(f"\n[dice] battery={row['battery_id']} cycle={row['cycle_idx']}: "
              f"current predicted SOH={current_pred:.2f}, seeking a counterfactual near {target:.2f}")
        try:
            cf = exp.generate_counterfactuals(query, total_CFs=3, desired_range=[target - 3, target + 3])
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                print("    NO counterfactual found within the search budget")
                results.append({"battery_id": row["battery_id"], "cycle_idx": row["cycle_idx"],
                                 "current_pred": current_pred, "target": target, "found": False})
                continue
            for i, cf_row in cf_df.iterrows():
                changed = {c: (float(query[c].iloc[0]), float(cf_row[c])) for c in feature_cols
                           if abs(float(cf_row[c]) - float(query[c].iloc[0])) > 1e-6}
                cf_pred = float(predict_fn(cf_row[feature_cols].to_frame().T.astype(float))[0])
                print(f"    CF {i}: predicted SOH={cf_pred:.2f}, changed features: {changed}")
                results.append({"battery_id": row["battery_id"], "cycle_idx": row["cycle_idx"],
                                 "current_pred": current_pred, "target": target, "found": True,
                                 "cf_pred": cf_pred, "n_features_changed": len(changed),
                                 "changed_features": str(changed)})
        except Exception as e:
            print(f"    DiCE generation failed: {e}")
            results.append({"battery_id": row["battery_id"], "cycle_idx": row["cycle_idx"],
                             "current_pred": current_pred, "target": target, "found": False, "error": str(e)})

    pd.DataFrame(results).to_csv(OUT_DIR / "stage6_6_counterfactuals_results.csv", index=False)
    print(f"\n[dice] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
