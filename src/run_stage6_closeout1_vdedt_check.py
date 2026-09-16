"""
Stage 6 closeout, item 1: is the VDEDT counterfactual anomaly (6.6) a
DiCE search-space bug, or a genuine, pre-existing instability in one
of the model's own trained features? Checked directly rather than
assumed either way.
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
DESIRED_SHIFT = -15.0


def main():
    t0 = time.time()
    print("=== Stage 6 closeout, item 1: VDEDT counterfactual check ===")
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

    # === 1.1/1.2: DiCE's own bounding behavior + VDEDT's real observed range ===
    train_df = merged.loc[train_mask].copy()
    vdedt = train_df["VDEDT"].to_numpy(dtype=float)
    vdedt_finite = vdedt[np.isfinite(vdedt)]
    print(f"\n[closeout1] VDEDT training-data range: min={vdedt_finite.min():.6g}, max={vdedt_finite.max():.6g}")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    p{p}: {np.percentile(vdedt_finite, p):.6g}")
    n_extreme = int((np.abs(vdedt_finite) > 1000).sum())
    print(f"    {n_extreme} of {len(vdedt_finite)} rows have |VDEDT| > 1000 "
          f"({n_extreme/len(vdedt_finite)*100:.4f}% of the training pool)")
    extreme_rows = train_df[np.abs(train_df["VDEDT"].astype(float)) > 1000]
    print(f"    extreme rows: {extreme_rows[['dataset','battery_id','cycle_idx','VDEDT']].to_dict('records')}")
    print("\n[closeout1] DiCE's own default bounding behavior (from its own source, "
          "PublicData.get_features_range): unless an explicit permitted_range is given, "
          "continuous-feature bounds = [training_data.min(), training_data.max()] - i.e. "
          "6.6's run WAS already bounded to the observed training range by default. The "
          "~21 million counterfactual values are NOT a DiCE search-space bug - they fall "
          "inside VDEDT's own genuine training-data range (up to 3.87e7), which itself "
          "contains this handful of extreme outlier rows.")

    X_train = train_df[all_cols].to_numpy(dtype=float)
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    medians = np.nanmedian(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(medians, inds[1])
    train_df_clean = train_df.copy()
    train_df_clean[all_cols] = X_train
    train_df_clean["SOH"] = train_df["SOH"].to_numpy(dtype=float)

    model = XGBRegressor()
    model.load_model(str(Path(ROOT) / "models" / "_experimental_xgb_soh_fusion_extended_reformulation.json"))

    def predict_fn(X_partial: pd.DataFrame) -> np.ndarray:
        full = np.tile(np.nanmedian(X_train[:, len(feature_cols):], axis=0), (len(X_partial), 1))
        X_full = np.concatenate([X_partial[feature_cols].to_numpy(dtype=float), full], axis=1)
        return model.predict(X_full)

    class _WrappedModel:
        def predict(self, X):
            return predict_fn(pd.DataFrame(X, columns=feature_cols))

    # === 1.3: re-run with VDEDT explicitly bounded to a SANE range
    # (1st-99th percentile, excluding the 3 known outlier rows) - does
    # a sensible, bounded counterfactual still exist? ===
    vdedt_sane_lo, vdedt_sane_hi = float(np.percentile(vdedt_finite, 1)), float(np.percentile(vdedt_finite, 99))
    print(f"\n[closeout1] re-running with VDEDT explicitly bounded to its 1st-99th percentile "
          f"range [{vdedt_sane_lo:.6g}, {vdedt_sane_hi:.6g}] (excludes the 3 known outlier rows) "
          f"- does a sensible counterfactual still exist?")

    dice_data_bounded = dice_ml.Data(
        dataframe=train_df_clean[feature_cols + ["SOH"]], continuous_features=feature_cols,
        outcome_name="SOH", permitted_range={"VDEDT": [vdedt_sane_lo, vdedt_sane_hi]})
    dice_model = dice_ml.Model(model=_WrappedModel(), backend="sklearn", model_type="regressor")
    exp_bounded = dice_ml.Dice(dice_data_bounded, dice_model, method="random")

    test_df = merged.loc[test_mask].copy()
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(test_df), size=min(5, len(test_df)), replace=False)
    query_cases = test_df.iloc[sample_idx]

    results = []
    for _, row in query_cases.iterrows():
        query = row[feature_cols].to_frame().T.astype(float)
        current_pred = float(predict_fn(query)[0])
        target = current_pred + DESIRED_SHIFT
        print(f"\n[closeout1] battery={row['battery_id']} cycle={row['cycle_idx']}: "
              f"current predicted SOH={current_pred:.2f}, seeking a bounded counterfactual near {target:.2f}")
        try:
            cf = exp_bounded.generate_counterfactuals(query, total_CFs=3, desired_range=[target - 3, target + 3])
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                print("    NO bounded counterfactual found within the search budget")
                results.append({"battery_id": row["battery_id"], "cycle_idx": row["cycle_idx"],
                                 "current_pred": current_pred, "target": target, "found": False})
                continue
            for i, cf_row in cf_df.iterrows():
                changed = {c: (float(query[c].iloc[0]), float(cf_row[c])) for c in feature_cols
                           if abs(float(cf_row[c]) - float(query[c].iloc[0])) > 1e-6}
                cf_pred = float(predict_fn(cf_row[feature_cols].to_frame().T.astype(float))[0])
                vdedt_change = changed.get("VDEDT")
                print(f"    CF {i}: predicted SOH={cf_pred:.2f}, VDEDT change: {vdedt_change}, "
                      f"all changed: {changed}")
                results.append({"battery_id": row["battery_id"], "cycle_idx": row["cycle_idx"],
                                 "current_pred": current_pred, "target": target, "found": True,
                                 "cf_pred": cf_pred, "vdedt_changed": vdedt_change is not None,
                                 "n_features_changed": len(changed), "changed_features": str(changed)})
        except Exception as e:
            print(f"    DiCE generation failed: {e}")
            results.append({"battery_id": row["battery_id"], "cycle_idx": row["cycle_idx"],
                             "current_pred": current_pred, "target": target, "found": False, "error": str(e)})

    pd.DataFrame(results).to_csv(OUT_DIR / "stage6_closeout1_vdedt_bounded_cf_results.csv", index=False)
    print(f"\n[closeout1] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
