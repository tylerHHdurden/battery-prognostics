"""
Final research pass, item 2: cycle_idx ablation.

Retrains the deployed XGBoost-fusion setup (Stage 1.5's own recipe -
canonical reformulated 8 HIs + cycle_idx + 16-dim fusion, monotone_
constraints=-1 on cycle_idx - the exact feature/hyperparameter pipeline
`run_audit_true_deployed_baseline.py` confirmed matches models/xgb_soh_
fusion.json) three ways, using stage1_common.fit_xgb unchanged so all
three differ ONLY in the one thing each variant changes:

  (a) as-is: canonical 8 HI (reformulated) + cycle_idx + fusion,
      monotone_constraints=-1 on cycle_idx. Reproduces the deployed
      model's own recipe exactly (same code path as
      run_stage1_5_monotone_constraints.py / the audit script).
  (b) without cycle_idx: canonical 8 HI (reformulated) + fusion only,
      no monotone_constraints (nothing left to constrain).
  (c) cycle_idx replaced by "equivalent full cycles" = cumulative
      discharge_capacity throughput / that battery's own nominal
      capacity (median of its first 3 cycles - same convention
      rul_labels.py uses for initial capacity), same monotone_
      constraints=-1 (equivalent full cycles rises monotonically as
      SOH falls, same sign as cycle_idx).

Evaluated fresh on in-domain TEST + all 13 held-out sets (CALCE/
Oxford/HUST/XJTU + 9 locally-available BatteryLife sources), zero-
retrain, same train-pool-median imputation convention as every other
item in this pass. Reports whether cycle_idx is acting as a dataset-
specific shortcut: if (a) beats (b)/(c) in-domain but loses badly
zero-retrain relative to them, that is exactly a shortcut signature
(a feature helping only where it was trained, hurting elsewhere).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, build_calce_merged, eval_calce,
    fusion_cols, OUT_DIR, PROC_DIR,
)

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]


def add_equivalent_full_cycles(hi_df: pd.DataFrame) -> pd.DataFrame:
    """EFC = cumulative discharge_capacity (through and including this
    cycle) / this battery's own nominal capacity (median of its first 3
    cycles' discharge_capacity - same "initial capacity" convention
    rul_labels.compute_eol_and_rul uses, so EFC and SOH share the same
    normalization philosophy). Protocol-invariant like the project's own
    _rel duration features: dividing by the battery's OWN nominal
    capacity removes any absolute-Ah-scale difference between chemistries/
    formats (e.g. NASA's ~2Ah cells vs. XJTU's larger-format cells)."""
    hi_df = hi_df.copy()
    efc = np.full(len(hi_df), np.nan, dtype=float)
    n_degenerate = 0
    for bid, g in hi_df.groupby("battery_id"):
        g_sorted = g.sort_values("cycle_idx")
        cap = g_sorted["discharge_capacity"].to_numpy(dtype=float)
        cap_clean = np.where(np.isfinite(cap), cap, 0.0)
        nominal = float(np.median(cap_clean[:3])) if len(cap_clean) >= 1 else np.nan
        if not np.isfinite(nominal) or abs(nominal) < 1e-6:
            nominal = float(np.median(cap_clean[cap_clean > 0])) if (cap_clean > 0).any() else np.nan
            n_degenerate += 1
        if not np.isfinite(nominal) or abs(nominal) < 1e-6:
            continue  # leave NaN, imputed later by train medians like every other feature
        cum = np.cumsum(cap_clean)
        efc[g_sorted.index] = cum / nominal
    print(f"[item2] equivalent_full_cycles: {n_degenerate} batteries needed a fallback nominal-capacity "
          f"(near-zero/non-finite first-3-cycle median), {int(np.isnan(efc).sum())} rows left NaN (of {len(hi_df)})")
    hi_df["equivalent_full_cycles"] = efc
    return hi_df


def fit_medians_generic(df: pd.DataFrame, cols: list[str], mask: np.ndarray = None) -> np.ndarray:
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    if mask is not None:
        X = X[mask]
    return np.nanmedian(X, axis=0)


def build_X_generic(df: pd.DataFrame, cols: list[str], medians: np.ndarray) -> np.ndarray:
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    return X


def r2_rmse_mae(y_true, pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return {
        "r2": float(r2_score(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
    }


def main():
    t0 = time.time()
    print("=== Final pass item 2: cycle_idx ablation (as-is / dropped / equivalent-full-cycles) ===")

    base_features = canonical_feature_cols(reformulated=True)
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    merged = add_equivalent_full_cycles(merged)
    hi_full = add_equivalent_full_cycles(hi_full)
    train_mask, test_mask, split = battery_split_masks(merged)

    n_fusion = len(fusion_cols())

    variants = {}

    # (a) as-is: cycle_idx + monotone_constraints=-1
    feat_a = base_features + ["cycle_idx"]
    mono_a = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)
    model_a, med_a, cols_a = fit_xgb(merged, train_mask, feat_a, xgb_extra_kwargs={"monotone_constraints": mono_a})
    variants["a_as_is_cycle_idx"] = (model_a, med_a, cols_a, feat_a)

    # (b) without cycle_idx: no monotone constraint
    feat_b = base_features
    model_b, med_b, cols_b = fit_xgb(merged, train_mask, feat_b)
    variants["b_no_cycle_idx"] = (model_b, med_b, cols_b, feat_b)

    # (c) equivalent full cycles instead of cycle_idx, same monotone constraint
    feat_c = base_features + ["equivalent_full_cycles"]
    mono_c = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)
    model_c, med_c, cols_c = fit_xgb(merged, train_mask, feat_c, xgb_extra_kwargs={"monotone_constraints": mono_c})
    variants["c_equivalent_full_cycles"] = (model_c, med_c, cols_c, feat_c)

    print(f"[item2] (a) feature cols: {cols_a}")
    print(f"[item2] (b) feature cols: {cols_b}")
    print(f"[item2] (c) feature cols: {cols_c}")

    calce_merged = build_calce_merged(hi_full)
    calce_merged = add_equivalent_full_cycles(calce_merged) if "equivalent_full_cycles" not in calce_merged.columns else calce_merged

    heldout_frames = {
        "Oxford": pd.read_parquet(PROC_DIR / "stage5_1_oxford_merged.parquet"),
        "HUST": pd.read_parquet(PROC_DIR / "stage5_1_hust_merged.parquet"),
        "XJTU": pd.read_parquet(PROC_DIR / "stage5_1_xjtu_merged.parquet"),
    }
    for name in heldout_frames:
        heldout_frames[name] = add_equivalent_full_cycles(heldout_frames[name])
    heldout_frames["CALCE"] = calce_merged

    for source in BATTERYLIFE_SOURCES:
        path = PROC_DIR / f"batterylife_{source}_merged.parquet"
        if not path.exists():
            print(f"[item2] {source}: no local merged table - skipped")
            continue
        df = pd.read_parquet(path)
        heldout_frames[source] = add_equivalent_full_cycles(df)

    rows = []
    for variant_name, (model, medians, cols, feat) in variants.items():
        indomain = eval_indomain(model, medians, cols, merged, test_mask)
        print(f"[item2] {variant_name} IN-DOMAIN: R2={indomain['r2']:.4f} RMSE={indomain['rmse']:.4f} MAE={indomain['mae']:.4f}")
        rows.append({"variant": variant_name, "eval_set": "in-domain (TEST)",
                     "r2": indomain["r2"], "rmse": indomain["rmse"], "mae": indomain["mae"],
                     "n": int(test_mask.sum())})

        for name, df in heldout_frames.items():
            X = build_X_generic(df, cols, medians)
            pred = model.predict(X)
            m = r2_rmse_mae(df["SOH"].to_numpy(), pred)
            print(f"[item2] {variant_name} {name}: R2={m['r2']:.4f} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} n={len(df)}")
            rows.append({"variant": variant_name, "eval_set": name, "r2": m["r2"], "rmse": m["rmse"],
                         "mae": m["mae"], "n": len(df)})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "finalpass_item2_cycleidx_ablation.csv", index=False)

    print("\n=== SUMMARY: R2 by variant x eval_set ===")
    pivot = results_df.pivot(index="eval_set", columns="variant", values="r2")
    pivot = pivot[["a_as_is_cycle_idx", "b_no_cycle_idx", "c_equivalent_full_cycles"]]
    print(pivot.to_string())
    pivot.to_csv(OUT_DIR / "finalpass_item2_r2_pivot.csv")

    # shortcut diagnosis: in-domain gap vs. mean zero-retrain gap, (a) vs (b)
    indomain_gap = pivot.loc["in-domain (TEST)", "a_as_is_cycle_idx"] - pivot.loc["in-domain (TEST)", "b_no_cycle_idx"]
    zero_retrain_sets = [s for s in pivot.index if s != "in-domain (TEST)"]
    mean_zr_gap = (pivot.loc[zero_retrain_sets, "a_as_is_cycle_idx"] - pivot.loc[zero_retrain_sets, "b_no_cycle_idx"]).mean()
    n_zr_worse = int((pivot.loc[zero_retrain_sets, "a_as_is_cycle_idx"] < pivot.loc[zero_retrain_sets, "b_no_cycle_idx"]).sum())
    print(f"\n[item2] cycle_idx's in-domain R2 contribution (a - b): {indomain_gap:+.4f}")
    print(f"[item2] cycle_idx's MEAN zero-retrain R2 contribution across {len(zero_retrain_sets)} held-out sets (a - b): {mean_zr_gap:+.4f}")
    print(f"[item2] cycle_idx makes zero-retrain WORSE on {n_zr_worse}/{len(zero_retrain_sets)} held-out sets")
    if indomain_gap > 0.005 and mean_zr_gap < -0.005:
        verdict = "SHORTCUT SIGNATURE: cycle_idx clearly helps in-domain but clearly hurts, on average, zero-retrain - consistent with it acting as a dataset-specific memorized index rather than a genuinely transferable health indicator."
    elif indomain_gap > 0.005 and mean_zr_gap >= -0.005:
        verdict = "NOT a shortcut by this test: cycle_idx helps in-domain without a corresponding zero-retrain penalty."
    else:
        verdict = "cycle_idx provides negligible in-domain benefit either way - inconclusive on the shortcut question."
    print(f"[item2] VERDICT: {verdict}")
    print(f"\n[item2] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
