"""
Final experiment pass 2, item C: complete baseline table on all 13
external datasets.

Columns:
  (i)   trivial baseline - a single linear SOH ~ cycle_idx model fit on
        the source pool (NASA+MIT train split - one global model, one
        feature, zero-retrain evaluated on each target exactly like
        every other model in this project).
  (ii)  Severson variance-model / Attia-style rich-feature baselines,
        EXACT SAME code (Stage 6.1's own `fit_eval`/`build_severson_rows`,
        imported not duplicated) as this project's existing Stage 6.1
        comparison, extended here from its original 4 datasets
        (CALCE/Oxford/HUST/XJTU) to all 9 additionally-available
        BatteryLife sources.
  (iii) deployed base model (already-established zero-retrain numbers,
        this pass's own item 5a 5-seed means - not recomputed).
  (iv)  routed model, explicitly labeled ORACLE-SELECTED (item 1's own
        framing: the live routing list was chosen using held-out R2,
        i.e. it "knows" which model wins per dataset - a label-informed
        selection, not a deployable zero-retrain-only decision).
  (v)   best LODO from item B (whichever of that item's own LODO R2
        beats the deployed base's zero-retrain number, reused not
        recomputed).

Paired battery-bootstrap and Wilcoxon signed-rank tests of the
deployed base model's per-battery errors against each baseline's own
per-battery errors (same battery set, same metric - absolute error -
paired by battery_id).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import load_nasa_mit_pool, battery_split_masks, OUT_DIR, PROC_DIR, ROOT
from run_stage6_1_severson_attia_baselines import (
    build_dataset_severson_df, fit_eval, VARIANCE_FEATS, RICH_FEATS,
    CALCE_CELLS, build_42pool_severson_df,
)
from run_stage5_1_new_datasets_eval import (
    oxford_cell_ids, iterate_oxford_cycles, hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids_soh_valid, iterate_xjtu_cycles,
)
from data_adapters import iterate_calce_cycles
from data_adapters_batterylife import batterylife_cell_ids, iterate_batterylife_cycles
from rul_labels import soh_per_cycle
from researchpass_partA_common import (
    load_all_heldout_base, load_all_heldout_extended, base_feature_cols,
    extended_feature_cols, fit_medians, load_base_model, load_extended_model, build_X,
)

BATTERYLIFE_LOWER = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth", "stanford", "stanford_2", "isu_ilcc"]
BATTERYLIFE_RAW_NAME = {"ul_pur": "UL_PUR", "hnei": "HNEI", "snl": "SNL", "mich": "MICH",
                        "mich_exp": "MICH_EXP", "rwth": "RWTH", "stanford": "Stanford",
                        "stanford_2": "Stanford_2", "isu_ilcc": "ISU_ILCC"}
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}

# Already-established numbers (item 5a mean, item 1's true-winner
# designation), reused not recomputed.
BASE_MODEL_R2 = {"CALCE": 0.748756, "Oxford": 0.939630, "HUST": 0.795020, "XJTU": -1.036732,
                 "ul_pur": 0.116107, "hnei": -0.037714, "snl": 0.146501, "mich": 0.572789,
                 "mich_exp": 0.720633, "rwth": -0.484674, "stanford": 0.111290,
                 "stanford_2": 0.065504, "isu_ilcc": 0.140112}
ROUTED_R2_ORACLE = {"CALCE": 0.739702, "Oxford": 0.953262, "HUST": 0.800007, "XJTU": -1.036732,
                    "ul_pur": 0.116107, "hnei": -0.037714, "snl": 0.146501, "mich": 0.572789,
                    "mich_exp": 0.720633, "rwth": -0.484674, "stanford": 0.111290,
                    "stanford_2": 0.065504, "isu_ilcc": 0.140112}


from severson_features import build_severson_rows


def batterylife_severson_df(source_lower):
    raw_name = BATTERYLIFE_RAW_NAME[source_lower]
    ids = batterylife_cell_ids(raw_name)
    rows = []
    for cid in ids:
        cycles = list(iterate_batterylife_cycles(raw_name, cid))
        if len(cycles) < 5:
            continue
        soh_map = soh_per_cycle(cycles)
        rows += build_severson_rows(raw_name, cid, cycles, soh_map)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("=== Final pass 2, item C: complete baseline table on all 13 external datasets ===")

    # --- (i) trivial linear baseline, fit on NASA+MIT train split ---
    merged, hi_full = load_nasa_mit_pool(reformulated=False)
    train_mask, test_mask, split = battery_split_masks(merged)
    lin = LinearRegression().fit(merged.loc[train_mask, ["cycle_idx"]], merged.loc[train_mask, "SOH"])
    print(f"[itemC] trivial linear baseline: SOH = {lin.intercept_:.3f} + {lin.coef_[0]:.5f}*cycle_idx")

    # --- (ii) Severson/Attia baselines: reuse Stage 6.1's models, extend to 9 BatteryLife sources ---
    print("[itemC] building Severson-style features for the 42-battery pool + all 13 held-out sets...")
    pool_df = build_42pool_severson_df()
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    train_df = pool_df[pool_df["battery_id"].isin(train_ids)]
    test_df_indomain = pool_df[pool_df["battery_id"].isin(test_ids)]

    held_out_severson = {
        "CALCE": build_dataset_severson_df("CALCE", CALCE_CELLS, iterate_calce_cycles),
        "Oxford": build_dataset_severson_df("Oxford", oxford_cell_ids(), iterate_oxford_cycles),
        "HUST": build_dataset_severson_df("HUST", hust_cell_ids(), iterate_hust_cycles),
        "XJTU": build_dataset_severson_df("XJTU", xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles),
    }
    for source in BATTERYLIFE_LOWER:
        try:
            df = batterylife_severson_df(source)
            if len(df) > 0:
                held_out_severson[source] = df
                print(f"[itemC]   {source}: {len(df)} severson rows, {df['battery_id'].nunique()} batteries")
            else:
                print(f"[itemC]   {source}: 0 usable severson rows - excluded")
        except Exception as e:
            print(f"[itemC]   {source}: FAILED to build severson features ({type(e).__name__}: {e}) - excluded")

    severson_models = {}
    baseline_rows = []
    for label, feat_cols in [("severson_variance", VARIANCE_FEATS), ("attia_rich", RICH_FEATS)]:
        r_indomain, model, scaler, medians = fit_eval(train_df, test_df_indomain, feat_cols, label)
        severson_models[label] = (model, scaler, medians, feat_cols)
        print(f"[itemC] {label} in-domain: R2={r_indomain['r2']:.4f}")

    # --- deployed base/extended models for oracle-routed reference + per-battery errors ---
    merged_b2, hi_full_base = load_nasa_mit_pool(reformulated=True)
    train_mask_b2, test_mask_b2, _ = battery_split_masks(merged_b2)
    base_cols = base_feature_cols()
    ext_cols = extended_feature_cols()
    base_medians = fit_medians(merged_b2, train_mask_b2, base_cols)
    base_model = load_base_model()
    ext_model = load_extended_model()
    heldout_base_frames = load_all_heldout_base(hi_full_base)
    from researchpass_partA_common import load_extended_pool_and_split
    merged_e2, hi_full_ext, hi_full_raw, train_mask_e2, test_mask_e2, _ = load_extended_pool_and_split()
    ext_medians = fit_medians(merged_e2, train_mask_e2, ext_cols)
    heldout_ext_frames = load_all_heldout_extended(hi_full_raw)

    def base_model_battery_mae(name):
        """Per-battery MAE of the deployed BASE model (never the routed/
        extended one - the paired test's own reference point is
        explicitly the zero-retrain base model, matching this table's
        (iii) column) on target dataset `name`."""
        if name in heldout_base_frames:
            df = heldout_base_frames[name]
        else:
            path = PROC_DIR / f"batterylife_{name}_merged.parquet"
            if not path.exists():
                return None
            df = pd.read_parquet(path)
        X = build_X(df, base_cols, base_medians)
        pred = base_model.predict(X)
        return pd.Series(np.abs(df["SOH"].to_numpy() - pred), index=df["battery_id"]).groupby(level=0).mean()

    # --- LODO results, reused from item B (assumed already run) ---
    lodo_path = OUT_DIR / "finalpass2_itemB_lodo_results.csv"
    lodo_df = pd.read_csv(lodo_path) if lodo_path.exists() else None
    if lodo_df is None:
        print("[itemC] WARNING: item B's LODO results not found yet - 'best_lodo_r2' column will be NaN")

    all_datasets = ["CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_LOWER
    summary_rows = []
    stat_test_rows = []

    for name in all_datasets:
        if name not in held_out_severson:
            print(f"[itemC] {name}: no Severson features available - row will have NaN for those columns")

        # trivial linear
        df_sev = held_out_severson.get(name)
        trivial_r2 = np.nan
        battery_mae_by_method = {}
        if df_sev is not None and "cycle_idx" in df_sev.columns:
            pred_triv = lin.predict(df_sev[["cycle_idx"]])
            trivial_r2 = float(r2_score(df_sev["SOH"], pred_triv))
            battery_mae_by_method["trivial_linear"] = pd.Series(
                np.abs(df_sev["SOH"].to_numpy() - pred_triv), index=df_sev["battery_id"]).groupby(level=0).mean()

        sev_r2, attia_r2 = np.nan, np.nan
        if df_sev is not None:
            for label, (model, scaler, medians, feat_cols) in severson_models.items():
                X = df_sev[feat_cols].to_numpy(dtype=float).copy()
                inds = np.where(np.isnan(X))
                X[inds] = np.take(medians, inds[1])
                Xs = scaler.transform(X)
                pred = model.predict(Xs)
                y_true = df_sev["SOH"].to_numpy(dtype=float)
                r2 = float(r2_score(y_true, pred))
                if label == "severson_variance":
                    sev_r2 = r2
                else:
                    attia_r2 = r2
                battery_mae_by_method[label] = pd.Series(np.abs(y_true - pred), index=df_sev["battery_id"]).groupby(level=0).mean()

        base_r2 = BASE_MODEL_R2.get(name, np.nan)
        routed_r2 = ROUTED_R2_ORACLE.get(name, np.nan)
        best_lodo_r2 = np.nan
        if lodo_df is not None:
            row = lodo_df[lodo_df["held_out_source"].str.lower() == name.lower()]
            if len(row):
                best_lodo_r2 = float(row["lodo_r2"].iloc[0])

        summary_rows.append({
            "dataset": name, "trivial_linear_r2": trivial_r2, "severson_variance_r2": sev_r2,
            "attia_rich_r2": attia_r2, "deployed_base_r2": base_r2,
            "routed_oracle_selected_r2": routed_r2, "best_lodo_r2": best_lodo_r2,
        })
        print(f"[itemC] {name}: trivial={trivial_r2:.4f} severson={sev_r2:.4f} attia={attia_r2:.4f} "
              f"base={base_r2:.4f} routed(ORACLE)={routed_r2:.4f} best_lodo={best_lodo_r2}")

        # --- paired battery-level tests: deployed base model vs. each baseline ---
        base_mae = base_model_battery_mae(name)
        if base_mae is not None:
            for method_label, other_mae in battery_mae_by_method.items():
                common = base_mae.index.intersection(other_mae.index)
                if len(common) < 3:
                    continue
                a = base_mae.loc[common].to_numpy()
                b = other_mae.loc[common].to_numpy()
                diff = a - b  # negative = base model has LOWER (better) MAE
                try:
                    stat, p = wilcoxon(a, b)
                except ValueError:
                    stat, p = np.nan, np.nan
                rng = np.random.default_rng(42)
                boot_diffs = []
                for _ in range(2000):
                    idx = rng.integers(0, len(common), len(common))
                    boot_diffs.append(float(np.mean(diff[idx])))
                ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
                stat_test_rows.append({
                    "dataset": name, "baseline_compared": method_label, "n_batteries": len(common),
                    "base_model_mean_mae": float(a.mean()), "baseline_mean_mae": float(b.mean()),
                    "mean_mae_diff_base_minus_baseline": float(diff.mean()),
                    "bootstrap_ci_lo": float(ci_lo), "bootstrap_ci_hi": float(ci_hi),
                    "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                    "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
                    "base_significantly_better": bool(np.isfinite(p) and p < 0.05 and diff.mean() < 0),
                })

    stat_test_df = pd.DataFrame(stat_test_rows)
    stat_test_df.to_csv(OUT_DIR / "finalpass2_itemC_paired_tests.csv", index=False)
    print("\n=== PAIRED TESTS: deployed base model vs. each baseline (battery-level MAE) ===")
    print(stat_test_df.to_string(index=False))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "finalpass2_itemC_baseline_table.csv", index=False)

    print("\n=== FULL BASELINE TABLE ===")
    print(summary_df.to_string(index=False))

    print(f"\n[itemC] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
