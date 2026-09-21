"""
Research pass Group B, item 5 (highest priority in this pass): survival
analysis reframing of RUL - Cox Proportional Hazards, censoring-aware.

MOTIVATION, root-caused not assumed: the CURRENT deployed RUL model
(joint_adaptive_fusion) trains as a plain point regression against
`rul_map` values, computed by compute_eol_and_rul_severson_aware. That
function ALREADY computes and returns a per-battery `censored` flag
(True when a battery's recorded test ends before its capacity ever
falls to the 80% EOL threshold - we know it survived AT LEAST that
long, but not its true remaining life) - but the regression framing
silently DISCARDS this signal, treating a censored battery's last-
recorded RUL value as if it were a known, exact target. This is
exactly the problem survival analysis exists to solve.

DATA: hi_table.parquet already carries a computed `RUL` column for
NASA/MIT/CALCE (Severson-aware convention, matching everything else in
this project) - censoring is inferred here as "this battery's own LAST
recorded row has RUL > 0" (test ended before reaching EOL), verified
per-battery not assumed uniform. Oxford/HUST/XJTU do NOT have RUL
precomputed anywhere in this project (confirmed by checking the
Stage 5.1 merged parquets directly) - extending to those would need a
fresh EOL/RUL computation pass, out of this item's time budget;
reported as a real, disclosed scope limitation, not silently skipped.

MODEL: lifelines' CoxPHFitter (standard, non-time-varying - each
(battery, cycle) row is treated as an independent "clock re-started
here" survival observation, matching this project's own established
per-cycle-RUL-target convention, just replacing point regression with
a proper censoring-aware model). scikit-survival (which would give
cumulative_dynamic_auc, i.e. genuine time-dependent AUC) could not be
installed in this environment (its `ecos` dependency needs a C++
build toolchain not present here) - disclosed, not silently
substituted with a fabricated number. Concordance index (lifelines'
own, the standard survival discrimination metric) is reported instead,
which the task itself listed as an acceptable primary metric.

COMPARABILITY: converts the fitted Cox model's own predicted median
survival time into a plain RUL point prediction, scored with the SAME
R2/RMSE this project has used for every other RUL result - so this is
directly comparable to the deployed model's own 0.6657 R2, not just a
concordance-index number in isolation.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import canonical_feature_cols, add_reformulated_duration_features, build_calce_merged, fusion_cols

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
BASELINE_RUL_R2 = 0.6656929850578308  # on record, outputs/stage1_followup_partB_joint_rul_results.csv


def infer_censoring(df: pd.DataFrame) -> pd.DataFrame:
    """Per-battery: censored=True if that battery's own LAST recorded
    cycle still has RUL > 0 (test ended before reaching 80% EOL) -
    verified directly per battery, not assumed uniform across a
    dataset."""
    last_rul = df.sort_values("cycle_idx").groupby("battery_id")["RUL"].last()
    censored_map = (last_rul > 0).to_dict()
    df = df.copy()
    df["censored"] = df["battery_id"].map(censored_map)
    df["event"] = ~df["censored"]
    return df


def main():
    t0 = time.time()
    print("=== Research pass Group B, item 5: Cox survival analysis for RUL ===")
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_reformulated = add_reformulated_duration_features(hi_df)
    feature_cols = canonical_feature_cols(reformulated=True)
    fcols = fusion_cols()
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")

    nasa_mit = hi_reformulated[hi_reformulated["dataset"].isin(["NASA", "MIT"])]
    merged = pd.merge(nasa_mit, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    merged = infer_censoring(merged)

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_mask = merged["battery_id"].isin(split["train_ids"])
    test_mask = merged["battery_id"].isin(split["test_ids"])
    print(f"[survival] train rows: {train_mask.sum()}, test rows: {test_mask.sum()}")
    n_censored_train = merged.loc[train_mask].groupby("battery_id")["censored"].first().sum()
    n_train_batteries = merged.loc[train_mask, "battery_id"].nunique()
    print(f"[survival] {n_censored_train}/{n_train_batteries} TRAIN batteries are censored "
          f"(never observed reaching 80% EOL within their recorded test)")

    all_feat_cols = feature_cols + ["cycle_idx"] + fcols
    train_df = merged.loc[train_mask, all_feat_cols + ["RUL", "event"]].copy()
    test_df = merged.loc[test_mask, all_feat_cols + ["RUL", "event"]].copy()

    # median-impute using TRAIN-only medians, same convention as every
    # other model in this project
    train_medians = train_df[all_feat_cols].median(numeric_only=True)
    train_df[all_feat_cols] = train_df[all_feat_cols].fillna(train_medians)
    test_df[all_feat_cols] = test_df[all_feat_cols].fillna(train_medians)
    # Cox needs duration > 0 strictly for its own internal log-space handling in some
    # versions; RUL=0 rows (already at/past EOL) get a tiny positive floor, disclosed.
    train_df["RUL"] = train_df["RUL"].clip(lower=0.5)
    test_df["RUL"] = test_df["RUL"].clip(lower=0.5)

    print(f"[survival] fitting CoxPHFitter on {len(train_df)} rows, {len(all_feat_cols)} covariates...")
    cph = CoxPHFitter(penalizer=0.1)  # L2 penalty - a 25-dim covariate Cox fit is prone to
                                       # separation/collinearity without it, a real risk with
                                       # this many correlated HI+fusion covariates, not a
                                       # tuned choice
    cph.fit(train_df, duration_col="RUL", event_col="event")
    print(f"[survival] fit converged: log-likelihood={cph.log_likelihood_:.2f}, "
          f"concordance (train, lifelines' own internal)={cph.concordance_index_:.4f}")

    # concordance index on TEST - higher predicted RISK should correspond
    # to SHORTER survival, so pass partial_hazard directly (lifelines'
    # own convention: concordance_index expects higher score = higher
    # risk = shorter duration when event_observed marks a real event)
    test_hazard = cph.predict_partial_hazard(test_df[all_feat_cols])
    c_index_test = concordance_index(test_df["RUL"], -test_hazard, test_df["event"])
    print(f"[survival] TEST concordance index: {c_index_test:.4f} (0.5=random, 1.0=perfect ranking)")

    # point-prediction comparability: median survival time -> RUL point estimate
    median_pred = cph.predict_median(test_df[all_feat_cols])
    # predict_median can return inf for very-low-risk rows (median beyond
    # observed follow-up) - clip to the max observed TRAIN duration as a
    # disclosed, reasonable ceiling rather than leaving inf in the metric
    max_train_rul = float(train_df["RUL"].max())
    median_pred_clipped = median_pred.replace([np.inf, -np.inf], max_train_rul).fillna(max_train_rul)
    n_inf = int(np.isinf(median_pred).sum())
    print(f"[survival] {n_inf}/{len(median_pred)} TEST predictions were infinite median survival "
          f"(clipped to TRAIN max RUL={max_train_rul:.0f} for R2/RMSE comparability)")

    y_true = test_df["RUL"].to_numpy()
    y_pred = median_pred_clipped.to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"[survival] TEST RUL (median-survival-time point estimate): RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f}")

    verdict = "WIN" if r2 > BASELINE_RUL_R2 else "LOSS"
    print(f"\n[survival] R2 vs. deployed joint model (R2={BASELINE_RUL_R2:.4f}): {verdict} ({r2:.4f})")

    # zero-retrain: CALCE (has RUL precomputed; Oxford/HUST/XJTU do not - disclosed above)
    calce_merged = build_calce_merged(hi_reformulated)
    calce_merged = infer_censoring(calce_merged)
    calce_censored_frac = calce_merged.groupby("battery_id")["censored"].first().mean()
    print(f"\n[survival] CALCE: {calce_censored_frac:.0%} of batteries censored")
    calce_df = calce_merged[all_feat_cols + ["RUL", "event"]].copy()
    calce_df[all_feat_cols] = calce_df[all_feat_cols].fillna(train_medians)
    calce_df["RUL"] = calce_df["RUL"].clip(lower=0.5)
    calce_hazard = cph.predict_partial_hazard(calce_df[all_feat_cols])
    c_index_calce = concordance_index(calce_df["RUL"], -calce_hazard, calce_df["event"])
    calce_median_pred = cph.predict_median(calce_df[all_feat_cols]).replace(
        [np.inf, -np.inf], max_train_rul).fillna(max_train_rul)
    calce_r2 = float(r2_score(calce_df["RUL"], calce_median_pred))
    calce_rmse = float(np.sqrt(mean_squared_error(calce_df["RUL"], calce_median_pred)))
    print(f"[survival] CALCE (zero-retrain): concordance={c_index_calce:.4f}, "
          f"RUL R2={calce_r2:.4f}, RMSE={calce_rmse:.2f}")

    results = pd.DataFrame([
        {"eval_set": "TEST (in-domain)", "concordance_index": c_index_test, "rul_r2": r2, "rul_rmse": rmse,
         "rul_mae": mae, "n": len(test_df), "n_infinite_median_clipped": n_inf},
        {"eval_set": "CALCE (zero-retrain)", "concordance_index": c_index_calce, "rul_r2": calce_r2,
         "rul_rmse": calce_rmse, "rul_mae": None, "n": len(calce_df), "n_infinite_median_clipped": None},
        {"eval_set": "DEPLOYED joint model (on record, point regression, no censoring awareness)",
         "concordance_index": None, "rul_r2": BASELINE_RUL_R2, "rul_rmse": None, "rul_mae": None, "n": None,
         "n_infinite_median_clipped": None},
    ])
    results.to_csv(OUT_DIR / "researchpass_groupB5_survival_analysis.csv", index=False)
    print("\n=== SUMMARY ===")
    print(results.to_string(index=False))
    print(f"\n[survival] SCOPE LIMITATION, disclosed: Oxford/HUST/XJTU not evaluated - no RUL/"
          f"censoring precomputed for these datasets anywhere in this project; would need a "
          f"fresh EOL computation pass, out of this item's time budget.")
    print(f"[survival] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
