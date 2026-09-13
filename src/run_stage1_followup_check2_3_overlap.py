"""
Stage 1 closeout, Checks 2+3 (run together - they share the same 4
model configurations, so building each config once and reusing it for
both the point-accuracy comparison (Check 2) and the Jackknife+
comparison (Check 3) avoids redundant retraining):

Check 2: does 1.5's monotone_constraints gain overlap with 1.1's
reformulated features, or are they independent/additive?

Check 3: is the Jackknife+ CALCE-coverage jump driven by 1.1, 1.5, or
their combination specifically?

4 configurations, each trained once:
  - baseline:      raw canonical features, no monotone_constraints
  - 1.1_alone:     1.1's reformulated features, no monotone_constraints
  - 1.5_alone:     raw canonical features + cycle_idx, monotone_constraints
  - 1.1_plus_1.5:  1.1's reformulated features + cycle_idx, monotone_constraints
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    CANONICAL_FEATURES, DURATION_FEATURES, canonical_feature_cols,
    load_nasa_mit_pool, battery_split_masks, fit_xgb, eval_indomain,
    build_calce_tensors, eval_calce, calce_coverage, fusion_cols, OUT_DIR, ALPHA,
)
from run_conformal import calib_eval_battery_split
from mapie.regression import CrossConformalRegressor


def build_calce_merged_fresh(hi_df_full):
    """ROOT-CAUSED BUG, not worked around silently: stage1_common.py's
    own build_calce_merged() caches its RESULT DataFrame at module level
    across calls (_CALCE_FUSION_CACHE["df"]), correct for every existing
    single-config script but WRONG here - this script calls it with TWO
    different hi_df_full arguments (raw vs. reformulated) in the same
    process, and the second call silently returned the first call's
    stale (raw) result, causing a KeyError when 1.1's _rel columns were
    looked up on what was actually still the raw frame. Fixed locally by
    reusing the (correctly hi_df-independent) CALCE TENSOR/embedding
    cache, but doing the final hi_df merge fresh every call instead of
    relying on the buggy cross-call DataFrame cache."""
    import json
    import torch
    from pathlib import Path as _Path
    from models.ica_encoder import ICAEncoder
    ROOT_ = _Path(__file__).resolve().parents[1]
    PROC_DIR_ = ROOT_ / "data" / "processed"
    X_calce, soh_calce, bid_calce, cyc_calce = build_calce_tensors()
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT_ / "models" / "ica_encoder.pt"))
    encoder.eval()
    with torch.no_grad():
        fusion_emb = encoder.encode(torch.tensor(X_calce[:, :, slice(3, 6)])).numpy()
    calce_hi = hi_df_full[hi_df_full["dataset"] == "CALCE"]
    seq_df = pd.DataFrame({"battery_id": bid_calce, "cycle_idx": cyc_calce})
    for i in range(16):
        seq_df[f"fusion_{i}"] = fusion_emb[:, i]
    return pd.merge(calce_hi, seq_df, on=["battery_id", "cycle_idx"], how="inner")


def run_jackknife(merged, cols, medians, calce_merged, calib_ids, test_mask, monotone):
    calib_mask_full = merged["battery_id"].isin(calib_ids).to_numpy() & test_mask
    X_all = merged[cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X_all))
    X_all[inds] = np.take(medians, inds[1])
    X_calib_cv = X_all[calib_mask_full]
    y_calib_cv = merged.loc[calib_mask_full, "SOH"].to_numpy()
    groups_cv = merged.loc[calib_mask_full, "battery_id"].to_numpy()
    n_calib_batteries = len(set(groups_cv))

    X_calce_cv = calce_merged[cols].to_numpy(dtype=float, copy=True)
    for j in range(len(cols)):
        nan_mask = np.isnan(X_calce_cv[:, j])
        if nan_mask.any():
            X_calce_cv[nan_mask, j] = medians[j]
    y_calce_true = calce_merged["SOH"].to_numpy()

    kwargs = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  n_jobs=-1, reg_lambda=1.0)
    if monotone is not None:
        kwargs["monotone_constraints"] = monotone
    base_est = XGBRegressor(**kwargs)
    cv_loo = GroupKFold(n_splits=n_calib_batteries)
    mapie_jk = CrossConformalRegressor(estimator=base_est, confidence_level=1 - ALPHA, method="plus", cv=cv_loo)
    mapie_jk.fit_conformalize(X_calib_cv, y_calib_cv, groups=groups_cv)
    pred_jk, interval_jk = mapie_jk.predict_interval(X_calce_cv)
    lo_jk, hi_jk = interval_jk[:, 0, 0], interval_jk[:, 1, 0]
    covered_jk = (y_calce_true >= lo_jk) & (y_calce_true <= hi_jk)
    return float(covered_jk.mean()), float(np.mean(hi_jk - lo_jk))


def build_and_eval(name, base_features, add_cycle_idx, merged, hi_full, calce_merged, train_mask, test_mask):
    feature_cols = list(base_features) + (["cycle_idx"] if add_cycle_idx else [])
    monotone = None
    if add_cycle_idx:
        n_fusion = len(fusion_cols())
        monotone = tuple([0] * len(base_features) + [-1] + [0] * n_fusion)

    kwargs = {"monotone_constraints": monotone} if monotone is not None else None
    model, medians, cols = fit_xgb(merged, train_mask, feature_cols, xgb_extra_kwargs=kwargs)
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    calce = eval_calce(model, medians, feature_cols, calce_merged)
    cov = calce_coverage(indomain, calce)

    calib_ids, eval_ids = calib_eval_battery_split(sorted(set(indomain["battery_id"].tolist())))
    jk_cov, jk_width = run_jackknife(merged, cols, medians, calce_merged, calib_ids, test_mask, monotone)

    print(f"[check2-3] [{name}] in-domain R2={indomain['r2']:.4f} | CALCE R2={calce['r2']:.4f} | "
          f"split-conformal coverage={cov['empirical_coverage']:.4f} (width={cov['avg_interval_width']:.3f}) | "
          f"Jackknife+ coverage={jk_cov:.4f} (width={jk_width:.3f})")
    return {"config": name, "indomain_r2": indomain["r2"], "calce_r2": calce["r2"],
            "split_conformal_coverage": cov["empirical_coverage"], "split_conformal_width": cov["avg_interval_width"],
            "jackknife_coverage": jk_cov, "jackknife_width": jk_width}


def main():
    print("[check2-3] === Check 2, point 1: correlation between cycle_idx and 1.1's reformulated features ===")
    merged_reformulated, hi_full_reformulated = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged_reformulated)
    train_df = merged_reformulated[train_mask]
    for feat in DURATION_FEATURES:
        col = f"{feat}_rel"
        r = train_df["cycle_idx"].corr(train_df[col])
        print(f"[check2-3] corr(cycle_idx, {col}) across training set: {r:.4f}")

    print("\n[check2-3] === Building 4 configurations (baseline / 1.1-alone / 1.5-alone / 1.1+1.5) ===")
    merged_raw, hi_full_raw = load_nasa_mit_pool(reformulated=False)
    calce_merged_raw = build_calce_merged_fresh(hi_full_raw)
    calce_merged_reformulated = build_calce_merged_fresh(hi_full_reformulated)
    # sanity check the fix: these must now actually differ
    assert "ICHV_rel" not in calce_merged_raw.columns, "raw CALCE frame should NOT have _rel columns"
    assert "ICHV_rel" in calce_merged_reformulated.columns, "reformulated CALCE frame MUST have _rel columns"
    print("[check2-3] verified: raw vs. reformulated CALCE frames are now genuinely distinct (cache bug fixed)")

    results = []
    results.append(build_and_eval("baseline", CANONICAL_FEATURES, False,
                                   merged_raw, hi_full_raw, calce_merged_raw, train_mask, test_mask))
    results.append(build_and_eval("1.1_alone", canonical_feature_cols(reformulated=True), False,
                                   merged_reformulated, hi_full_reformulated, calce_merged_reformulated, train_mask, test_mask))
    results.append(build_and_eval("1.5_alone", CANONICAL_FEATURES, True,
                                   merged_raw, hi_full_raw, calce_merged_raw, train_mask, test_mask))
    results.append(build_and_eval("1.1_plus_1.5", canonical_feature_cols(reformulated=True), True,
                                   merged_reformulated, hi_full_reformulated, calce_merged_reformulated, train_mask, test_mask))

    df = pd.DataFrame(results)
    print("\n[check2-3] === FULL 4-ROW COMPARISON TABLE ===")
    print(df.to_string(index=False))

    base_calce_r2 = df.loc[df.config == "baseline", "calce_r2"].iloc[0]
    c11_calce_r2 = df.loc[df.config == "1.1_alone", "calce_r2"].iloc[0]
    c15_calce_r2 = df.loc[df.config == "1.5_alone", "calce_r2"].iloc[0]
    c115_calce_r2 = df.loc[df.config == "1.1_plus_1.5", "calce_r2"].iloc[0]
    delta_15_alone = c15_calce_r2 - base_calce_r2
    delta_15_on_11 = c115_calce_r2 - c11_calce_r2
    print(f"\n[check2-3] === CHECK 2: does 1.5's gain depend on 1.1's features? ===")
    print(f"[check2-3] 1.5's CALCE R2 gain WITHOUT 1.1 (raw features): {delta_15_alone:+.4f} "
          f"({base_calce_r2:.4f} -> {c15_calce_r2:.4f})")
    print(f"[check2-3] 1.5's CALCE R2 gain WITH 1.1 (reformulated features, i.e. layered on top): "
          f"{delta_15_on_11:+.4f} ({c11_calce_r2:.4f} -> {c115_calce_r2:.4f})")
    similar = abs(delta_15_alone - delta_15_on_11) < 0.01
    if similar:
        check2_outcome = ("(a) SIMILAR magnitude with or without 1.1 - the two capture genuinely distinct "
                          "signal and can be described as independent, additive contributions.")
    else:
        check2_outcome = ("(b) SUBSTANTIALLY DIFFERENT depending on whether 1.1's features are present - "
                          "the two interact, and 1.5's contribution should be described as layered on 1.1 "
                          "specifically, not as an independent finding.")
    print(f"[check2-3] CHECK 2 OUTCOME: {check2_outcome}")

    base_split = df.loc[df.config == "baseline", "split_conformal_coverage"].iloc[0]
    base_jk = df.loc[df.config == "baseline", "jackknife_coverage"].iloc[0]
    c11_split = df.loc[df.config == "1.1_alone", "split_conformal_coverage"].iloc[0]
    c11_jk = df.loc[df.config == "1.1_alone", "jackknife_coverage"].iloc[0]
    c15_split = df.loc[df.config == "1.5_alone", "split_conformal_coverage"].iloc[0]
    c15_jk = df.loc[df.config == "1.5_alone", "jackknife_coverage"].iloc[0]
    c115_split = df.loc[df.config == "1.1_plus_1.5", "split_conformal_coverage"].iloc[0]
    c115_jk = df.loc[df.config == "1.1_plus_1.5", "jackknife_coverage"].iloc[0]

    jump_base = base_jk - base_split
    jump_11 = c11_jk - c11_split
    jump_15 = c15_jk - c15_split
    jump_115 = c115_jk - c115_split
    print(f"\n[check2-3] === CHECK 3: which factor drives the Jackknife+ jump? ===")
    print(f"[check2-3] baseline jump:     {base_split:.4f} -> {base_jk:.4f} ({jump_base:+.4f})")
    print(f"[check2-3] 1.1-alone jump:    {c11_split:.4f} -> {c11_jk:.4f} ({jump_11:+.4f})")
    print(f"[check2-3] 1.5-alone jump:    {c15_split:.4f} -> {c15_jk:.4f} ({jump_15:+.4f})")
    print(f"[check2-3] 1.1+1.5 jump:      {c115_split:.4f} -> {c115_jk:.4f} ({jump_115:+.4f})")

    # attribute: does either alone approach the combined jump?
    thresh = 0.7 * jump_115  # "approaches" = at least 70% of the combined jump
    if jump_11 >= thresh and jump_15 < thresh:
        check3_outcome = "1.1 (reformulated features) is the primary driver of the large Jackknife+ jump."
    elif jump_15 >= thresh and jump_11 < thresh:
        check3_outcome = "1.5 (monotone_constraints) is the primary driver of the large Jackknife+ jump."
    elif jump_11 >= thresh and jump_15 >= thresh:
        check3_outcome = "BOTH 1.1 and 1.5 individually approach the combined jump - largely redundant drivers."
    else:
        check3_outcome = ("NEITHER 1.1 nor 1.5 alone approaches the combined jump - this is a genuine "
                          "INTERACTION EFFECT between the two, not attributable to either individually.")
    print(f"[check2-3] CHECK 3 OUTCOME: {check3_outcome}")

    df.to_csv(OUT_DIR / "stage1_followup_check2_3_comparison.csv", index=False)
    print("\n[check2-3] saved outputs/stage1_followup_check2_3_comparison.csv")
    print("[check2-3] DONE")


if __name__ == "__main__":
    main()
