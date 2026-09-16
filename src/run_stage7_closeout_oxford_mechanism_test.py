"""
Stage 7 closeout, item 1: Oxford-mechanism test. Oxford has shown a
disproportionate benefit across FOUR independent contexts now (Stage
5's SCV/MATD/VIECT/MET reformulation, Stage 7.1's World Model, Stage
7.3's DeepONet-embedding ablation, plus Stage 5's own pool-composition
work referenced in the task). Two competing, untested explanations:
(a) Oxford is simply the SMALLEST held-out set (8 cells) - more
    statistical headroom/variance to show a large swing by chance, or
(b) Oxford has the LOWEST domain-classifier AUC (0.9881, most similar
    to training data) among the 4 held-out sets - genuinely easier to
    improve on mechanistically, independent of sample size.

TEST: construct a SIZE-MATCHED (8-cell) random subsample of HUST -
HUST is the natural comparison because it is large enough (77 cells)
to subsample from, and its own full-77-cell result was the OPPOSITE
of Oxford's in both interventions tested here (reformulation helped
HUST too, but far less dramatically; the World Model actively LOST to
persistence on full HUST). If explanation (a) is right, an 8-cell
HUST subsample should show a swing comparable in MAGNITUDE to
Oxford's own (small-sample variance is a property of the subsample
size, not of which dataset it's drawn from). If explanation (b) is
right, subsampling HUST does NOT change its own distributional
similarity to the training pool (AUC is a property of the full
feature distribution, not of which 8 cells happen to be drawn) - so
the subsample should show an effect much closer to full HUST's own
(modest) result, not Oxford's (dramatic) one.

STRATIFIED, DISCLOSED SAMPLING: HUST's 77 cell IDs are named
"{batch}-{cell_within_batch}" (confirmed directly, not assumed - see
data_adapters.hust_cell_ids()), a real natural grouping of 10 batches
(charging-protocol conditions), 7-8 cells each. A representative,
non-cherry-picked 8-cell sample: pick 8 of the 10 batches at random
(seed=42), one cell from each selected batch (also random within
batch, same seed) - spans most of HUST's own protocol diversity rather
than risking all 8 cells from one or two batches.

Two interventions re-run on this subsample, reusing ALREADY-TRAINED,
ALREADY-SAVED models (no retraining of either model - only the
per-feature-set train-only imputation medians are recomputed, the
same fast, deterministic statistic `fit_xgb` itself would compute,
not a retrain):
1. Stage 5's SCV/MATD/VIECT/MET reformulation: models/xgb_soh_fusion.json
   (base, Stage 1.1-only) vs models/_experimental_xgb_soh_fusion_
   extended_reformulation.json (extended) - exactly the same "before"/
   "after" model pair run_stage5_extended_reformulation_eval.py itself
   used for the original Oxford/HUST/CALCE/XJTU numbers.
2. Stage 7.1's World Model (models/_experimental_world_model.pt) vs.
   the naive persistence baseline - exactly the same protocol as
   run_stage7_1_world_model.py's own eval_rollout.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage1_common import (
    load_nasa_mit_pool, canonical_feature_cols, fusion_cols, battery_split_masks, OUT_DIR, PROC_DIR, ROOT,
)
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols
from data_adapters import hust_cell_ids, iterate_hust_cycles
from stage7_common import fit_norm_stats, load_pool_train_test
from run_stage7_1_world_model import make_windows, eval_rollout, WINDOW, HORIZON, WINDOW_STRIDE
from sequence_features import build_dataset_tensors, apply_channel_norm
from models.world_model import WorldModel
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42


def sample_hust_cells_stratified(n=8, seed=SEED):
    ids = hust_cell_ids()
    from collections import defaultdict
    groups = defaultdict(list)
    for cid in ids:
        groups[cid.split("-")[0]].append(cid)
    batches = sorted(groups.keys(), key=lambda x: int(x))
    rng = np.random.default_rng(seed)
    chosen_batches = rng.choice(batches, size=min(n, len(batches)), replace=False)
    sample = []
    for b in chosen_batches:
        cell = rng.choice(groups[b])
        sample.append(cell)
    return sorted(sample)


def eval_generic(model, medians, feature_cols, df, fcols):
    cols = feature_cols + fcols
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    for j in range(len(feature_cols)):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = medians[j]
    y_true = df["SOH"].to_numpy()
    pred = model.predict(X)
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(np.mean(np.abs(y_true - pred))),
            "r2": float(r2_score(y_true, pred)), "n": len(y_true)}


def reformulation_intervention(hust_subsample_ids):
    print("\n=== Intervention 1: Stage 5 SCV/MATD/VIECT/MET reformulation on HUST-8-cell subsample ===")
    merged_nm, hi_full = load_nasa_mit_pool(reformulated=True)
    hi_full_ext = add_scv_matd_viect_reformulated(hi_full)
    fcols = fusion_cols()
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    nasa_mit_ext = hi_full_ext[hi_full_ext["dataset"].isin(["NASA", "MIT"])]
    merged_ext = pd.merge(nasa_mit_ext, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    train_mask, test_mask, split = battery_split_masks(merged_ext)
    train_df = merged_ext.loc[train_mask]

    # feature_cols includes cycle_idx - matching run_stage5_extended_
    # reformulation_eval.py's OWN convention exactly (feature_cols =
    # extended_cols + ["cycle_idx"]; fit_xgb/eval_generic then append
    # fcols on top) - both saved models were trained on this exact
    # column set + order, so eval must match it precisely or XGBoost's
    # own feature-count check fails loudly (as it did on the first
    # attempt here, missing cycle_idx - caught immediately, not
    # silently mis-scored).
    base_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    extended_cols = extended_canonical_feature_cols(canonical_feature_cols(reformulated=True)) + ["cycle_idx"]

    # SAME train-only-median statistic fit_xgb itself computes - not a
    # retrain, just recomputing the imputation reference for each
    # already-established feature set so the saved models can be scored.
    X_base_train = train_df[base_cols + fcols].to_numpy(dtype=float)
    med_base = np.nanmedian(X_base_train, axis=0)
    X_ext_train = train_df[extended_cols + fcols].to_numpy(dtype=float)
    med_ext = np.nanmedian(X_ext_train, axis=0)

    model_base = XGBRegressor(); model_base.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    model_ext = XGBRegressor(); model_ext.load_model(str(ROOT / "models" / "_experimental_xgb_soh_fusion_extended_reformulation.json"))

    hust_full = pd.read_parquet(PROC_DIR / "stage5_1_hust_merged.parquet")
    # reset_index: add_scv_matd_viect_reformulated internally does
    # positional-style `rel[idx] = ...` assuming a clean 0..N-1
    # RangeIndex (true for every OTHER caller in this project, which
    # always pass a freshly-merged or full hi_df) - a filtered SUBSET
    # keeps the ORIGINAL (non-contiguous, out-of-range) row labels
    # unless reset here. A real bug this script's own filtering step
    # introduced, fixed at the source of the problem (this filter),
    # not by changing the shared, already-verified module.
    hust_sub = hust_full[hust_full["battery_id"].isin(hust_subsample_ids)].reset_index(drop=True).copy()
    print(f"[oxmech] HUST-8-cell subsample: {len(hust_sub)} rows, cells={sorted(hust_sub['battery_id'].unique())}")
    hust_sub_ext = add_scv_matd_viect_reformulated(hust_sub)

    r_before = eval_generic(model_base, med_base, base_cols, hust_sub, fcols)
    r_after = eval_generic(model_ext, med_ext, extended_cols, hust_sub_ext, fcols)
    print(f"[oxmech] HUST-8-cell subsample: R2 BEFORE (base reformulation)={r_before['r2']:.4f} "
          f"-> AFTER (extended reformulation)={r_after['r2']:.4f} (delta={r_after['r2']-r_before['r2']:+.4f})")

    rows = [
        {"comparison": "reformulation", "dataset": "Oxford (full, 8 cells, ORIGINAL Stage 5 result)",
         "r2_before": -2.694, "r2_after": 0.9533, "delta_r2": 0.9533 - (-2.694), "n_cells": 8},
        {"comparison": "reformulation", "dataset": "HUST (full, 77 cells, ORIGINAL Stage 5 result)",
         "r2_before": -0.152, "r2_after": 0.8000, "delta_r2": 0.8000 - (-0.152), "n_cells": 77},
        {"comparison": "reformulation", "dataset": "HUST (size-matched 8-cell subsample, THIS TEST)",
         "r2_before": r_before["r2"], "r2_after": r_after["r2"],
         "delta_r2": r_after["r2"] - r_before["r2"], "n_cells": len(hust_subsample_ids)},
    ]
    return rows


def world_model_intervention(hust_subsample_ids):
    print("\n=== Intervention 2: Stage 7.1 World Model on HUST-8-cell subsample ===")
    train, test = load_pool_train_test()
    stats = fit_norm_stats(train)

    model = WorldModel(embed_dim=32, window=WINDOW, patch_len=2, d_model=64)
    model.load_state_dict(torch.load(ROOT / "models" / "_experimental_world_model.pt"))
    model.eval()

    sub_pool = {}
    for cid in hust_subsample_ids:
        cycles = list(iterate_hust_cycles(cid))
        if len(cycles) < 5:
            print(f"[oxmech] WARNING: {cid} has <5 valid cycles, excluded from this subsample honestly")
            continue
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        sub_pool[cid] = (apply_channel_norm(X.astype(np.float32), stats), soh.astype(np.float32), rul.astype(np.float32))
    print(f"[oxmech] HUST-8-cell subsample loaded: {len(sub_pool)} usable cells, "
          f"{sum(len(v[1]) for v in sub_pool.values())} total cycles")

    sub_samples = make_windows(sub_pool, stride=WINDOW_STRIDE, max_per_battery=1000)
    print(f"[oxmech] {len(sub_samples)} windows from the HUST-8-cell subsample")
    if not sub_samples:
        print("[oxmech] NO windows available from this subsample (cells too short) - cannot evaluate, reporting honestly.")
        return []
    rows = eval_rollout(model, sub_samples, "HUST (size-matched 8-cell subsample, THIS TEST)")
    for r in rows:
        print(f"  horizon={r['horizon']}: model={r['rmse_model']:.4f} persistence={r['rmse_persistence_baseline']:.4f} "
              f"beats_persistence={r['model_beats_persistence']}")
    return rows


def main():
    t0 = time.time()
    print("=== Stage 7 closeout: Oxford-mechanism test (sample size vs. distributional similarity) ===")
    hust_sample = sample_hust_cells_stratified(n=8, seed=SEED)
    print(f"[oxmech] stratified HUST-8-cell sample (1 per batch, 8 of 10 batches, seed={SEED}): {hust_sample}")

    reform_rows = reformulation_intervention(hust_sample)
    reform_df = pd.DataFrame(reform_rows)
    reform_df.to_csv(OUT_DIR / "stage7_closeout_oxmech_reformulation.csv", index=False)
    print("\n--- reformulation intervention summary ---")
    print(reform_df.to_string(index=False))

    wm_rows = world_model_intervention(hust_sample)
    if wm_rows:
        wm_df = pd.DataFrame(wm_rows)
        wm_df.to_csv(OUT_DIR / "stage7_closeout_oxmech_world_model.csv", index=False)
        print("\n--- World Model intervention: HUST-8-cell subsample per-horizon result ---")
        print(wm_df.to_string(index=False))
        n_beats = sum(r["model_beats_persistence"] for r in wm_rows)
        print(f"\n[oxmech] World Model beats persistence on {n_beats}/{len(wm_rows)} horizons on the "
              f"HUST-8-cell subsample (full 77-cell HUST: 0/10 horizons; Oxford full 8-cell: 10/10 horizons)")

    print(f"\n[oxmech] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
