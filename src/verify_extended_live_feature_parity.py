"""
Dataset-aware routing, step 2 (verification GATE, per explicit
instruction - run standalone BEFORE wiring anything into live_inference
.py/app.py): does a NEW live, single-cycle feature-vector function for
Stage 5's extended reformulation (mixed ratio/delta convention)
reproduce the ORIGINAL BATCH-VALIDATED numbers (CALCE=0.740,
Oxford=0.953, HUST=0.800) when run row-by-row through the same
precomputed-data sources live_inference.py's own
predict_and_explain_precomputed already uses for these 3 datasets?

DESIGN NOTE, corrected after a real near-miss caught before running
anything: the first draft of this function mirrored live_inference.
build_reformulated_hi_vector's own fallback philosophy (always fill
missing/near-zero values from RAW-feature medians, never leave NaN).
That does NOT match how the extended model was actually TRAINED -
run_stage5_extended_reformulation_eval.py calls fit_xgb, which computes
column medians on the ALREADY-REFORMULATED (post ratio/delta) train
matrix and fills NaNs with THOSE, not raw-feature medians. For CALCE
specifically, whose MATD column is 100% NaN (no temperature channel,
documented throughout this project) and which reformulates MATD into a
DELTA, this is not a rare edge case - it affects EVERY CALCE row. Using
the wrong fallback statistic here would have produced numbers that look
plausible but don't actually match what the model was validated
against - exactly the "silently produces different numbers" risk this
whole verification step exists to catch. Fixed: this version mirrors
the BATCH pipeline's mechanics exactly - the per-feature reformulation
guard matches stage5_extended_reformulation.py's own (ratio: near-zero
baseline -> that battery's own median fallback, else leave NaN; delta:
non-finite baseline -> leave NaN, no guard), and any resulting NaN is
filled with `ext_medians` - the SAME statistic (post-reformulation,
TRAIN-pool column median) fit_xgb computed when this exact model was
trained, recomputed fresh here from the same pool/split.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from live_inference import ROOT, PROC_DIR, MODELS_DIR, BASELINE_CYCLE, _calce_fusion_embeddings
from models.ica_encoder import ICAEncoder
from stage1_common import load_nasa_mit_pool, battery_split_masks, canonical_feature_cols, fusion_cols
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols

EXTENDED_CANONICAL_REL = ["ICHV_rel", "SCV_rel", "VDEDT", "VIECT_rel", "MATD_rel", "MET_rel", "TEVD_rel", "TEVI_rel"]
RATIO_RAW_FEATURES = {"ICHV", "TEVD", "TEVI", "SCV", "MET"}
DELTA_RAW_FEATURES = {"MATD", "VIECT"}
TOLERANCE = 0.01

EXPECTED = {"CALCE": 0.740, "Oxford": 0.953, "HUST": 0.800}


def compute_ext_medians() -> dict:
    """TRAIN-pool (NASA+MIT), POST-reformulation column medians for
    EXTENDED_CANONICAL_REL - the SAME statistic fit_xgb computed when
    the extended model was actually trained, recomputed fresh here."""
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    hi_full_ext = add_scv_matd_viect_reformulated(hi_full)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    nasa_mit_ext = hi_full_ext[hi_full_ext["dataset"].isin(["NASA", "MIT"])]
    merged_ext = pd.merge(nasa_mit_ext, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    train_mask, _, _ = battery_split_masks(merged_ext)
    base_cols = canonical_feature_cols(reformulated=True)
    ext_cols = extended_canonical_feature_cols(base_cols)  # matches EXTENDED_CANONICAL_REL minus cycle_idx ordering
    X_train = merged_ext.loc[train_mask, ext_cols].to_numpy(dtype=float)
    medians = np.nanmedian(X_train, axis=0)
    return dict(zip(ext_cols, medians))


def build_extended_reformulated_hi_vector(his: dict, baseline_his: dict | None,
                                           battery_medians: dict, ext_medians: dict) -> np.ndarray:
    """Mirrors stage5_extended_reformulation.add_scv_matd_viect_reformulated's
    OWN per-feature guard exactly (ratio: near-zero baseline -> that
    battery's own median fallback; delta: non-finite baseline -> leave
    NaN), for the NEW features (SCV_rel/MET_rel/MATD_rel/VIECT_rel).
    ICHV_rel/TEVD_rel/TEVI_rel keep the EXISTING, already-shipped
    live_inference.build_reformulated_hi_vector fallback unchanged (not
    touched by this item, no reason to alter already-working production
    logic). VDEDT passed through raw, unreformulated, same as always.
    Any resulting NaN filled with `ext_medians` - the SAME post-
    reformulation train-pool statistic fit_xgb used at training time."""
    vec = np.full(len(EXTENDED_CANONICAL_REL), np.nan, dtype=float)
    for i, col in enumerate(EXTENDED_CANONICAL_REL):
        raw_feat = col[:-4] if col.endswith("_rel") else col
        raw_val = his.get(raw_feat)
        raw_val = float(raw_val) if raw_val is not None else float("nan")

        if raw_feat in ("ICHV", "TEVD", "TEVI"):
            # UNCHANGED existing production convention (live_inference.build_reformulated_hi_vector).
            # Confirmed via direct check: these 3 are never NaN in CALCE/Oxford/HUST, so this
            # fallback (0.0, matching the existing function's own literal default) never fires here.
            if raw_val != raw_val:
                raw_val = 0.0
            base_val = None
            if baseline_his is not None:
                base_val = baseline_his.get(raw_feat)
                if base_val is not None and (base_val != base_val or abs(base_val) < 1e-6):
                    base_val = None
            if base_val is None:
                base_val = raw_val if abs(raw_val) >= 1e-6 else 1.0
            vec[i] = raw_val / base_val if abs(base_val) >= 1e-6 else 1.0

        elif raw_feat in ("SCV", "MET"):
            # batch-faithful RATIO guard: near-zero baseline -> that battery's own median
            base_val = baseline_his.get(raw_feat) if baseline_his is not None else None
            if base_val is None or base_val != base_val or abs(base_val) < 1e-6:
                base_val = battery_medians.get(raw_feat)
            if base_val is not None and np.isfinite(base_val) and abs(base_val) >= 1e-6 and raw_val == raw_val:
                vec[i] = raw_val / base_val
            # else leave NaN -> filled by ext_medians below

        elif raw_feat in ("MATD", "VIECT"):
            # batch-faithful DELTA: non-finite baseline -> leave NaN, no guard
            base_val = baseline_his.get(raw_feat) if baseline_his is not None else None
            if base_val is not None and base_val == base_val and raw_val == raw_val:
                vec[i] = raw_val - base_val
            # else leave NaN -> filled by ext_medians below

        else:  # VDEDT, raw passthrough
            vec[i] = raw_val

    nan_mask = np.isnan(vec)
    if nan_mask.any():
        for i in np.where(nan_mask)[0]:
            vec[i] = ext_medians[EXTENDED_CANONICAL_REL[i]]
    return vec


def score_dataset(name: str, df: pd.DataFrame, ext_medians: dict, model, fusion_lookup) -> dict:
    preds, trues = [], []
    n_nan_filled = 0
    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_idx").reset_index(drop=True)
        base_row = g[g["cycle_idx"] == BASELINE_CYCLE]
        if not base_row.empty:
            baseline_his = base_row.iloc[0].to_dict()
        else:
            # Matches stage5_extended_reformulation._baseline_map's OWN
            # fallback exactly: that battery's own FIRST available cycle,
            # not None. Oxford has ZERO batteries with a real cycle_idx==10
            # row (confirmed directly - its own HI table samples every
            # ~20-30 cycles, not every cycle) - this fallback fires for
            # EVERY Oxford row, not a rare edge case, so getting it wrong
            # here was the dominant source of Oxford's gate mismatch.
            baseline_his = g.iloc[0].to_dict()
        battery_medians = {feat: g[feat].median() for feat in ("SCV", "MET")}
        for pos, row in g.iterrows():
            his = row.to_dict()
            true_soh = his.get("SOH")
            if true_soh is None or true_soh != true_soh:
                continue
            fusion_emb = fusion_lookup(bid, pos, row)
            if fusion_emb is None:
                continue
            hi_rel = build_extended_reformulated_hi_vector(his, baseline_his, battery_medians, ext_medians)
            hi_vector = np.concatenate([hi_rel, [float(row["cycle_idx"])]])
            feat_vector = np.concatenate([hi_vector, fusion_emb])
            pred = float(model.predict(feat_vector.reshape(1, -1))[0])
            preds.append(pred)
            trues.append(float(true_soh))
    preds, trues = np.array(preds), np.array(trues)
    r2 = float(r2_score(trues, preds))
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    return {"dataset": name, "r2": r2, "rmse": rmse, "n": len(trues)}


def main():
    t0 = time.time()
    print("=== Verification GATE: live single-cycle extended-feature pipeline vs. batch-validated numbers ===")

    model = XGBRegressor()
    model.load_model(str(MODELS_DIR / "_experimental_xgb_soh_fusion_extended_reformulation.json"))
    print("[verify] scoring model: models/_experimental_xgb_soh_fusion_extended_reformulation.json")

    print("[verify] computing ext_medians (TRAIN-pool, post-reformulation column medians - "
          "the same statistic fit_xgb used at training time)...")
    ext_medians = compute_ext_medians()
    print(f"[verify] ext_medians: {ext_medians}")

    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(MODELS_DIR / "ica_encoder.pt"))
    encoder.eval()

    results = []

    # --- CALCE ---
    calce_df = hi_df[hi_df["dataset"] == "CALCE"].reset_index(drop=True)
    calce_fusion_cache = {}

    def calce_fusion_lookup(bid, pos_in_group, row):
        if bid not in calce_fusion_cache:
            calce_fusion_cache[bid] = _calce_fusion_embeddings(bid, encoder, norm_stats)  # NOW FIXED (apply_channel_norm)
        emb = calce_fusion_cache[bid]
        if emb is None:
            return None
        sub = calce_df[calce_df["battery_id"] == bid].sort_values("cycle_idx").reset_index(drop=True)
        match_idx = sub.index[sub["cycle_idx"] == row["cycle_idx"]]
        if len(match_idx) == 0 or match_idx[0] >= len(emb):
            return None
        return emb[match_idx[0]]

    print("\n[verify] scoring CALCE row-by-row through the live single-cycle path...")
    results.append(score_dataset("CALCE", calce_df, ext_medians, model, calce_fusion_lookup))

    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"), ("HUST", "stage5_1_hust_merged.parquet")]:
        df = pd.read_parquet(PROC_DIR / fname)

        def fusion_lookup(bid, pos_in_group, row):
            return row[[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float)

        print(f"\n[verify] scoring {name} row-by-row through the live single-cycle path...")
        results.append(score_dataset(name, df, ext_medians, model, fusion_lookup))

    results_df = pd.DataFrame(results)
    print("\n=== RESULTS: live single-cycle pipeline vs. batch-validated numbers ===")
    all_pass = True
    for r in results:
        expected = EXPECTED[r["dataset"]]
        delta = r["r2"] - expected
        passed = abs(delta) <= TOLERANCE
        all_pass = all_pass and passed
        status = "MATCH" if passed else "MISMATCH"
        print(f"[verify] {r['dataset']}: live-pipeline R2={r['r2']:.4f} (n={r['n']}) vs. "
              f"batch-validated={expected:.3f} | delta={delta:+.4f} | tolerance=+/-{TOLERANCE} -> {status}")

    print(f"\n[verify] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")
    if all_pass:
        print("\n[verify] GATE PASSED: all 3 datasets match within tolerance - safe to wire into live routing.")
    else:
        print("\n[verify] GATE FAILED: at least one dataset does NOT match the batch-validated numbers. "
              "STOP - do not wire this into live_inference.py/app.py until the discrepancy is understood.")
    return all_pass


if __name__ == "__main__":
    main()
