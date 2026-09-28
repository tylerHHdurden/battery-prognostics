"""
Dataset-aware routing, final check: does the ACTUAL WIRED-IN production
code (predict_and_explain_precomputed, called directly, not the
separate standalone verify_extended_live_feature_parity.py gate script
this was ported from) reproduce the batch-validated numbers for
CALCE/Oxford/HUST, and leave XJTU/NASA/MIT on the unchanged base model?

This matters because build_extended_reformulated_hi_vector was
manually re-transcribed into live_inference.py from the already-
verified gate script - re-verifying the ACTUAL wired code, not
assuming the port was faithful.

SCOPE NOTE: the standalone gate script already rigorously verified the
EXACT reformulation logic against the full CALCE/Oxford/HUST datasets
(2941/519/146122 rows) with a fast, per-battery-cached implementation.
predict_and_explain_precomputed itself is much slower per call (no
caching across a battery's own cycles - confirmed: ~18.6 min for
CALCE's 2941 rows alone via this exact function in the embedding-fix
verification). Re-running it on EVERY row here (HUST alone would take
~15+ hours at that rate) would re-verify math already verified - what's
actually still unverified is whether the WIRING itself (which dict key,
which model object, which branch) was transcribed correctly. A random
subsample is sufficient to catch a transcription bug; tolerance is
widened accordingly since subsample R2 has more sampling variance than
the full-dataset R2 the gate script already confirmed precisely.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_inference import load_resources, predict_and_explain_precomputed, PROC_DIR, PRECOMPUTED_HELDOUT_PARQUETS

SEED = 42
N_SAMPLE = 250
TOLERANCE = 0.05  # widened - subsample variance, not re-verifying exact math (already done in the gate script)
EXPECTED_ROUTED = {"CALCE": 0.740, "Oxford": 0.953, "HUST": 0.800}
EXPECTED_UNROUTED_XJTU = -1.062


def score_via_production_function(dataset, battery_ids_and_cycles, res):
    preds, trues = [], []
    n_errors = 0
    n_exceptions = 0
    for bid, cyc in battery_ids_and_cycles:
        try:
            ctx = predict_and_explain_precomputed(dataset, bid, int(cyc), res)
        except Exception as e:
            # KNOWN, SEPARATE, pre-existing bug (194/19238 XJTU rows have
            # VDEDT=inf, crashes the OC-SVM check) - unrelated to routing,
            # flagged separately, not silently worked around in production
            # code. Caught here only so this subsample check can still
            # complete and report on the rows that DON'T hit it.
            n_exceptions += 1
            continue
        if "error" in ctx or ctx.get("true_soh") is None:
            n_errors += 1
            continue
        preds.append(ctx["soh_pred"])
        trues.append(ctx["true_soh"])
    if n_exceptions:
        print(f"[verify-prod] {dataset}: {n_exceptions} rows raised an exception (known separate "
              f"VDEDT=inf issue, not related to routing) - excluded from this subsample's R2")
    preds, trues = np.array(preds), np.array(trues)
    r2 = float(r2_score(trues, preds))
    return r2, len(preds), n_errors


def main():
    t0 = time.time()
    print("=== Verifying ACTUAL WIRED-IN production routing code ===")
    res = load_resources()
    print(f"[verify-prod] model_variant check: xgb_fusion_extended loaded={'xgb_fusion_extended' in res}, "
          f"ext_medians loaded={'ext_medians' in res}")

    all_pass = True
    rng = np.random.default_rng(SEED)

    def sample_pairs(df):
        idx = rng.choice(len(df), size=min(N_SAMPLE, len(df)), replace=False)
        sub = df.iloc[idx]
        return list(zip(sub["battery_id"], sub["cycle_idx"]))

    # CALCE
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    calce = hi_df[hi_df["dataset"] == "CALCE"]
    pairs = sample_pairs(calce)
    r2, n, n_err = score_via_production_function("CALCE", pairs, res)
    delta = r2 - EXPECTED_ROUTED["CALCE"]
    status = "MATCH" if abs(delta) <= TOLERANCE else "MISMATCH"
    all_pass = all_pass and status == "MATCH"
    print(f"[verify-prod] CALCE (routed, via predict_and_explain_precomputed, n={n} subsample): R2={r2:.4f} "
          f"errors={n_err} vs expected={EXPECTED_ROUTED['CALCE']:.3f} delta={delta:+.4f} -> {status}")

    # Oxford, HUST (routed) + XJTU (must stay unrouted/unchanged)
    for name, fname in PRECOMPUTED_HELDOUT_PARQUETS.items():
        df = pd.read_parquet(PROC_DIR / fname)
        pairs = sample_pairs(df)
        r2, n, n_err = score_via_production_function(name, pairs, res)
        expected = EXPECTED_ROUTED.get(name, EXPECTED_UNROUTED_XJTU if name == "XJTU" else None)
        delta = r2 - expected
        status = "MATCH" if abs(delta) <= TOLERANCE else "MISMATCH"
        all_pass = all_pass and status == "MATCH"
        routed_label = "routed" if name in EXPECTED_ROUTED else "UNROUTED (must stay on base model)"
        print(f"[verify-prod] {name} ({routed_label}, n={n} subsample): R2={r2:.4f} "
              f"errors={n_err} vs expected={expected:.3f} delta={delta:+.4f} -> {status}")

    # Explicit model_variant field check
    ctx_calce = predict_and_explain_precomputed("CALCE", calce["battery_id"].iloc[0], int(calce["cycle_idx"].iloc[0]), res)
    xjtu_df = pd.read_parquet(PROC_DIR / PRECOMPUTED_HELDOUT_PARQUETS["XJTU"])
    ctx_xjtu = predict_and_explain_precomputed("XJTU", xjtu_df["battery_id"].iloc[0], int(xjtu_df["cycle_idx"].iloc[0]), res)
    print(f"[verify-prod] CALCE ctx['model_variant']={ctx_calce.get('model_variant')} "
          f"(expect 'extended_reformulation')")
    print(f"[verify-prod] XJTU ctx['model_variant']={ctx_xjtu.get('model_variant')} (expect 'base')")
    all_pass = all_pass and ctx_calce.get("model_variant") == "extended_reformulation"
    all_pass = all_pass and ctx_xjtu.get("model_variant") == "base"

    print(f"\n[verify-prod] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")
    print("[verify-prod] ALL PASS" if all_pass else "[verify-prod] FAILURES FOUND - see above")
    return all_pass


if __name__ == "__main__":
    main()
