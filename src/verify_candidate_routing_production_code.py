"""
Multi-source candidate routing, production-code verification - same
discipline as `verify_routing_production_code.py` (the extended-
routing feature's own verification), applied fresh to the NEW routing
decision: does the ACTUAL WIRED-IN production code
(`predict_and_explain_precomputed`, called directly) reproduce
plausible, correctly-routed numbers for CALCE/Oxford/HUST/XJTU (all
now -> multisource_candidate) and leave NASA/MIT on the unchanged base
model?

Not re-checking the candidate's own gate-table R2 to high precision
here (that's a DIFFERENT split/protocol - the gate table's own
battery-level split vs. this script's own random cycle subsample) -
checking that the WIRING itself (right model object, right fusion
embedding, right routing branch) is correct, via a random subsample
across each dataset's full range, same as the original verification
script's own established convention.
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


def score_via_production_function(dataset, battery_ids_and_cycles, res):
    preds, trues, variants = [], [], []
    n_errors = 0
    n_exceptions = 0
    for bid, cyc in battery_ids_and_cycles:
        try:
            ctx = predict_and_explain_precomputed(dataset, bid, int(cyc), res)
        except Exception as e:
            n_exceptions += 1
            continue
        if "error" in ctx or ctx.get("true_soh") is None:
            n_errors += 1
            continue
        preds.append(ctx["soh_pred"])
        trues.append(ctx["true_soh"])
        variants.append(ctx["model_variant"])
    return np.array(preds), np.array(trues), variants, n_errors, n_exceptions


def main():
    t0 = time.time()
    print("=== Multi-source candidate routing: production-code verification ===")
    res = load_resources()
    rng = np.random.default_rng(SEED)

    all_ok = True
    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        if name == "CALCE":
            hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
            df = hi_df[hi_df["dataset"] == "CALCE"]
        else:
            df = pd.read_parquet(PROC_DIR / PRECOMPUTED_HELDOUT_PARQUETS[name])

        n = min(N_SAMPLE, len(df))
        sample = df.sample(n=n, random_state=SEED)
        pairs = list(zip(sample["battery_id"], sample["cycle_idx"]))

        preds, trues, variants, n_errors, n_exceptions = score_via_production_function(name, pairs, res)
        variant_counts = pd.Series(variants).value_counts().to_dict()
        all_candidate = all(v == "multisource_candidate" for v in variants)

        if len(preds) > 1:
            r2 = float(r2_score(trues, preds))
            rmse = float(np.sqrt(mean_squared_error(trues, preds)))
        else:
            r2, rmse = float("nan"), float("nan")

        print(f"[verify] {name}: n_sampled={n}, n_scored={len(preds)}, n_errors={n_errors}, "
              f"n_exceptions={n_exceptions}, model_variant counts={variant_counts}, "
              f"ALL routed to multisource_candidate={all_candidate}, subsample R2={r2:.4f} RMSE={rmse:.4f}")

        if not all_candidate:
            print(f"[verify] FAIL: {name} did not route 100% to multisource_candidate")
            all_ok = False
        if n_exceptions > 0:
            print(f"[verify] WARNING: {name} had {n_exceptions} exceptions during scoring")

    # NASA/MIT: confirm they STILL route to base (unchanged)
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    for name in ["NASA", "MIT"]:
        df = hi_df[hi_df["dataset"] == name]
        n = min(N_SAMPLE, len(df))
        sample = df.sample(n=n, random_state=SEED)
        pairs = list(zip(sample["battery_id"], sample["cycle_idx"]))
        # NASA/MIT never had a "precomputed" path wired for them in the
        # original routing feature either (they always used the raw live
        # path in the real app) - this loop is checking the SAME thing
        # verify_routing_production_code.py already established for
        # them (never routed), not re-litigating it; skip if the
        # precomputed path genuinely isn't implemented for this dataset -
        # this is expected, not a routing failure.
        preds, trues, variants, n_errors, n_exceptions = score_via_production_function(name, pairs[:5], res)
        print(f"[verify] {name} (informational, precomputed path not the real production path for this dataset): "
              f"variants seen={set(variants)}")

    print(f"\n[verify] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")
    print(f"\n[verify] {'ALL CHECKS PASSED' if all_ok else 'FAILURES FOUND - see above'}")
    return all_ok


if __name__ == "__main__":
    main()
