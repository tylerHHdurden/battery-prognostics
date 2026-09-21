"""
CALCE embedding-normalization fix, step 2 verification: does the FIXED
_calce_fusion_embeddings() restore the base (currently-deployed) model's
CALCE zero-retrain R2 to what batch validation has always reported
(0.568, per this session's own audit_true_deployed_baseline.csv)?

Calls the ACTUAL production function (predict_and_explain_precomputed)
directly, in a loop over every CALCE cycle - not a reimplementation,
the real code path a live user hits.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_inference import load_resources, predict_and_explain_precomputed, PROC_DIR

EXPECTED = 0.568
TOLERANCE = 0.01


def main():
    t0 = time.time()
    print("=== Verifying the CALCE embedding fix restores the base model's own batch-validated R2 ===")
    res = load_resources()
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    calce = hi_df[hi_df["dataset"] == "CALCE"]

    preds, trues = [], []
    n_errors = 0
    for bid, g in calce.groupby("battery_id"):
        for cyc in sorted(g["cycle_idx"].unique()):
            ctx = predict_and_explain_precomputed("CALCE", bid, int(cyc), res)
            if "error" in ctx:
                n_errors += 1
                continue
            if ctx["true_soh"] is None:
                continue
            preds.append(ctx["soh_pred"])
            trues.append(ctx["true_soh"])

    preds, trues = np.array(preds), np.array(trues)
    r2 = float(r2_score(trues, preds))
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    print(f"[verify-fix] CALCE via ACTUAL predict_and_explain_precomputed (base model): "
          f"R2={r2:.4f} RMSE={rmse:.4f} n={len(preds)} (errors={n_errors})")

    delta = r2 - EXPECTED
    status = "MATCH" if abs(delta) <= TOLERANCE else "MISMATCH"
    print(f"[verify-fix] vs. batch-validated base model R2={EXPECTED:.3f} | delta={delta:+.4f} | "
          f"tolerance=+/-{TOLERANCE} -> {status}")
    print(f"\n[verify-fix] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
