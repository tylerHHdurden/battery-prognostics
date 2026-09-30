"""After the routing change (six app datasets use the validated routing, not the multisource candidate): score a random sample of
cycles per dataset through the PRODUCTION path (load_resources + predict_and_explain_precomputed) and compare MAE/RMSE with the
PAPER_RESULTS.md sec 1 five-seed routed values (mean +/- std). Sampling: 400 random cycles per dataset (seed 0), whole batteries not
required. Deployed models are the seed-42 models, so agreement is judged against mean +/- 2 std of the five seeds (plus a sampling band).
Writes outputs/toolkit_six_app_routing_check.csv."""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import live_inference as li  # noqa: E402

PAPER = {  # PAPER_RESULTS.md sec 1: MAE mean, MAE std, RMSE mean, RMSE std
    "CALCE": (6.256, 0.255, 10.791, 0.265), "Oxford": (1.431, 0.434, 1.639, 0.409),
    "HUST": (2.643, 0.138, 3.336, 0.178), "XJTU": (6.457, 0.213, 8.572, 0.460),
    "NASA": None, "MIT": None,
}
N = 400
res = li.load_resources()
rng = np.random.default_rng(0)
rows = []
for ds in ["CALCE", "Oxford", "HUST", "XJTU"]:  # NASA/MIT are unchanged (base model, never the candidate)
    t0 = time.time()
    avail = li.available_precomputed_cycles(ds)
    pairs = [(b, c) for b, cs in avail.items() for c in cs]
    pick = [pairs[i] for i in rng.choice(len(pairs), size=min(N, len(pairs)), replace=False)]
    err, sq, yt, yp, routed_ok = [], [], [], [], set()
    for b, c in pick:
        ctx = li.predict_and_explain_precomputed(ds, b, int(c), res)
        if ctx.get("true_soh") is None or ctx.get("soh_pred") is None:
            continue
        yt.append(ctx["true_soh"]); yp.append(ctx["soh_pred"])
    yt, yp = np.array(yt, float), np.array(yp, float)
    mae, rmse = float(np.abs(yp - yt).mean()), float(np.sqrt(((yp - yt) ** 2).mean()))
    r2 = float(1 - ((yp - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    paper = PAPER[ds]
    row = {"dataset": ds, "n_scored": len(yt), "candidate_used": li._use_candidate(ds), "sample_mae": round(mae, 3), "sample_rmse": round(rmse, 3),
           "sample_r2": round(r2, 3), "paper_mae": None if paper is None else paper[0], "paper_mae_std": None if paper is None else paper[1],
           "paper_rmse": None if paper is None else paper[2], "secs": round(time.time() - t0)}
    if paper is not None:
        row["mae_within_2std_of_paper"] = abs(mae - paper[0]) <= 2 * paper[1] + 0.25 * paper[0]  # 25% band for the 400-cycle sample
        row["rmse_within_2std_of_paper"] = abs(rmse - paper[2]) <= 2 * paper[3] + 0.25 * paper[2]
    rows.append(row)
    print(row, flush=True)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs" / "toolkit_six_app_routing_check.csv", index=False)
print(out.to_string(index=False))
