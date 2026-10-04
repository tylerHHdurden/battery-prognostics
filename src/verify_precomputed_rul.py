"""Check the precomputed NASA/MIT RUL (data/processed/precomputed_rul_nasa_mit.parquet) against the FULL local pipeline (raw curve -> live_inference.predict_and_explain)
for random battery/cycle pairs. Reports the max difference in the shown (rounded) RUL, in the interval ends, and in the unrounded value vs a single-sample forward pass
(batched vs single-sample numerics). Writes outputs/precomputed_rul_verification.csv. Run: python src/verify_precomputed_rul.py [n_pairs] [seed]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import live_inference as li  # noqa: E402
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles, mit_cell_ids  # noqa: E402
from health_indicators import compute_health_indicators  # noqa: E402
from stage1_common import BASELINE_CYCLE  # noqa: E402
from precompute_rul_browse import rul_inputs  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
res = li.load_resources()
pre = pd.read_parquet(ROOT / "data" / "processed" / "precomputed_rul_nasa_mit.parquet")
mit_map = {gid: (bf, ci) for bf, ci, gid in mit_cell_ids()}
rng = np.random.default_rng(SEED)
nasa, mit = pre[pre.dataset == "NASA"], pre[pre.dataset == "MIT"]
n_nasa = min(4, N // 3)   # stratified: NASA is only 3% of the rows, so force a few
picks = pd.concat([nasa.iloc[rng.choice(len(nasa), size=n_nasa, replace=False)], mit.iloc[rng.choice(len(mit), size=N - n_nasa, replace=False)]])
rows = []
cache = {}
for _, r in picks.iterrows():
    ds, bid, cyc = r["dataset"], r["battery_id"], int(r["cycle_idx"])
    if (ds, bid) not in cache:
        cache[(ds, bid)] = list(iterate_nasa_cycles(bid)) if ds == "NASA" else list(iterate_mit_cycles(*mit_map[bid]))
    cycles = cache[(ds, bid)]
    cycle = next(c for c in cycles if c["cycle_idx"] == cyc)
    base = compute_health_indicators(next((c for c in cycles if c["cycle_idx"] == BASELINE_CYCLE), cycles[0]))
    full = li.predict_and_explain(cycle, res, baseline_his=base, dataset=ds, battery_id=bid)   # the full local pipeline (SHAP, VLSTM, ... included)
    x_norm, hi_z = rul_inputs(cycle, base, res)
    with torch.no_grad():
        _, rz = res["joint_fusion"](torch.tensor(x_norm[None]), torch.tensor(hi_z[None], dtype=torch.float32))
    single = float(rz.numpy().reshape(-1)[0]) * res["constants"]["rul_std"] + res["constants"]["rul_mean"]
    half = res["constants"]["rul_conformal_half_width"]
    p = float(r["rul_pred"])
    rows.append({"dataset": ds, "battery_id": bid, "cycle_idx": cyc, "precomputed_float": round(p, 4), "single_sample_float": round(single, 4),
                 "float_diff_vs_single": abs(p - single), "full_pipeline_rul": full["rul_pred"], "precomputed_shown": max(0, round(p)),
                 "shown_diff": abs(full["rul_pred"] - max(0, round(p))),
                 "interval_lo_diff": abs(full["rul_conformal_lo"] - max(0, round(p - half))), "interval_hi_diff": abs(full["rul_conformal_hi"] - round(p + half))})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "outputs" / "precomputed_rul_verification.csv", index=False)
pd.set_option("display.width", 220)
print(df[["dataset", "battery_id", "cycle_idx", "precomputed_float", "single_sample_float", "full_pipeline_rul", "precomputed_shown", "shown_diff"]].to_string(index=False))
print(f"\npairs: {len(df)} | max |shown RUL diff| (cycles): {df.shown_diff.max()} | max interval-end diff: {max(df.interval_lo_diff.max(), df.interval_hi_diff.max())} | "
      f"max |float diff| precomputed vs single-sample forward: {df.float_diff_vs_single.max():.2e}")
