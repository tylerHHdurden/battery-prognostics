"""Precompute the deployed joint-model RUL for every NASA and MIT browse battery/cycle, offline, from the raw curves (which Streamlit Cloud does not have).
Mirrors the RUL part of live_inference.predict_and_explain exactly (same HI vector, same joint-HI z-scoring, same channel normalisation, same joint_fusion forward),
but batched and without SHAP. Output: data/processed/precomputed_rul_nasa_mit.parquet (dataset, battery_id, cycle_idx, rul_pred). The app only looks the value up and
shows it for NASA/MIT batteries that the trust check does not flag. No model file is modified. Run: python src/precompute_rul_browse.py"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import live_inference as li  # noqa: E402
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles, mit_cell_ids  # noqa: E402
from health_indicators import compute_health_indicators  # noqa: E402
from sequence_features import get_cycle_tensor, apply_channel_norm  # noqa: E402
from stage1_common import BASELINE_CYCLE  # noqa: E402

OUT = ROOT / "data" / "processed" / "precomputed_rul_nasa_mit.parquet"
CHUNK = 256


def load_cycles(dataset, bid, mit_subset):
    if dataset == "NASA":
        return list(iterate_nasa_cycles(bid))
    batch_file, cell_index = mit_subset[bid]
    return list(iterate_mit_cycles(batch_file, cell_index))


def rul_inputs(cycle, baseline_his, res):
    """Same steps as predict_and_explain lines for the RUL branch; returns (x_norm[200,6], hi_rel_z[8]) or None."""
    his = compute_health_indicators(cycle)
    hi_rel_vector = li.build_reformulated_hi_vector(his, res["train_medians"], baseline_his)
    jn = res["joint_hi_norm"]
    hi_rel_z = (hi_rel_vector - np.array(jn["hi_mean"])) / np.array(jn["hi_std"])
    x_raw = get_cycle_tensor(cycle, n_bins=200)
    if x_raw is None:
        return None
    x_norm = apply_channel_norm(x_raw[None].astype(np.float32), res["norm_stats"])[0]
    return x_norm, hi_rel_z


def main():
    t0 = time.time()
    res = li.load_resources()
    jf = res["joint_fusion"]
    jf.eval()
    rul_mean, rul_std = res["constants"]["rul_mean"], res["constants"]["rul_std"]
    mit_subset = {gid: (bf, ci) for bf, ci, gid in mit_cell_ids()}  # every MIT cell in the raw batches (the browse pool has 31, mit_subset.json only 28)
    rows, skipped = [], []
    for dataset in ("NASA", "MIT"):
        avail = li.available_precomputed_cycles(dataset)
        for bi, (bid, cyc_list) in enumerate(sorted(avail.items())):
            tb = time.time()
            cycles = load_cycles(dataset, bid, mit_subset)
            by_idx = {c["cycle_idx"]: c for c in cycles}
            base_cycle = by_idx.get(BASELINE_CYCLE, cycles[0])
            baseline_his = compute_health_indicators(base_cycle)
            X, Z, keep = [], [], []
            for ci in cyc_list:
                c = by_idx.get(int(ci))
                if c is None:
                    skipped.append((dataset, bid, int(ci), "cycle not in raw data")); continue
                r = rul_inputs(c, baseline_his, res)
                if r is None:
                    skipped.append((dataset, bid, int(ci), "tensor failed")); continue
                X.append(r[0]); Z.append(r[1]); keep.append(int(ci))
            preds = []
            with torch.no_grad():
                for s in range(0, len(X), CHUNK):
                    xb = torch.tensor(np.stack(X[s:s + CHUNK]), dtype=torch.float32)
                    zb = torch.tensor(np.stack(Z[s:s + CHUNK]), dtype=torch.float32)
                    _, rz = jf(xb, zb)
                    preds.extend((rz.detach().numpy().reshape(-1) * rul_std + rul_mean).tolist())
            rows += [(dataset, bid, ci, float(p)) for ci, p in zip(keep, preds)]
            print(f"[rul] {dataset} {bid}: {len(keep)}/{len(cyc_list)} cycles ({time.time()-tb:.0f}s, total {(time.time()-t0)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows, columns=["dataset", "battery_id", "cycle_idx", "rul_pred"])
    df["cycle_idx"] = df["cycle_idx"].astype("int32"); df["rul_pred"] = df["rul_pred"].astype("float32")
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"[rul] wrote {OUT.name}: {len(df)} rows, {OUT.stat().st_size/1e3:.0f} KB; skipped {len(skipped)}; total {(time.time()-t0)/60:.1f} min", flush=True)
    if skipped:
        pd.DataFrame(skipped, columns=["dataset", "battery_id", "cycle_idx", "reason"]).to_csv(ROOT / "outputs" / "precomputed_rul_skipped.csv", index=False)


if __name__ == "__main__":
    main()
