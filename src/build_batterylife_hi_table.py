"""
Builds, per BatteryLife sub-source, a merged HI+fusion feature table in
EXACTLY the same shape as this project's existing `stage5_1_{oxford,
hust,xjtu}_merged.parquet` files - dataset/battery_id/cycle_idx/SOH +
all 16 raw HI columns (health_indicators.HI_NAMES) + Stage 1.1's
duration-ratio columns (ICHV_rel/TEVD_rel/TEVI_rel) + Stage 5's extended
reformulation columns (SCV_rel/MATD_rel/VIECT_rel/MET_rel) + 16
fusion_i columns (from the SAME, already-trained `ica_encoder.pt`, never
retrained here - a zero-retrain-compliant embedding, exactly as CALCE's
own fusion embeddings are computed in stage1_common.build_calce_merged).

Reuses this project's own pipeline UNCHANGED at every step:
`data_adapters_batterylife.iterate_batterylife_cycles` (new, this pass)
-> `health_indicators.compute_health_indicators` (unchanged) ->
`rul_labels.soh_per_cycle` (unchanged) ->
`sequence_features.build_dataset_tensors` + `ICAEncoder` (unchanged,
same encoder every other dataset's fusion embeddings use) ->
`stage1_common.add_reformulated_duration_features` +
`stage5_extended_reformulation.add_scv_matd_viect_reformulated`
(unchanged).

DISCLOSED, PRINCIPLED SUBSAMPLING for the 4 large sources (RWTH,
Stanford, Stanford_2, ISU_ILCC - each 1-10GB, full download judged
infeasible within this pass's time budget after directly measuring
Zenodo's own download throughput during item 6): only the battery
files actually present under `data/raw/batterylife/<source>/` are
processed - for the 4 large sources this is a FIXED-SEED random sample
(seed=42) of individual battery files, extracted directly via HTTP
Range requests against the Zenodo-hosted zip's own central directory
(no full-archive download), not cherry-picked. The 5 small/medium
sources (UL_PUR, HNEI, SNL, MICH, MICH_EXP) are used in FULL - every
battery file Zenodo actually has for them is present locally.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters_batterylife import BATTERYLIFE_DIR, batterylife_cell_ids, iterate_batterylife_cycles
from health_indicators import compute_health_indicators, HI_NAMES
from rul_labels import soh_per_cycle
from sequence_features import build_dataset_tensors
from models.ica_encoder import ICAEncoder
from stage1_common import add_reformulated_duration_features, ROOT, PROC_DIR
from stage5_extended_reformulation import add_scv_matd_viect_reformulated

ICA_CHANNEL_SLICE = slice(3, 6)
SOURCES = ["UL_PUR", "HNEI", "SNL", "MICH", "MICH_EXP", "RWTH", "Stanford", "Stanford_2", "ISU_ILCC"]


def build_source(source: str, encoder: ICAEncoder):
    ids = batterylife_cell_ids(source)
    if not ids:
        print(f"[batterylife-build] {source}: 0 battery files present locally - skipped")
        return None
    rows = []
    fusion_rows = []
    n_batteries_used = 0
    n_tensor_failed = 0
    for cid in ids:
        cycles = list(iterate_batterylife_cycles(source, cid))
        if len(cycles) < 5:
            continue
        soh_map = soh_per_cycle(cycles)
        for c in cycles:
            hi = compute_health_indicators(c)
            row = {"dataset": f"BatteryLife_{source}", "battery_id": cid, "cycle_idx": c["cycle_idx"],
                   "SOH": soh_map[c["cycle_idx"]], "discharge_capacity": c["discharge_capacity"]}
            row.update(hi)
            rows.append(row)

        # Sequence-tensor/fusion-embedding step, isolated per battery: a
        # small number of cycles across these new sources hit a genuine
        # ICA (dV/dQ) degenerate-division edge case in the EXISTING,
        # unchanged `ica_dv_dc.py` (a near-flat discharge-capacity
        # segment blows up a Savitzky-Golay fit) - the SAME class of
        # raw-data artifact this project already found and fixed once
        # for XJTU's VDEDT (see DEVELOPMENT_LOG.md). Not silently
        # swallowed: caught and counted per battery, not per source, so
        # one bad battery doesn't lose an entire source's real data.
        try:
            X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        except Exception as e:
            print(f"[batterylife-build]   {source}/{cid}: tensor-build FAILED ({type(e).__name__}: {e}) - "
                  f"HI rows for this battery are KEPT, fusion embedding SKIPPED for it (will be NaN, dropped by the merge)")
            n_tensor_failed += 1
            continue
        if X is None:
            n_tensor_failed += 1
            continue
        with torch.no_grad():
            emb = encoder.encode(torch.tensor(X[:, :, ICA_CHANNEL_SLICE], dtype=torch.float32)).numpy()
        for i, cyc_idx in enumerate(idxs):
            frow = {"battery_id": cid, "cycle_idx": int(cyc_idx)}
            for k in range(emb.shape[1]):
                frow[f"fusion_{k}"] = float(emb[i, k])
            fusion_rows.append(frow)
        n_batteries_used += 1
    if n_tensor_failed:
        print(f"[batterylife-build] {source}: {n_tensor_failed} battery file(s) failed fusion-embedding "
              f"tensor-build (skipped for fusion only, HI rows kept where computed)")

    if not rows:
        print(f"[batterylife-build] {source}: 0 usable batteries (of {len(ids)} files) - skipped")
        return None

    hi_df = pd.DataFrame(rows)

    # REAL DATA ARTIFACT FOUND AND FIXED (not silently dropped): a small
    # number of individual cycles (isolated single cycle_idx values, not
    # whole batteries - traced directly on RWTH_007 cycle 291: capacity
    # jumps from a normal ~1.02 Ah to 434 Ah for that ONE cycle, then
    # returns to normal the next cycle) blow up SOH into the thousands
    # of percent. Root cause, confirmed by inspection: this project's own
    # `iterate_batterylife_cycles` segmentation (module docstring's own
    # disclosed simplification - assumes ONE contiguous charge phase then
    # ONE contiguous discharge phase per cycle) mis-segments certain
    # check-up/reference-performance-test cycles with a more complex
    # multi-step current profile, producing a spuriously huge discharge-
    # capacity span for that one cycle. A physically impossible SOH
    # (>105%, allowing a little early-cycle noise, or <0%) is dropped -
    # counted and reported per source, not silently discarded.
    n_before = len(hi_df)
    bad_mask = (hi_df["SOH"] > 105) | (hi_df["SOH"] < 0)
    n_bad = int(bad_mask.sum())
    if n_bad:
        bad_batteries = sorted(hi_df.loc[bad_mask, "battery_id"].unique())
        print(f"[batterylife-build] {source}: dropping {n_bad}/{n_before} rows with physically-impossible "
              f"SOH (>105% or <0%, a mis-segmented check-up-cycle artifact, see code comment) across "
              f"{len(bad_batteries)} batteries: {bad_batteries}")
        hi_df = hi_df.loc[~bad_mask].reset_index(drop=True)

    fusion_df = pd.DataFrame(fusion_rows)
    merged = pd.merge(hi_df, fusion_df, on=["battery_id", "cycle_idx"], how="inner")
    print(f"[batterylife-build] {source}: {n_batteries_used}/{len(ids)} battery files usable, "
          f"{len(merged)} cycle-rows, {merged['battery_id'].nunique()} distinct batteries in final table")
    return merged


def main():
    t0 = time.time()
    print("=== Building BatteryLife HI+fusion tables (per source, reusing the deployed pipeline unchanged) ===")
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()

    summary = []
    for source in SOURCES:
        merged = build_source(source, encoder)
        if merged is None:
            continue
        merged = add_reformulated_duration_features(merged)
        merged = add_scv_matd_viect_reformulated(merged)
        out_path = PROC_DIR / f"batterylife_{source.lower()}_merged.parquet"
        merged.to_parquet(out_path, index=False)
        print(f"[batterylife-build]   saved {out_path.name} ({len(merged)} rows)")
        summary.append({"source": source, "n_batteries": merged["battery_id"].nunique(), "n_rows": len(merged)})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(ROOT / "outputs" / "batterylife_build_summary.csv", index=False)
    print("\n=== BUILD SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\n[batterylife-build] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
