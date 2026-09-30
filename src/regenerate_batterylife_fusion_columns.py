"""
Step 1 (2026-09-30, revised per user decision): regenerate the BatteryLife parquets' `fusion_*` columns as CORRECTED
OLD-encoder embeddings through the FIXED builder path (build_batterylife_hi_table.encode_tensors: apply_channel_norm ->
old encoder), so they are mutually consistent with the Stage-5.1 Oxford/HUST/XJTU parquets, CALCE, and NASA/MIT
fusion_embeddings.csv, and with the old models that consume them. Candidate-encoder embeddings stay in the sidecar
fusion_embeddings_multisource.csv.

RACE SAFETY: writes ONLY new files (`batterylife_<src>_merged_OLDENC_CORRECTED.parquet` + `.meta.json`, temp file then
atomic rename). It reads - never writes - the canonical parquets the background rerun queue is reading. Installing the
corrected files under the canonical names is a separate step (swap_in_corrected_batterylife_parquets.py) that must only
run after the queue has drained.

Checks: (a) HI rows all retained; (b) cross-check against the independently-built
old_encoder_embeddings_batterylife_corrected.parquet (max diff < 1e-6); (c) as requested, no fusion value exceeds 10x
the 99.9th percentile of the OLD encoder's NASA+MIT training rows, in ANY per-source parquet (BatteryLife corrected files
and the Stage-5.1 Oxford/HUST/XJTU parquets); NaN embeddings (non-finite dVdQ cells - the old path never sanitized, as
deployed) are counted and reported, not hidden.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from build_batterylife_hi_table import load_encoder_and_stats, encode_tensors
from data_adapters_batterylife import iterate_batterylife_cycles
from sequence_features import build_dataset_tensors
from split_utils import battery_level_split
import encoder_provenance as ep

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FC = [f"fusion_{i}" for i in range(16)]
RAW = {"ul_pur": "UL_PUR", "hnei": "HNEI", "snl": "SNL", "mich": "MICH", "mich_exp": "MICH_EXP", "rwth": "RWTH",
       "stanford": "Stanford", "stanford_2": "Stanford_2", "isu_ilcc": "ISU_ILCC", "tongji": "Tongji"}


def old_train_p999() -> np.ndarray:
    old = pd.read_csv(PROC / "fusion_embeddings.csv")
    old["battery_id"] = old["battery_id"].astype(str)
    dataset_of = dict(zip(old["battery_id"], old["dataset"]))
    tr_ids, _ = battery_level_split(sorted(set(old["battery_id"])), test_every=5, dataset_of=dataset_of)
    tr = old["battery_id"].isin(set(tr_ids)).to_numpy()
    return np.percentile(np.abs(old.loc[tr, FC].to_numpy(float)), 99.9, axis=0)


def main():
    t0 = time.time()
    encoder, stats, sanitize = load_encoder_and_stats("old")
    thr = 10 * old_train_p999()
    independent = pd.read_parquet(PROC / "old_encoder_embeddings_batterylife_corrected.parquet")
    independent["battery_id"] = independent["battery_id"].astype(str)

    rows = []
    only = sys.argv[1:]  # optional resume: only these sources (the run that was killed had finished all but tongji)
    for src, raw in RAW.items():
        if only and src not in only:
            continue
        t_s = time.time()
        canon = PROC / f"batterylife_{src}_merged.parquet"          # READ ONLY
        hi = pd.read_parquet(canon).drop(columns=FC)
        hi["battery_id"] = hi["battery_id"].astype(str)
        frames = []
        for cid in sorted(hi["battery_id"].unique()):
            cycles = list(iterate_batterylife_cycles(raw, cid))
            if len(cycles) < 5:
                continue
            X, soh, rul, idxs, cens = build_dataset_tensors(cycles)
            if X is None:
                continue
            df = pd.DataFrame(encode_tensors(X, encoder, stats, sanitize), columns=FC)
            df.insert(0, "cycle_idx", np.asarray(idxs).astype(int))
            df.insert(0, "battery_id", cid)
            frames.append(df)
        new = pd.merge(hi, pd.concat(frames, ignore_index=True), on=["battery_id", "cycle_idx"], how="inner")
        assert len(new) == len(hi), f"{src}: lost {len(hi) - len(new)} HI rows"

        ind = independent[independent["source"] == src][["battery_id", "cycle_idx"] + FC]
        m = pd.merge(new[["battery_id", "cycle_idx"] + FC], ind, on=["battery_id", "cycle_idx"], suffixes=("", "_ind"), how="left")
        both = m[FC].notna().all(axis=1) & m[[c + "_ind" for c in FC]].notna().all(axis=1)
        diff = float(np.abs(m.loc[both, FC].to_numpy(float) - m.loc[both, [c + "_ind" for c in FC]].to_numpy(float)).max())
        assert diff < 1e-6, f"{src}: differs from the independently built corrected embeddings by {diff:.3g}"
        vals = new[FC].to_numpy(float)
        exceed = int((np.abs(vals) > thr).any(axis=1).sum())
        assert exceed == 0, f"{src}: {exceed} rows exceed 10x the old encoder's train p99.9"

        out = PROC / f"batterylife_{src}_merged_OLDENC_CORRECTED.parquet"
        tmp = out.with_suffix(".parquet.tmp")
        table = pa.Table.from_pandas(new, preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"encoder_tag": ep.OLD.encode(),
                                               b"encoder_md5": ep.encoder_md5(ep.OLD).encode(),
                                               b"note": b"corrected old-encoder embeddings (apply_channel_norm before encoder)"})
        pq.write_table(table, tmp)
        os.replace(tmp, out)                                         # atomic
        ep.write_meta(out, "embeddings", ep.OLD, note="regenerated via fixed build_batterylife_hi_table.encode_tensors")
        rows.append({"source": src, "rows": len(new), "max_abs_fusion": float(np.nanmax(np.abs(vals))), "rows_exceeding_10x_p999": exceed,
                     "rows_nan_fusion": int(np.isnan(vals).any(axis=1).sum()), "max_diff_vs_independent_build": diff,
                     "secs": round(time.time() - t_s)})
        print(f"[regen] {src}: {len(new)} rows | max|fusion| {rows[-1]['max_abs_fusion']:.2f} | >10x p99.9: {exceed} | NaN rows "
              f"{rows[-1]['rows_nan_fusion']} | diff vs independent build {diff:.1e} | {rows[-1]['secs']}s (total {(time.time()-t0)/60:.1f} min)", flush=True)

    for name in ["oxford", "hust", "xjtu"]:
        pqf = pd.read_parquet(PROC / f"stage5_1_{name}_merged.parquet")
        mx = np.abs(pqf[FC].to_numpy(float))
        exceed = int((mx > thr).any(axis=1).sum())
        assert exceed == 0, f"stage5_1 {name}: {exceed} rows exceed 10x train p99.9"
        rows.append({"source": f"stage5_1_{name}", "rows": len(pqf), "max_abs_fusion": float(np.nanmax(mx)), "rows_exceeding_10x_p999": exceed,
                     "rows_nan_fusion": int(np.isnan(mx).any(axis=1).sum())})
        print(f"[regen] stage5_1 {name}: max|fusion| {np.nanmax(mx):.2f}, rows > 10x p99.9: {exceed}", flush=True)
    suffix = "_" + "_".join(only) if only else ""
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / f"toolkit_batterylife_fusion_regeneration_summary{suffix}.csv", index=False)
    print(f"[regen] DONE in {(time.time()-t0)/60:.1f} min - all assertions passed", flush=True)


if __name__ == "__main__":
    main()
