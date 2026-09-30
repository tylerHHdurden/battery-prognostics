"""
Install the corrected BatteryLife parquets under their canonical names (step 1's second half). MUST NOT run while the
step-2 rerun queue is still reading `batterylife_<src>_merged.parquet` - it refuses unless the queue log says ALL DONE
(or --force). The original raw-X parquets are kept, renamed `..._RAWX_UNNORMALIZED.parquet`, with a sidecar marking them
DEFECTIVE so encoder-provenance loaders refuse them.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_provenance as ep

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
SRC = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth", "stanford", "stanford_2", "isu_ilcc", "tongji"]


def main(force=False):
    summary = ROOT / "outputs" / "rerun_queue" / "summary.txt"
    if not force and not (summary.exists() and "ALL DONE" in summary.read_text()):
        sys.exit("refusing: the rerun queue has not finished (no 'ALL DONE' in outputs/rerun_queue/summary.txt); use --force to override")
    for s in SRC:
        canon = PROC / f"batterylife_{s}_merged.parquet"
        corrected = PROC / f"batterylife_{s}_merged_OLDENC_CORRECTED.parquet"
        defective = PROC / f"batterylife_{s}_merged_RAWX_UNNORMALIZED.parquet"
        assert corrected.exists(), f"missing {corrected.name} - run regenerate_batterylife_fusion_columns.py first"
        assert not defective.exists(), f"{defective.name} already exists - swap already done?"
        os.replace(canon, defective)
        ep.sidecar(defective).write_text(json.dumps({
            "role": "embeddings", "encoder_tag": "rawx_unnormalized_DEFECTIVE",
            "note": "fusion_* computed from RAW un-normalized tensors (build_batterylife_hi_table.py before 2026-09-30); "
                    "values reach 1e7-1e13. Kept for the record only - never load."}, indent=1))
        os.replace(corrected, canon)
        ep.write_meta(canon, "embeddings", ep.OLD, note="corrected old-encoder embeddings (installed by swap_in_corrected_batterylife_parquets.py)")
        os.replace(ep.sidecar(corrected), ep.sidecar(canon)) if ep.sidecar(corrected).exists() else None
        ep.write_meta(canon, "embeddings", ep.OLD, note="corrected old-encoder embeddings")
        print(f"[swap] {s}: canonical parquet now holds corrected old-encoder embeddings; defective original kept as {defective.name}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
