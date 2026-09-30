"""Hash bookkeeping and verification around the corrected-parquet swap.
  python src/verify_swapped_parquets.py pre    -> record sha256 of every *_OLDENC_CORRECTED.parquet (before the swap)
  python src/verify_swapped_parquets.py post   -> after the swap: canonical files must (1) hash-equal the recorded corrected files,
                                                 (2) carry a sidecar with encoder_tag old_v1 and the old encoder's md5, (3) hold
                                                 bounded fusion columns, (4) still hash-equal after a pause (OneDrive sync lag check),
                                                 and (5) the RAWX_UNNORMALIZED files must be marked DEFECTIVE.
Writes outputs/toolkit_swap_verification.csv (post) or data/processed/_corrected_parquet_hashes.json (pre)."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_provenance as ep

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
SRC = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth", "stanford", "stanford_2", "isu_ilcc", "tongji"]
FC = [f"fusion_{i}" for i in range(16)]
REC = PROC / "_corrected_parquet_hashes.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pre():
    rec = {s: {"sha256": sha(PROC / f"batterylife_{s}_merged_OLDENC_CORRECTED.parquet"),
               "bytes": (PROC / f"batterylife_{s}_merged_OLDENC_CORRECTED.parquet").stat().st_size} for s in SRC}
    REC.write_text(json.dumps(rec, indent=1))
    print("recorded", len(rec), "hashes ->", REC.name)


def post():
    rec = json.loads(REC.read_text())
    want_md5 = ep.encoder_md5(ep.OLD)
    rows, ok = [], True
    for s in SRC:
        canon = PROC / f"batterylife_{s}_merged.parquet"
        bad = PROC / f"batterylife_{s}_merged_RAWX_UNNORMALIZED.parquet"
        h1 = sha(canon)
        meta = ep.read_meta(canon)
        d = pd.read_parquet(canon, columns=FC)
        mx = float(np.nanmax(np.abs(d.to_numpy(float))))
        bad_meta = json.loads(ep.sidecar(bad).read_text()) if ep.sidecar(bad).exists() else {}
        rows.append({"source": s, "sha_matches_recorded_corrected": h1 == rec[s]["sha256"], "bytes_match": canon.stat().st_size == rec[s]["bytes"],
                     "sidecar_tag": meta.get("encoder_tag"), "sidecar_md5_matches_old_encoder": meta.get("encoder_md5") == want_md5,
                     "max_abs_fusion": round(mx, 2), "bounded_lt_100": mx < 100, "rawx_marked_defective": "DEFECTIVE" in str(bad_meta.get("encoder_tag", ""))})
    time.sleep(60)  # OneDrive sync-lag guard: hash again after a pause
    for r in rows:
        r["sha_stable_after_60s"] = sha(PROC / f"batterylife_{r['source']}_merged.parquet") == rec[r["source"]]["sha256"]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / "toolkit_swap_verification.csv", index=False)
    print(df.to_string(index=False))
    checks = ["sha_matches_recorded_corrected", "bytes_match", "sidecar_md5_matches_old_encoder", "bounded_lt_100", "rawx_marked_defective", "sha_stable_after_60s"]
    good = bool(df[checks].all().all()) and bool((df["sidecar_tag"] == ep.OLD).all())
    print("ALL SWAP CHECKS PASSED" if good else "SWAP VERIFICATION FAILED")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    {"pre": pre, "post": post}[sys.argv[1]]()
