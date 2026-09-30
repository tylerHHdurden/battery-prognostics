"""
16-source embedding provenance table (2026-09-30). For each source, re-encodes a sample of REAL cycles three ways -
OLD encoder + old norm stats (the deployed path), CANDIDATE encoder + candidate stats (sanitized, as trained), and
the defective RAW-X path (old encoder, no normalization) - and reports which one each stored embedding store matches.
Also states which model consumes each store and which encoder that model expects. Nothing is assumed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles, iterate_calce_cycles
from data_adapters_batterylife import batterylife_cell_ids, iterate_batterylife_cycles
from sequence_features import get_cycle_tensor, apply_channel_norm
from stage7_common import (oxford_cell_ids, iterate_oxford_cycles, hust_cell_ids, iterate_hust_cycles,
                           xjtu_cell_ids_soh_valid, iterate_xjtu_cycles)
from stage1_common import build_calce_merged, add_reformulated_duration_features
from models.ica_encoder import ICAEncoder

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FC = [f"fusion_{i}" for i in range(16)]
SL = slice(3, 6)
RAW = {"ul_pur": "UL_PUR", "hnei": "HNEI", "snl": "SNL", "mich": "MICH", "mich_exp": "MICH_EXP", "rwth": "RWTH",
       "stanford": "Stanford", "stanford_2": "Stanford_2", "isu_ilcc": "ISU_ILCC", "tongji": "Tongji"}


def main():
    old_enc = ICAEncoder(3, 16); old_enc.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt")); old_enc.eval()
    cand_enc = ICAEncoder(3, 16); cand_enc.load_state_dict(torch.load(ROOT / "models" / "_candidate_ica_encoder.pt")); cand_enc.eval()
    old_stats = json.loads((PROC / "channel_norm_stats.json").read_text())
    cand_stats = json.loads((PROC / "candidate_channel_norm_stats.json").read_text())

    def enc_all(x):
        with torch.no_grad():
            e_old = old_enc.encode(torch.tensor(apply_channel_norm(x[None].astype(np.float32), old_stats)[:, :, SL])).numpy()[0]
            xs = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
            e_cand = cand_enc.encode(torch.tensor(apply_channel_norm(xs[None].astype(np.float32), cand_stats)[:, :, SL])).numpy()[0]
            e_raw = old_enc.encode(torch.tensor(x[None][:, :, SL].astype(np.float32))).numpy()[0]
        return e_old, e_cand, e_raw

    cand_csv = pd.read_csv(PROC / "fusion_embeddings_multisource.csv")
    cand_csv["battery_id"] = cand_csv["battery_id"].astype(str)
    cand_idx = cand_csv.set_index(["battery_id", "cycle_idx"])[FC]
    old_nm = pd.read_csv(PROC / "fusion_embeddings.csv"); old_nm["battery_id"] = old_nm["battery_id"].astype(str)
    old_nm_idx = old_nm.set_index(["battery_id", "cycle_idx"])[FC]
    hi_full = add_reformulated_duration_features(pd.read_parquet(PROC / "hi_table.parquet"))
    calce = build_calce_merged(hi_full); calce["battery_id"] = calce["battery_id"].astype(str)
    calce_idx = calce.set_index(["battery_id", "cycle_idx"])[FC]
    stage51 = {}
    for n in ["oxford", "hust", "xjtu"]:
        pq = pd.read_parquet(PROC / f"stage5_1_{n}_merged.parquet"); pq["battery_id"] = pq["battery_id"].astype(str)
        stage51[n.capitalize() if n == "oxford" else n.upper()] = pq.set_index(["battery_id", "cycle_idx"])[FC]
    bl_canon = {}
    for s in RAW:
        pq = pd.read_parquet(PROC / f"batterylife_{s}_merged.parquet"); pq["battery_id"] = pq["battery_id"].astype(str)
        bl_canon[s] = pq.set_index(["battery_id", "cycle_idx"])[FC]
    bl_corr = pd.read_parquet(PROC / "old_encoder_embeddings_batterylife_corrected.parquet")
    bl_corr["battery_id"] = bl_corr["battery_id"].astype(str)

    mit = json.load(open(PROC / "mit_subset.json"))
    plan = [("NASA", "B0005", "B0005", lambda: iterate_nasa_cycles("B0005"), old_nm_idx, "fusion_embeddings.csv"),
            ("MIT", mit[0]["global_id"], mit[0]["global_id"], lambda: iterate_mit_cycles(mit[0]["batch_file"], mit[0]["cell_index"]), old_nm_idx, "fusion_embeddings.csv"),
            ("CALCE", "CS2_35", "CS2_35", lambda: iterate_calce_cycles("CS2_35"), calce_idx, "build_calce_merged (stage1_common)")]
    for nm, idf, itf, key in [("Oxford", oxford_cell_ids, iterate_oxford_cycles, "Oxford"), ("HUST", hust_cell_ids, iterate_hust_cycles, "HUST"),
                              ("XJTU", xjtu_cell_ids_soh_valid, iterate_xjtu_cycles, "XJTU")]:
        raw_id = idf()[0]; cid = str(raw_id)
        plan.append((nm, cid, cid, (lambda itf=itf, raw_id=raw_id: itf(raw_id)), stage51[key], f"stage5_1_{nm.lower()}_merged.parquet"))
    for s, raw in RAW.items():
        cid = batterylife_cell_ids(raw)[0]
        plan.append((s, f"{s}::{cid}", str(cid), (lambda raw=raw, cid=cid: iterate_batterylife_cycles(raw, cid)), bl_canon[s], f"batterylife_{s}_merged.parquet (canonical, PRE-fix)"))

    rows = []
    for src, cand_bid, store_bid, mk, old_store, store_name in plan:
        cycles = list(mk())
        step = max(1, len(cycles) // 20)
        d = {"old_store_vs_old": [], "old_store_vs_cand": [], "old_store_vs_raw": [], "cand_store_vs_cand": [], "corr_store_vs_old": []}
        for c in cycles[::step][:20]:
            x = get_cycle_tensor(c, n_bins=200)
            if x is None:
                continue
            e_old, e_cand, e_raw = enc_all(x)
            ci = int(c["cycle_idx"])
            if (store_bid, ci) in old_store.index:
                so = old_store.loc[(store_bid, ci)].to_numpy(float)
                if np.isfinite(so).all() and np.isfinite(e_old).all():
                    d["old_store_vs_old"].append(np.abs(so - e_old).max())
                    d["old_store_vs_cand"].append(np.abs(so - e_cand).max())
                    d["old_store_vs_raw"].append(np.abs(so - e_raw).max())
            if (cand_bid, ci) in cand_idx.index:
                d["cand_store_vs_cand"].append(np.abs(cand_idx.loc[(cand_bid, ci)].to_numpy(float) - e_cand).max())
            if src in RAW:
                m = bl_corr[(bl_corr["source"] == src) & (bl_corr["battery_id"] == store_bid) & (bl_corr["cycle_idx"] == ci)]
                if len(m) and np.isfinite(m[FC].to_numpy(float)).all() and np.isfinite(e_old).all():
                    d["corr_store_vs_old"].append(np.abs(m[FC].to_numpy(float)[0] - e_old).max())
        med = lambda k: float(np.max(d[k])) if d[k] else np.nan
        vo, vc, vr = med("old_store_vs_old"), med("old_store_vs_cand"), med("old_store_vs_raw")
        carries = ("OLD encoder, normalized" if vo < 1e-4 else ("OLD encoder on RAW un-normalized tensors (DEFECTIVE)" if vr < 1e-4 else
                   ("CANDIDATE encoder" if vc < 1e-4 else "UNMATCHED")))
        rows.append({"source": src, "pre_fix_store_read_by_old_models": store_name, "n_compared": len(d["old_store_vs_old"]),
                     "max_diff_vs_old_encoder": vo, "max_diff_vs_candidate_encoder": vc, "max_diff_vs_rawX_old_encoder": vr,
                     "store_actually_carries": carries,
                     "candidate_store": "fusion_embeddings_multisource.csv", "candidate_store_max_diff_vs_candidate": med("cand_store_vs_cand"),
                     "corrected_old_store_max_diff_vs_old": med("corr_store_vs_old") if src in RAW else np.nan,
                     "old_models_expect": "old_v1 (xgb_soh_fusion, extended, deployed OC-SVM)",
                     "candidate_models_expect": "candidate_v1 (_candidate_multisource, _candidate_ocsvm, trust profiles)"})
        print(f"[prov] {src}: old-model store ({store_name}) carries {carries} | diffs old/cand/raw = {vo:.2e}/{vc:.2e}/{vr:.2e} | "
              f"cand store vs cand {rows[-1]['candidate_store_max_diff_vs_candidate']:.2e}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "toolkit_embedding_provenance_table.csv", index=False)
    print("[prov] saved outputs/toolkit_embedding_provenance_table.csv")


if __name__ == "__main__":
    main()
