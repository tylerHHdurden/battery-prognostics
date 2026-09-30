"""
Encoder-divergence quantification (2026-09-30 priority item 1).

For every source, the fraction of rows with ANY |fusion_k| > 10x that dimension's TRAINING-set 99.9th
percentile, for BOTH encoders that exist in this project:
  - CANDIDATE encoder (data/processed/fusion_embeddings_multisource.csv) - the one live for every
    non-NASA/MIT dataset and all uploads. Threshold from ITS OWN training rows (Phase 2's split).
  - OLD deployed encoder embeddings as stored in the per-source merged parquets. Threshold from the
    OLD encoder's own NASA+MIT training rows (fusion_embeddings.csv).
plus SOH MAE on flagged vs normal rows, using the frozen candidate XGBoost (no retraining).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from xgboost import XGBRegressor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import battery_level_split
from stage1_common import canonical_feature_cols

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FC = [f"fusion_{i}" for i in range(16)]
REL = canonical_feature_cols(reformulated=True)
PARQ = {"Oxford": "stage5_1_oxford_merged.parquet", "HUST": "stage5_1_hust_merged.parquet",
        "XJTU": "stage5_1_xjtu_merged.parquet"}
for s in ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth", "stanford", "stanford_2", "isu_ilcc"]:
    PARQ[s] = f"batterylife_{s}_merged.parquet"
PARQ["tongji"] = "batterylife_tongji_merged.parquet"


def p999_train(df, dcol="dataset"):
    dataset_of = dict(zip(df["battery_id"], df[dcol]))
    tr_ids, _ = battery_level_split(sorted(set(df["battery_id"].tolist())), test_every=5, dataset_of=dataset_of)
    tr = df["battery_id"].isin(set(tr_ids)).to_numpy()
    return np.percentile(np.abs(df.loc[tr, FC].to_numpy(float)), 99.9, axis=0), tr


def main():
    cand = pd.read_csv(PROC / "fusion_embeddings_multisource.csv")
    cand["battery_id"] = cand["battery_id"].astype(str)
    p_c, _ = p999_train(cand)
    thr_c = np.maximum(10 * p_c, 1e-6)
    cand["flag_cand"] = (np.abs(cand[FC].to_numpy(float)) > thr_c).any(axis=1)

    old = pd.read_csv(PROC / "fusion_embeddings.csv")
    old["battery_id"] = old["battery_id"].astype(str)
    p_o, _ = p999_train(old)
    thr_o = np.maximum(10 * p_o, 1e-6)

    hi_full = pd.read_parquet(PROC / "hi_table.parquet")  # not used for features here; parquets carry *_rel cols
    model = XGBRegressor(); model.load_model(str(ROOT / "models" / "_candidate_multisource.json"))
    med = json.loads((PROC / "candidate_multisource_medians.json").read_text())
    medians = dict(zip(med["cols"], med["medians"]))
    base_model = XGBRegressor(); base_model.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    cand_cols = REL + ["cycle_idx"] + FC

    rows = []
    for name, fn in PARQ.items():
        pq = pd.read_parquet(PROC / fn)
        pq = pq.copy()
        pq["battery_id"] = (name + "::" if name not in ("Oxford", "HUST", "XJTU") else "") + pq["battery_id"].astype(str)
        old_flag = (np.abs(pq[FC].to_numpy(float)) > thr_o).any(axis=1) | ~np.isfinite(pq[FC].to_numpy(float)).all(axis=1)
        pq["flag_old"] = old_flag
        emb = cand[cand["dataset"] == name][["battery_id", "cycle_idx"] + FC + ["flag_cand"]]
        m = pd.merge(pq.drop(columns=FC), emb, on=["battery_id", "cycle_idx"], how="inner")
        # old-flag must be carried through the merge (merge used pq.drop(FC)); recompute alignment
        m = pd.merge(m, pq[["battery_id", "cycle_idx", "flag_old"]].rename(columns={"flag_old": "flag_old2"}),
                     on=["battery_id", "cycle_idx"], how="left") if "flag_old" not in m.columns else m
        m["flag_old"] = m["flag_old"] if "flag_old" in m.columns else m["flag_old2"]
        X = m[cand_cols].to_numpy(float)
        for j, c in enumerate(cand_cols):  # impute like Phase 2 / live_inference
            bad = ~np.isfinite(X[:, j]); X[bad, j] = medians[c]
        err_c = np.abs(model.predict(X) - m["SOH"].to_numpy(float))
        # candidate model fed the OLD (possibly diverged) parquet embeddings = what the buggy pooled loader did
        pq_m = pd.merge(pq, m[["battery_id", "cycle_idx"]], on=["battery_id", "cycle_idx"], how="inner")
        X2 = pq_m[cand_cols].to_numpy(float)
        for j, c in enumerate(cand_cols):
            bad = ~np.isfinite(X2[:, j]); X2[bad, j] = medians[c]
        err_mixed = np.abs(model.predict(X2) - pq_m["SOH"].to_numpy(float))
        err_base = np.abs(base_model.predict(pq_m[cand_cols].to_numpy(float)) - pq_m["SOH"].to_numpy(float))
        fo = pq_m["flag_old"].to_numpy(bool) if "flag_old" in pq_m.columns else np.zeros(len(pq_m), bool)
        rows.append({
            "source": name, "n_rows": len(m),
            "frac_flagged_candidate": float(m["flag_cand"].mean()),
            "frac_flagged_old_encoder": float(fo.mean()),
            "cand_MAE_on_old_flagged_rows": float(err_c[fo]) if False else (float(err_c[m["flag_old"].to_numpy(bool)].mean()) if m["flag_old"].any() else np.nan),
            "cand_MAE_on_old_normal_rows": float(err_c[~m["flag_old"].to_numpy(bool)].mean()) if (~m["flag_old"]).any() else np.nan,
            "cand_model_fed_OLD_emb_MAE_flagged": float(err_mixed[fo].mean()) if fo.any() else np.nan,
            "cand_model_fed_OLD_emb_MAE_normal": float(err_mixed[~fo].mean()) if (~fo).any() else np.nan,
            "old_base_model_MAE_flagged": float(err_base[fo].mean()) if fo.any() else np.nan,
            "old_base_model_MAE_normal": float(err_base[~fo].mean()) if (~fo).any() else np.nan,
            "old_emb_max_abs": float(np.nanmax(np.abs(pq[FC].to_numpy(float)))),
        })
    # in-domain sources for completeness (candidate only; old encoder n/a - same code path)
    for name in ["NASA", "MIT", "CALCE"]:
        sub = cand[cand["dataset"] == name]
        rows.append({"source": name, "n_rows": len(sub), "frac_flagged_candidate": float(sub["flag_cand"].mean())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "toolkit_encoder_divergence_quantification.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
