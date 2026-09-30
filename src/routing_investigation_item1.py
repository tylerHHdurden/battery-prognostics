"""Item 1 routing investigation (READ-ONLY): recompute base vs extended R2 AND MAE on the
held-out sets used by run_finalpass_item1_labelfree_routing.py (same helpers, same models,
no AUC rule recomputed - rule_pick/AUC are read from finalpass_item1_labelfree_routing.csv).
Writes only outputs/toolkit_item1_routing_investigation.csv.
Routing was selected on held-out data (deployment choice, not an unbiased result)."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split, load_all_heldout_base,
    load_all_heldout_extended, base_feature_cols, extended_feature_cols, fit_medians,
    load_base_model, load_extended_model, build_X, score, OUT_DIR, PROC_DIR,
)

APP = ["CALCE", "Oxford", "HUST", "XJTU"]  # NASA/MIT are the train pool: no held-out base/ext pair exists
BL = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth", "stanford", "stanford_2", "isu_ilcc"]
PRE_CANDIDATE_TABLE = {"CALCE": "extended", "Oxford": "extended", "HUST": "extended"}  # else base


def main():
    mb, hb, trb, _, _ = load_base_pool_and_split()
    me, _, hraw, tre, _, _ = load_extended_pool_and_split()
    bc, ec = base_feature_cols(), extended_feature_cols()
    bmed, emed = fit_medians(mb, trb, bc), fit_medians(me, tre, ec)
    bm, em = load_base_model(), load_extended_model()
    hob, hoe = load_all_heldout_base(hb), load_all_heldout_extended(hraw)
    item1 = pd.read_csv(OUT_DIR / "finalpass_item1_labelfree_routing.csv").set_index("dataset")
    rows = []

    def one(name, dfb, dfe, scope):
        rb = score(dfb["SOH"].to_numpy(), bm.predict(build_X(dfb, bc, bmed)))
        re_ = score(dfe["SOH"].to_numpy(), em.predict(build_X(dfe, ec, emed)))
        w_r2 = "extended" if re_["r2"] > rb["r2"] else "base"
        w_mae = "extended" if re_["mae"] < rb["mae"] else "base"
        deployed = PRE_CANDIDATE_TABLE.get(name, "base")
        rule = item1.loc[name, "rule_pick"]
        rows.append({
            "dataset": name, "scope": scope, "n": rb["n"],
            "base_r2": rb["r2"], "ext_r2": re_["r2"], "base_mae": rb["mae"], "ext_mae": re_["mae"],
            "r2_margin_ext_minus_base": re_["r2"] - rb["r2"],
            "mae_margin_base_minus_ext": rb["mae"] - re_["mae"],
            "true_winner_r2": w_r2, "true_winner_mae": w_mae,
            "deployed_pick_precandidate_table": deployed, "deployed_correct_r2": deployed == w_r2,
            "deployed_correct_mae": deployed == w_mae,
            "labelfree_rule_pick": rule, "labelfree_correct_r2": rule == w_r2, "labelfree_correct_mae": rule == w_mae,
            "csv_r2_base": item1.loc[name, "r2_base_model"], "csv_r2_ext": item1.loc[name, "r2_ext_model"],
        })

    for n in APP:
        one(n, hob[n], hoe[n], "app")
    for s in BL:
        df = pd.read_parquet(PROC_DIR / f"batterylife_{s}_merged.parquet")
        one(s, df, df, "BatteryLife (INVALIDATED-scope, rerun on corrected embeddings)")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "toolkit_item1_routing_investigation.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
