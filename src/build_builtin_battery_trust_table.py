"""
Precompute the nearest-source trust verdict for every BUILT-IN battery (all 16 sources) so the app can look it
up instead of re-encoding cycles at page-load time. Same math as trust_report.nearest_source_trust_report
(standardize -> per-source log-likelihood -> nearest -> raw-Mahalanobis threshold check), applied to each
battery's own MEDIAN 25-dim vector (candidate encoder's embeddings for every source - see
run_toolkit_phase2b_federated._attach_candidate_fusion for the bug this depends on being fixed).

`in_profile_train` records whether the battery was one of the training batteries of the profile it matched
(in-sample batteries are near their own source's mean by construction - the honest, out-of-sample check is
the held-out validation table, outputs/toolkit_phase3_trust_report_validation.csv).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_toolkit_phase2b_federated import load_pooled_data, ALL_SOURCES
from build_source_trust_profiles import battery_mean_vectors, split_with_fallback
from trust_report import nearest_source_trust_report

ROOT = Path(__file__).resolve().parent.parent
BL = {"ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth", "stanford", "stanford_2", "isu_ilcc", "tongji"}


def app_id(name, bid):
    return bid if name in BL else bid[len(name) + 2:]


def main():
    sources, all_cols, _ = load_pooled_data()
    rows = []
    for name in ALL_SOURCES:
        df = sources[name]
        vecs = battery_mean_vectors(df, all_cols)
        train_ids, test_ids = split_with_fallback(list(vecs.keys()), test_every=5)
        train_set = set(train_ids)
        for bid, v in vecs.items():
            r = nearest_source_trust_report(v)
            rows.append({"dataset": name, "battery_id": app_id(name, bid), "nearest_source": r["nearest_source"],
                         "trust_level": r["trust_level"], "distance": r["distance_to_nearest"],
                         "thr_familiar": r["distance_threshold_familiar"],
                         "thr_somewhat": r["distance_threshold_somewhat_familiar"],
                         "measured_mae": r["measured_mae"], "measured_mae_source": r["measured_mae_source"],
                         "nll_min": r["nll_min"], "gate_table_mae": r["gate_table_mae"], "lodo_family_mae": r["lodo_family_mae"],
                         "in_profile_train": bool(bid in train_set and r["nearest_source"] == name)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "models" / "_builtin_battery_trust.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.groupby(["dataset", "trust_level"]).size().unstack(fill_value=0).to_string())
    print("\nnearest==own source:", float((out["nearest_source"] == out["dataset"]).mean()))
    print(out[out["dataset"].isin(["NASA", "MIT"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
