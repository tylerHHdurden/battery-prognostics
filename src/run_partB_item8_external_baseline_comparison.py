"""
Part B, item 8: external baseline comparison against BatteryLife's own
published benchmark methods, on the same (BatteryLife-sourced) data used
in item 7.

FINDING, established BEFORE writing any comparison code (verified via
WebFetch of the paper itself, arXiv:2502.18807, not assumed): BatteryLife's
own benchmark task is NOT this project's task. Their benchmark predicts a
single scalar - the CYCLE NUMBER at which SOH first reaches 80% (90% for
CALB) - from only the first S<=100 cycles of a battery, scored by MAPE and
"15%-Acc" (fraction of predictions within 15% relative error). This
project's own protocol predicts SOH CONTINUOUSLY, per cycle, across a
battery's full life, scored by R2/RMSE. Different prediction target,
different input horizon, different metric family entirely.

A second, independently-checked limitation: BatteryLife's own paper
reports results ONLY aggregated by CHEMISTRY FAMILY (Li-ion/Zn-ion/
Na-ion/CALB), not broken out per individual data SOURCE (no separate
CALCE/HUST/XJTU/RWTH/Stanford/... rows anywhere in the paper - checked
directly, not assumed absent). So even setting the task/metric mismatch
aside, there is no published per-source number to compare THIS
project's own per-source R2 (item 7) against directly.

CONSEQUENCE, stated plainly: a literal "beats BatteryLife's published
number" comparison is not constructed here, because doing so would
require pretending two different tasks scored by two different metrics
are the same comparison - exactly the kind of manufactured, misleading
comparison this project's standing no-fabrication discipline exists to
prevent. What IS reported: BatteryLife's own published aggregate numbers
(for scale/context only), side-by-side with this project's own item-7
per-source R2 numbers, with the mismatch stated inline rather than
implied away by proximity on a page.

Reproducing BatteryLife's own early-cycle-life-prediction protocol
exactly, to get a genuinely comparable number, was judged out of this
pass's time budget - a real, disclosed scope limit, not attempted.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import OUT_DIR

# BatteryLife's own published Table 3 numbers (best-performing methods,
# aggregated by chemistry family - transcribed directly from the paper,
# arXiv:2502.18807, via WebFetch, not estimated or fabricated). Task:
# early-cycle-life (EOL cycle number) prediction, MAPE lower=better.
BATTERYLIFE_PUBLISHED_BENCHMARK = [
    {"chemistry_family": "Li-ion", "n_batteries": 837, "best_method": "CPTransformer",
     "mape": 0.184, "mape_std": 0.003, "acc15pct": 0.573, "acc15pct_std": 0.016},
    {"chemistry_family": "Li-ion (alt best)", "n_batteries": 837, "best_method": "CPMLP",
     "mape": 0.179, "mape_std": 0.003, "acc15pct": None, "acc15pct_std": None},
    {"chemistry_family": "Zn-ion", "n_batteries": 95, "best_method": "CPTransformer",
     "mape": 0.515, "mape_std": 0.067, "acc15pct": None, "acc15pct_std": None},
    {"chemistry_family": "Na-ion", "n_batteries": 31, "best_method": "CPTransformer",
     "mape": 0.255, "mape_std": 0.036, "acc15pct": None, "acc15pct_std": None},
    {"chemistry_family": "CALB", "n_batteries": 27, "best_method": "CPTransformer",
     "mape": 0.149, "mape_std": 0.005, "acc15pct": 0.672, "acc15pct_std": 0.107},
]


def main():
    print("=== Part B, item 8: external baseline comparison (BatteryLife's own published benchmark) ===")
    print("\nTASK/METRIC MISMATCH, stated up front: BatteryLife's benchmark = early-cycle-life (EOL-cycle) "
          "prediction from the first <=100 cycles, scored by MAPE/15%-Acc. This project's protocol = "
          "continuous per-cycle SOH regression, scored by R2/RMSE. NOT the same task - no single number "
          "below should be read as 'beats' or 'loses to' the other.")

    pub_df = pd.DataFrame(BATTERYLIFE_PUBLISHED_BENCHMARK)
    pub_df.to_csv(OUT_DIR / "partB_item8_batterylife_published_benchmark.csv", index=False)
    print("\nBatteryLife's own published aggregate numbers (context only):")
    print(pub_df.to_string(index=False))

    item7_path = OUT_DIR / "partB_item7_zeroretrain_eval.csv"
    if item7_path.exists():
        item7_df = pd.read_csv(item7_path)
        print("\nThis project's own item-7 per-source R2 (different task/metric - shown for scale only, "
              "NOT a head-to-head comparison):")
        print(item7_df.to_string(index=False))
    else:
        print("\n[item8] item 7's results not found yet - run item 7 first for the side-by-side context table.")

    print("\n[item8] No per-source published number exists in BatteryLife's own paper to compare against "
          "directly (checked directly - results tables are aggregated by chemistry family only). "
          "Reproducing their exact early-cycle-life protocol for a genuinely comparable number was judged "
          "out of this pass's time budget - not attempted, disclosed as a real scope limit.")


if __name__ == "__main__":
    main()
