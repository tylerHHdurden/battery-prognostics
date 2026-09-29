"""
Final experiment pass 2, item D: honest routing selection.

Item 1 (prior pass) evaluated a label-free AUC rule against all 13
datasets AT ONCE - the rule's own construction never used labels, but
no genuine held-out MODEL-SELECTION step was ever done (no candidate
routing strategy was ever CHOSEN using one group's labels and then
checked on an untouched group). This item adds that missing step:

Direction 1: choose the best-performing routing STRATEGY (among a
small, pre-specified candidate set - not fit/tuned per-dataset, which
would just be item 1's own already-tried approach) using ONLY the 9
BatteryLife sources' TRUE outcomes as the validation set, then apply
that one fixed choice to CALCE/Oxford/HUST/XJTU, untouched.

Direction 2 (reverse): choose using ONLY CALCE/Oxford/HUST/XJTU's true
outcomes, apply to the 9 BatteryLife sources, untouched.

Candidate routing strategies (evaluated identically in both
directions - same 4 candidates, no per-direction customization):
  - always_base
  - always_extended
  - auc_rule_tie_base   (item 1's own rule: lower-AUC representation
                          wins; ties -> base)
  - auc_rule_tie_ext    (same AUC comparison; ties -> extended instead)

Reuses item 1's own already-computed, unchanged numbers
(outputs/finalpass_item1_labelfree_routing.csv) - no model retrained,
no AUC recomputed, this item is purely about the SELECTION step on top
of numbers item 1 already established.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import OUT_DIR

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
CLASSIC_4 = ["CALCE", "Oxford", "HUST", "XJTU"]
AUC_TIE_MARGIN = 0.01


def apply_strategy(row, strategy: str) -> str:
    if strategy == "always_base":
        return "base"
    if strategy == "always_extended":
        return "extended"
    if strategy in ("auc_rule_tie_base", "auc_rule_tie_ext"):
        diff = row["auc_base_repr"] - row["auc_ext_repr"]
        if diff > AUC_TIE_MARGIN:
            return "extended"
        if -diff > AUC_TIE_MARGIN:
            return "base"
        return "base" if strategy == "auc_rule_tie_base" else "extended"
    raise ValueError(strategy)


def accuracy_of_strategy(df: pd.DataFrame, strategy: str) -> float:
    picks = df.apply(lambda r: apply_strategy(r, strategy), axis=1)
    return float((picks == df["true_winner"]).mean())


def main():
    print("=== Final pass 2, item D: honest routing selection (proper validation/test split, both directions) ===")

    item1_df = pd.read_csv(OUT_DIR / "finalpass_item1_labelfree_routing.csv")
    strategies = ["always_base", "always_extended", "auc_rule_tie_base", "auc_rule_tie_ext"]

    results = []
    for direction, val_names, test_names in [
        ("BatteryLife-validation -> classic-4-test", BATTERYLIFE_SOURCES, CLASSIC_4),
        ("classic-4-validation -> BatteryLife-test", CLASSIC_4, BATTERYLIFE_SOURCES),
    ]:
        val_df = item1_df[item1_df["dataset"].isin(val_names)]
        test_df = item1_df[item1_df["dataset"].isin(test_names)]

        val_acc = {s: accuracy_of_strategy(val_df, s) for s in strategies}
        best_strategy = max(val_acc, key=val_acc.get)
        best_val_acc = val_acc[best_strategy]
        test_acc = accuracy_of_strategy(test_df, best_strategy)

        print(f"\n--- {direction} ---")
        print(f"[itemD] validation-set accuracy per candidate strategy: "
              f"{', '.join(f'{s}={a:.3f}' for s, a in val_acc.items())}")
        print(f"[itemD] SELECTED strategy (best on validation): '{best_strategy}' (val_acc={best_val_acc:.3f})")
        print(f"[itemD] applied UNTOUCHED to test set -> test_acc={test_acc:.3f} (n={len(test_df)})")

        test_picks = test_df.apply(lambda r: apply_strategy(r, best_strategy), axis=1)
        for name, pick, truth in zip(test_df["dataset"], test_picks, test_df["true_winner"]):
            flag = "CORRECT" if pick == truth else "WRONG"
            print(f"    {name}: honest-selected pick='{pick}' true_winner='{truth}' -> {flag}")

        oracle_ceiling = 1.0  # by definition (always matches true_winner) - a reference, not a real strategy
        current_live_acc = float((test_df["currently_routed_live"] == test_df["true_winner"]).mean())
        print(f"[itemD] for reference: oracle ceiling (always right, not achievable honestly)={oracle_ceiling:.3f}, "
              f"CURRENT LIVE routing's own accuracy on this same test set={current_live_acc:.3f}")

        results.append({
            "direction": direction, "n_val": len(val_df), "n_test": len(test_df),
            "selected_strategy": best_strategy, "val_accuracy": best_val_acc, "test_accuracy": test_acc,
            "current_live_accuracy_on_test": current_live_acc,
            **{f"val_acc_{s}": val_acc[s] for s in strategies},
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "finalpass2_itemD_honest_routing.csv", index=False)

    print("\n=== SUMMARY ===")
    print(results_df.to_string(index=False))

    reproduces_oracle = results_df["test_accuracy"].eq(1.0)
    print(f"\n[itemD] Honest selection reproduces the (trivial, always-100%) oracle on "
          f"{int(reproduces_oracle.sum())}/{len(results_df)} directions - "
          f"{'both' if reproduces_oracle.all() else 'NOT both'} directions achieve perfect test accuracy "
          f"from a strategy chosen without ever looking at test labels.")

    print("\n[itemD] HNEI/RWTH detail (both flagged in Part B as plausible new routing candidates):")
    for name in ["hnei", "rwth"]:
        row = item1_df[item1_df["dataset"] == name].iloc[0]
        print(f"    {name}: true_winner={row['true_winner']}, item1's own label-free rule picked "
              f"'{row['rule_pick']}' ({'correct' if row['rule_correct'] else 'wrong'}), "
              f"currently routed live to '{row['currently_routed_live']}'")


if __name__ == "__main__":
    main()
