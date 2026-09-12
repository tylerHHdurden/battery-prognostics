"""
Targeted improvement pass, Part 2: pin NASA/B0018 to the test set
permanently, verify the fix, and produce a demonstration split file
ready for the next full expanded-pool retrain.

Scope decision, stated explicitly: this script verifies the new
`pinned_test_ids` logic in split_utils.py and writes
`battery_split_expanded_b0018pinned.json` as a real, usable split - but
does NOT retroactively retrain the entire 11-model Dataset Expansion
pipeline against it this session. That would mean re-running XGBoost,
VLSTM, CNN-LSTM, PiFormer, CNN-BiGRU, the fusion encoder, XGBoost-
fusion, and the ensemble all over again (the ~9h Dataset Expansion
Phase 1 chain), which is out of scope for "fix and verify the split
logic." Parts 1/3/4 of this session's work continue to use the
existing battery_split_expanded.json (B0018 in train) so their
before/after numbers stay directly comparable to the just-completed
Dataset Expansion report. This pinned split is verified correct and
ready to be the split used the next time the full pipeline is retrained.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import battery_level_split

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def describe(train_ids, test_ids, dataset_of, label):
    n_nasa_train = sum(1 for b in train_ids if dataset_of[b] == "NASA")
    n_nasa_test = sum(1 for b in test_ids if dataset_of[b] == "NASA")
    n_mit_train = len(train_ids) - n_nasa_train
    n_mit_test = len(test_ids) - n_nasa_test
    print(f"[b0018-pin] {label}: train={len(train_ids)} ({n_nasa_train} NASA, "
          f"{n_mit_train} MIT), test={len(test_ids)} ({n_nasa_test} NASA, "
          f"{n_mit_test} MIT) - NASA test fraction={n_nasa_test/(n_nasa_train+n_nasa_test):.3f}, "
          f"MIT test fraction={n_mit_test/(n_mit_train+n_mit_test):.3f}")
    return n_nasa_train, n_nasa_test, n_mit_train, n_mit_test


def main():
    df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    df = df[df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    dataset_of = dict(zip(df["battery_id"], df["dataset"]))
    all_ids = df["battery_id"].tolist()

    print("[b0018-pin] === Step 1: reproduce the EXISTING (unpinned) split exactly, "
          "confirm it matches the on-disk battery_split_expanded.json ===")
    train_orig, test_orig = battery_level_split(all_ids, dataset_of=dataset_of)
    on_disk = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    assert train_orig == on_disk["train_ids"] and test_orig == on_disk["test_ids"], \
        "Reproduced split does not match battery_split_expanded.json - split_utils.py's " \
        "unpinned behavior changed unexpectedly, stop and investigate before trusting pinning."
    print("[b0018-pin] MATCH CONFIRMED - unpinned reproduction is byte-identical to the "
          "on-disk split. split_utils.py's existing behavior is unchanged for every caller "
          "that doesn't pass pinned_test_ids (backward-compatible).")
    print(f"[b0018-pin] B0018 in existing split: "
          f"{'TEST' if 'B0018' in test_orig else 'TRAIN'} (confirms the bug this Part fixes)")
    describe(train_orig, test_orig, dataset_of, "BEFORE pinning (existing)")

    print("\n[b0018-pin] === Step 2: same split WITH B0018 pinned to test ===")
    train_pinned, test_pinned = battery_level_split(
        all_ids, dataset_of=dataset_of, pinned_test_ids={"B0018"}
    )
    assert "B0018" in test_pinned, "PINNING FAILED - B0018 not in test set after pinning"
    assert "B0018" not in train_pinned, "PINNING FAILED - B0018 still in train set after pinning"
    print("[b0018-pin] VERIFIED: B0018 is now in the test set.")
    describe(train_pinned, test_pinned, dataset_of, "AFTER pinning")

    # Everything else about the split should be untouched (only B0018 moved
    # train->test). Compare each side with B0018 removed - any leftover
    # difference would mean pinning disturbed some OTHER battery's
    # membership, which it must not.
    other_train_before = set(train_orig) - {"B0018"}
    other_train_after = set(train_pinned) - {"B0018"}
    other_test_before = set(test_orig) - {"B0018"}
    other_test_after = set(test_pinned) - {"B0018"}
    unexpected_train_diff = other_train_before ^ other_train_after
    unexpected_test_diff = other_test_before ^ other_test_after
    print(f"\n[b0018-pin] Any OTHER battery whose split membership changed "
          f"(should be empty): train-side={unexpected_train_diff}, "
          f"test-side={unexpected_test_diff}")
    assert not unexpected_train_diff and not unexpected_test_diff, \
        "Pinning changed some other battery's split membership - not a surgical override"
    print("[b0018-pin] VERIFIED: no other battery's split membership changed - this is a "
          "single, surgical override (B0018 train->test only), not a re-shuffle.")

    n_nasa_train_b, n_nasa_test_b, n_mit_train_b, n_mit_test_b = describe(
        train_orig, test_orig, dataset_of, "  (before, restated)")
    n_nasa_train_a, n_nasa_test_a, n_mit_train_a, n_mit_test_a = describe(
        train_pinned, test_pinned, dataset_of, "  (after, restated)")
    print(f"\n[b0018-pin] NASA test-battery count: {n_nasa_test_b} -> {n_nasa_test_a} "
          f"(+1, exactly B0018). MIT test-battery count unchanged: {n_mit_test_b} -> "
          f"{n_mit_test_a}. Both datasets remain represented in both splits (the original "
          f"session 2 stratification bug this project already fixed once) - "
          f"NASA train={n_nasa_train_a}>0, NASA test={n_nasa_test_a}>0, "
          f"MIT train={n_mit_train_a}>0, MIT test={n_mit_test_a}>0.")
    assert n_nasa_train_a > 0 and n_nasa_test_a > 0 and n_mit_train_a > 0 and n_mit_test_a > 0

    out_path = PROC_DIR / "battery_split_expanded_b0018pinned.json"
    out_path.write_text(json.dumps({
        "train_ids": train_pinned, "test_ids": test_pinned,
        "_note": "Same as battery_split_expanded.json's per-dataset stride split, with "
                 "NASA/B0018 explicitly pinned to test regardless of where the stride "
                 "lands - see split_utils.py's pinned_test_ids and "
                 "verify_b0018_pinned_split.py. Ready for the next full expanded-pool "
                 "retrain; not yet applied to the currently-deployed *_expanded models, "
                 "which were trained on the unpinned battery_split_expanded.json.",
    }, indent=2))
    print(f"\n[b0018-pin] Wrote {out_path.name} ({len(train_pinned)} train / "
          f"{len(test_pinned)} test batteries).")
    print("[b0018-pin] DONE")


if __name__ == "__main__":
    main()
