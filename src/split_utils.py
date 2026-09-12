"""
Battery-level (not cycle-level) train/test split, shared by every Phase 2+
training script so the same batteries are held out everywhere — required
for the Phase 3 stacking ensemble to combine predictions meaningfully.

ASSUMPTION: deterministic split (every 5th battery in a sorted battery
list goes to test), not random — reproducible without needing to persist
a random seed choice, and gives ~20% test batteries spread evenly across
the sorted battery_id order rather than clustering.

The split is done PER DATASET (NASA batteries and MIT batteries each
split independently, then unioned) rather than on the pooled/sorted list
of all battery_ids together. Caught during Phase 2: pooling first meant
"every 5th" landed entirely inside the MIT block (since NASA sorts first
alphabetically and has exactly 4 batteries, a multiple of nothing in
particular but small enough that the stride skipped over it), so the
first XGBoost run's test set had zero NASA batteries. Per-dataset
stratification guarantees every dataset is represented in both splits.

PINNING (added: targeted improvement pass, Part 2). The Dataset
Expansion session found a real, unintended consequence of the purely
deterministic "every Nth" rule: adding 19 more NASA batteries shifted
where the stride landed, and NASA/B0018 - this project's single most-
studied case study (second-life grading session 25, sensor-noise
robustness session 26, the dedicated root-cause investigation session
27) - silently moved from test into train. Every future retrain would
silently repeat this for B0018 or any other battery, making before/
after comparisons for whichever specific battery happens to currently
be under study permanently unreliable. `pinned_test_ids` lets a caller
force specific battery IDs into the test set regardless of how the
stride lands, without abandoning the deterministic/reproducible
design for everything else.
"""

from __future__ import annotations


def battery_level_split(battery_ids: list[str], test_every: int = 5,
                         dataset_of: dict | None = None,
                         pinned_test_ids: set[str] | None = None):
    """
    dataset_of: optional {battery_id: dataset_name} map. When given, the
    split is stratified so each dataset contributes its own "every Nth"
    slice to test rather than the pooled list being split as one block.

    pinned_test_ids: optional set of battery IDs that MUST end up in the
    test set, overriding the deterministic stride for those IDs only.
    A pinned ID already selected by the stride is a no-op; one that
    would otherwise have landed in train is moved to test instead - a
    simple, transparent "add to test" (not a swap with some other
    battery), so it never silently drops the count of batteries
    normally selected by the stride. This does shift the affected
    dataset's train/test proportion very slightly (+1 test battery,
    same total pool) - by design, small, and meant to be verified by
    the caller (see split_utils_verify.py), not hidden.
    """
    unique_ids = sorted(set(battery_ids))
    pinned_test_ids = set(pinned_test_ids or ())
    unknown_pins = pinned_test_ids - set(unique_ids)
    if unknown_pins:
        raise ValueError(f"pinned_test_ids not present in battery_ids: {sorted(unknown_pins)}")

    if dataset_of is None:
        test_ids = set(unique_ids[test_every - 1 :: test_every]) | pinned_test_ids
        return sorted(set(unique_ids) - test_ids), sorted(test_ids)

    by_dataset: dict[str, list[str]] = {}
    for bid in unique_ids:
        by_dataset.setdefault(dataset_of[bid], []).append(bid)

    train_ids, test_ids = [], []
    for ds, ids in by_dataset.items():
        ids = sorted(ids)
        ds_test = (set(ids[test_every - 1 :: test_every]) or {ids[-1]})
        ds_test = ds_test | (pinned_test_ids & set(ids))
        train_ids.extend(i for i in ids if i not in ds_test)
        test_ids.extend(ds_test)
    return sorted(train_ids), sorted(test_ids)
