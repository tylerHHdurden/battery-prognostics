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
"""

from __future__ import annotations


def battery_level_split(battery_ids: list[str], test_every: int = 5,
                         dataset_of: dict | None = None):
    """
    dataset_of: optional {battery_id: dataset_name} map. When given, the
    split is stratified so each dataset contributes its own "every Nth"
    slice to test rather than the pooled list being split as one block.
    """
    unique_ids = sorted(set(battery_ids))
    if dataset_of is None:
        test_ids = set(unique_ids[test_every - 1 :: test_every])
        return sorted(set(unique_ids) - test_ids), sorted(test_ids)

    by_dataset: dict[str, list[str]] = {}
    for bid in unique_ids:
        by_dataset.setdefault(dataset_of[bid], []).append(bid)

    train_ids, test_ids = [], []
    for ds, ids in by_dataset.items():
        ids = sorted(ids)
        ds_test = set(ids[test_every - 1 :: test_every]) or {ids[-1]}
        train_ids.extend(i for i in ids if i not in ds_test)
        test_ids.extend(ds_test)
    return sorted(train_ids), sorted(test_ids)
