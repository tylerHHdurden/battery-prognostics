"""
Stage 2, Item 2.2: reconcile this project's EOL/censoring convention
against Severson et al. 2019's own published convention for the same
MIT batch-1/3 fast-charging cells.

STEP 1, stated explicitly - THIS PROJECT's current rule
(rul_labels.compute_eol_and_rul, UNCHANGED here): EOL = first cycle
where discharge_capacity <= 0.8 * median(this battery's OWN first 3
logged cycles' capacity); censored=True if that's never crossed within
the logged cycles.

ROOT-CAUSE FINDING, made by actually checking the numbers rather than
trusting the task's framing at face value: a first version of this
script re-implemented "Severson's convention" as 0.8 * 1.1Ah (nominal
rated capacity) and found ZERO cells flip between conventions - ALL 92
batch-1/3 cells are censored under BOTH the project's own rule AND
that reimplementation. Investigated why before concluding anything:
the raw MIT HDF5 files (`batch['cycle_life']`, Severson's OWN
precomputed field, published directly in their data release) show
`published_cycle_life - our_own_logged_n_cycles = 2.0 EXACTLY, for
ALL 90 matched cells (zero variance)`. This means Severson's OWN
released data is truncated at (within a fixed 2-cycle indexing offset
of) their own computed cycle_life for every one of these cells - their
capacity trace, as PUBLISHED, never extends far enough past EOL to
show a threshold-crossing either, regardless of which threshold
formula is applied to it. This is NOT a threshold-definition mismatch
(the task's working hypothesis) - it is a DATA AVAILABILITY
constraint shared by both this project's data and Severson's own
release. The correct reconciliation is therefore to read Severson's
own precomputed `cycle_life` value directly (available in the raw
file), not to re-derive an EOL cycle from a capacity trace that was
never released far enough to show it under any formula.

The +2 offset is applied to align published_cycle_life (Severson's own
1-indexed convention, apparently counting the skipped diagnostic cycle
0 and one further truncated/invalid final cycle our own loader also
drops) with this project's own cycle_idx numbering.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters import iterate_mit_cycles, MIT_DIR

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

CYCLE_IDX_OFFSET = 2  # published_cycle_life - our_own_n_cycles, verified exact/zero-variance across 90 cells


def eol_project_convention(caps, idxs):
    initial_cap = float(np.median(caps[:min(3, len(caps))]))
    threshold = 0.8 * initial_cap
    below = np.where(caps <= threshold)[0]
    if len(below) > 0:
        return int(idxs[below[0]]), False, initial_cap
    return int(idxs[-1]) + 1, True, initial_cap


def load_published_cycle_life(mit_full):
    """Reads Severson et al.'s own precomputed cycle_life field directly
    from the raw HDF5 files - the authoritative ground truth, not a
    reimplementation of their threshold formula."""
    cl_map = {}
    for bf in sorted(set(e["batch_file"] for e in mit_full)):
        path = MIT_DIR / bf
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            batch = f["batch"]
            n = batch["summary"].shape[0]
            for i in range(n):
                cl = f[batch["cycle_life"][i, 0]][:]
                cl_map[(bf, i)] = float(cl[0][0])
    return cl_map


def main():
    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full = json.load(f)
    b13 = [e for e in mit_full if e["global_id"].startswith("b1c") or e["global_id"].startswith("b3c")]
    print(f"[eol-reconcile] {len(b13)} MIT batch-1/3 cells to process")

    cl_map = load_published_cycle_life(mit_full)
    print(f"[eol-reconcile] loaded Severson's published cycle_life for {len(cl_map)} total MIT cells "
          f"(all batches)")

    rows = []
    for i, e in enumerate(b13):
        gid = e["global_id"]
        cycles = list(iterate_mit_cycles(e["batch_file"], e["cell_index"]))
        caps = np.array([c["discharge_capacity"] for c in cycles], dtype=float)
        idxs = np.array([c["cycle_idx"] for c in cycles], dtype=int)
        if len(caps) == 0:
            continue

        eol_proj, censored_proj, initial_cap = eol_project_convention(caps, idxs)

        published_cl = cl_map.get((e["batch_file"], e["cell_index"]))
        if published_cl is None:
            eol_severson, censored_severson = None, None
        else:
            eol_severson = published_cl - CYCLE_IDX_OFFSET
            # Severson's cycle_life is itself a finite, real observation for
            # every cell that has one published - never "censored" by
            # construction, since it's their own already-computed EOL.
            censored_severson = False

        rows.append({
            "global_id": gid, "n_cycles": len(caps), "initial_cap_measured": initial_cap,
            "eol_project": eol_proj, "censored_project": censored_proj,
            "published_cycle_life": published_cl, "eol_severson": eol_severson,
            "censored_severson": censored_severson,
        })
        if (i + 1) % 30 == 0:
            print(f"[eol-reconcile] ...{i+1}/{len(b13)} cells processed")

    df = pd.DataFrame(rows)
    n_matched = df["published_cycle_life"].notna().sum()
    print(f"\n[eol-reconcile] === RESULT: {len(df)} cells processed, "
          f"{n_matched} matched to a published cycle_life ===")
    if n_matched < len(df):
        unmatched = df[df["published_cycle_life"].isna()]["global_id"].tolist()
        print(f"[eol-reconcile] {len(df)-n_matched} cells with NO published cycle_life match: {unmatched} "
              f"(not covered by this reconciliation - reported, not silently dropped)")

    print(f"\n[eol-reconcile] censored under PROJECT convention: {df['censored_project'].sum()} "
          f"({df['censored_project'].mean()*100:.1f}%)")
    matched = df[df["published_cycle_life"].notna()]
    print(f"[eol-reconcile] censored under SEVERSON's OWN published cycle_life "
          f"(of the {len(matched)} matched cells): 0 (0.0%) - by construction, a precomputed "
          f"value IS a finite observation")

    flipped = matched[matched["censored_project"]]
    print(f"\n[eol-reconcile] cells that FLIP from censored (project) -> finite (Severson's own value): "
          f"{len(flipped)} of {len(matched)} matched cells")

    diff = matched["eol_project"] - matched["eol_severson"]
    print(f"\n[eol-reconcile] EOL cycle difference where project reports a finite value anyway vs. "
          f"Severson's own (project - Severson, all cells, since project's is always finite by "
          f"construction - either a real crossing or last-cycle+1): mean={diff.mean():+.1f}, "
          f"median={diff.median():+.1f}, std={diff.std():.1f}")
    print(f"[eol-reconcile] (this diff is dominated by the {df['censored_project'].sum()} still-censored-"
          f"under-project cells, where eol_project = n_cycles+1 by definition, NOT a genuine "
          f"threshold-crossing - those rows are NOT a meaningful RUL-label comparison, only a "
          f"raw cycle-count comparison)")

    df.to_csv(OUT_DIR / "eol_convention_reconciliation.csv", index=False)
    print(f"\n[eol-reconcile] saved outputs/eol_convention_reconciliation.csv")
    print("[eol-reconcile] DONE")


if __name__ == "__main__":
    main()
