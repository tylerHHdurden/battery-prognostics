"""
Dataset Expansion Phase 1 - a real data-quality bug caught during Step 2
(BFA feature selection), root-caused rather than passed through.

**Finding**: BFA's baseline Ridge RMSE on the expanded pool came back at
26.54 (SOH%), vs. 3.897 on the original 32-battery pool - a 6.8x jump
that demanded investigation rather than being accepted at face value.
Root cause: `rul_labels.soh_per_cycle` (shared, UNTOUCHED code) defines
SOH = capacity / median(first 3 logged cycles' capacity) * 100 - a
convention that was silently correct for every battery in the original
32-battery pool, but breaks down for a subset of batteries only reached
by this session's expansion, where the first several logged cycles are
NOT ordinary aging cycles but a separate low-rate/characterization
protocol phase with genuinely tiny measured capacity (confirmed by
inspecting NASA/B0041's raw trace directly: capacities of ~0.044-0.057Ah
for its first 41 cycles, then an abrupt jump to ~1.09-1.22Ah from cycle
42 onward, where its real aging trend begins) - dividing later, normal
cycles by that degenerate near-zero baseline produces physically
impossible SOH values (up to 2177% for B0041).

**Exclusion criterion** (principled, not cherry-picked to hit any target
number): any battery whose SOH ever exceeds 110% - chosen because
session 16 already independently established that MILD readings above
100% (up to ~101.5%, e.g. MIT/b3c0's real formation-cycle bump to
100.29%) are genuine, physically plausible early-life effects, not
artifacts - so 110% is a deliberately generous ceiling that keeps every
battery showing that kind of normal behavior, and excludes only the
ones with clearly nonsensical (150%+) readings. A `min SOH == 0%` alone
does NOT trigger exclusion - many otherwise-clean batteries (e.g.
B0042-48, B0053-56) legitimately fade all the way to 0% capacity at
true end-of-life, a real data point, not an artifact.

**Scope**: checked BOTH NASA and MIT (not just the NASA groups where it
was first noticed) - found 4 affected MIT batteries too (b1c18 max
270.4%, b2c44 max 144.1%, b1c0 max 143.6%, b2c12 max 138.9%), the exact
same underlying mechanism (an anomalous low-capacity early-cycle
baseline), confirming this is a general property of `rul_labels.py`'s
labeling convention when applied to protocol variety it was never
validated against, not a NASA-specific or MIT-specific quirk.

**Disposition**: excluded from the expanded pool entirely (not
"included with a warning") - training a model against physically
impossible SOH labels would corrupt every downstream fit that touches
these battery's rows, not just look slightly worse. `rul_labels.py`
ITSELF is left completely UNTOUCHED (it worked correctly for the
original 32 batteries and is shared by scripts this session doesn't
touch) - this is an exclusion applied at the expanded-pool level only.
"""

EXCLUDED_NASA_BATTERIES = {
    "B0033", "B0034", "B0036", "B0038", "B0039", "B0040", "B0041",
    "B0049", "B0050", "B0051",
}
EXCLUDED_MIT_BATTERIES = {
    "b1c18", "b2c44", "b1c0", "b2c12",
}
EXCLUDED_BATTERIES = EXCLUDED_NASA_BATTERIES | EXCLUDED_MIT_BATTERIES

# B0052 is separately excluded (too_few_cycles, only 3 usable cycles
# after the empty-Capacity fix - see data/processed/phase1_expanded_failures.csv)
# by run_phase1_features_expanded.py itself; not repeated here since it
# never made it into hi_table_expanded.parquet in the first place.
