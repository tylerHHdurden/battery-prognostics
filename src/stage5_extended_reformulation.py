"""
Stage 5 follow-on, item 1: tests item 2's hypothesis directly - does
reformulating SCV/MATD/VIECT (the non-duration absolute-scale features
found driving near-total domain-classifier separation on Oxford/HUST/
XJTU/CALCE) the same way Stage 1.1 reformulated ICHV/TEVD/TEVI reduce
that separation and improve zero-retrain generalization?

EXPERIMENTAL, ADDITIVE ONLY - does not modify stage1_common.py or any
canonical/deployed feature-building code. Nothing here is imported by
live_inference.py or app.py, unless/until explicitly promoted.

Reformulation choice, reasoned per-feature (not applying one blanket
rule) - the same distinction Stage 1.1 itself relied on to determine
duration features needed a RATIO not a delta:
- SCV = discharge_capacity / voltage_range (health_indicators.py) - a
  MULTIPLICATIVELY capacity-scaled quantity (directly proportional to
  a cell's own nominal capacity, confirmed in item 2: Oxford 0.74 Ah,
  HUST 1.1 Ah, XJTU 2.0 Ah, NASA/MIT ~1.1-1.9 Ah). A RATIO against the
  battery's own early-life SCV removes this multiplicative nominal-
  capacity-scale factor cleanly, the same logic as ICHV/TEVD/TEVI's
  own ratio reformulation - SCV_rel = SCV(cycle_n)/SCV(baseline).
- MATD = mean |temperature| during discharge (health_indicators.py) -
  an ADDITIVELY offset quantity (each dataset's own fixed ambient/
  thermal-chamber setpoint, e.g. Oxford's 40degC vs. NASA/MIT's
  ambient ~24degC), not a multiplicative scale. A ratio of Celsius
  temperatures is not physically meaningful (and risks a near-zero-
  denominator edge case near 0degC that a delta avoids entirely) - a
  DELTA against the battery's own early-life MATD removes the fixed
  ambient offset while preserving any real within-battery thermal
  drift - MATD_rel = MATD(cycle_n) - MATD(baseline).
- VIECT = a raw voltage reading at one fixed point in the cycle
  (health_indicators.py) - chemistry-dependent absolute scale (LFP's
  ~3.2V plateau vs. NCM's ~3.6-3.7V), also an ADDITIVE offset in
  nature (a voltage "shift", not a multiplicative rescaling - unlike
  capacity, cell voltage doesn't scale with nominal capacity). Same
  reasoning as MATD: a DELTA against the battery's own early-life
  VIECT - VIECT_rel = VIECT(cycle_n) - VIECT(baseline).
- MET = mean charge/discharge energy throughput, Wh (health_
  indicators.py) - energy = capacity(Ah) x voltage(V), a MULTIPLICATIVE
  combination of two scale factors this project already treats as
  multiplicative on their own (capacity via SCV's own ratio treatment;
  voltage-level via chemistry). Added in this follow-up pass (the
  original item 1 explicitly scoped MET out; identified there as
  XJTU's likely remaining driver - z=5.49, by far the largest
  surviving z-score of any feature/dataset after SCV/MATD/VIECT were
  fixed - and confirmed worth reformulating on the same evidence
  rather than guessed at). Same ratio treatment as SCV -
  MET_rel = MET(cycle_n)/MET(baseline).

Same baseline convention as add_reformulated_duration_features
(battery's own cycle_idx==10 row, median-of-own-cycles fallback if
missing), same near-zero-baseline guard for the two ratios (SCV_rel,
MET_rel) - deltas need no such guard (no division).
"""
import numpy as np
import pandas as pd

BASELINE_CYCLE = 10
RATIO_FEATURES = ["SCV", "MET"]
DELTA_FEATURES = ["MATD", "VIECT"]


def _baseline_map(hi_df: pd.DataFrame, feat: str) -> dict:
    baseline_rows = hi_df[hi_df["cycle_idx"] == BASELINE_CYCLE]
    out = {}
    for bid, g in hi_df.groupby("battery_id"):
        base_row = baseline_rows[baseline_rows["battery_id"] == bid]
        if len(base_row) > 0:
            out[bid] = float(base_row[feat].iloc[0])
        else:
            out[bid] = float(g.sort_values("cycle_idx")[feat].iloc[0])
    return out


def add_scv_matd_viect_reformulated(hi_df: pd.DataFrame) -> pd.DataFrame:
    hi_df = hi_df.copy()

    # SCV_rel, MET_rel - RATIO (multiplicative scale), near-zero-
    # baseline guard, same pattern as add_reformulated_duration_features.
    for feat in RATIO_FEATURES:
        base = _baseline_map(hi_df, feat)
        rel = np.full(len(hi_df), np.nan, dtype=float)
        n_fallback = 0
        for bid, g in hi_df.groupby("battery_id"):
            b = base[bid]
            if not np.isfinite(b) or abs(b) < 1e-6:
                b = float(g[feat].median())
                n_fallback += 1
            if not np.isfinite(b) or abs(b) < 1e-6:
                continue
            idx = g.index
            rel[idx] = hi_df.loc[idx, feat].to_numpy(dtype=float) / b
        hi_df[f"{feat}_rel"] = rel
        print(f"[extended-reformulation] {feat} -> {feat}_rel (ratio): {n_fallback} batteries used "
              f"median-fallback baseline, {int((~np.isfinite(rel)).sum())} non-finite output values")

    # MATD_rel, VIECT_rel - DELTA (additive offset), no divide-by-zero risk.
    for feat in DELTA_FEATURES:
        base = _baseline_map(hi_df, feat)
        rel = np.full(len(hi_df), np.nan, dtype=float)
        for bid, g in hi_df.groupby("battery_id"):
            b = base[bid]
            idx = g.index
            if np.isfinite(b):
                rel[idx] = hi_df.loc[idx, feat].to_numpy(dtype=float) - b
        hi_df[f"{feat}_rel"] = rel
        print(f"[extended-reformulation] {feat} -> {feat}_rel (delta): "
              f"{int((~np.isfinite(rel)).sum())} non-finite output values")

    return hi_df


EXTENDED_FEATURE_MAP = {"SCV": "SCV_rel", "MATD": "MATD_rel", "VIECT": "VIECT_rel", "MET": "MET_rel"}


def extended_canonical_feature_cols(base_cols: list[str]) -> list[str]:
    """Swaps SCV/MATD/VIECT/MET for their _rel versions in an already-
    duration-reformulated canonical column list (leaves ICHV_rel/
    TEVD_rel/TEVI_rel/VDEDT untouched - VDEDT is a dV/dt RATE, already
    scale-invariant in a way the others weren't, never flagged as an
    outlier in item 2's z-score check)."""
    return [EXTENDED_FEATURE_MAP.get(c, c) for c in base_cols]
