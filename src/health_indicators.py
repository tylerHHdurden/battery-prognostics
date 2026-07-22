"""
16 Health Indicators (HIs), computed per-cycle from a data_adapters cycle
record: {"charge": {t,V,I,T}, "discharge": {t,V,I,T}, "discharge_capacity"}.

*** IMPORTANT — READ BEFORE TRUSTING ANY VALUE FROM THIS MODULE ***
The 16 acronyms given in the task (CDECT; ICHV, UVP, SCV, VDEDT, VIECT,
LVP; MATC, MATD, MATDL, MET; TCCC, TCVC, TECD, TEVD, TEVI) are not
accompanied by formal definitions. I was told to "comment every definition
assumption" rather than skip them, so every single one below is MY
best-effort, physically-reasonable expansion of the acronym, chosen to be
(a) plausible given standard battery-HI literature (CC/CV timing,
equal-voltage/equal-time sampling, thermal spread, energy throughput —
all real, commonly-used families of HI), and (b) computable uniformly
across NASA/CALCE/MIT with the fields actually available.
If the source the user has in mind defines these differently, the
FEATURE NAMES will still line up 1:1 for an easy swap-in of the real
formula later — only the arithmetic inside each function would need to
change.

Known dataset limitation: CALCE has no Temperature column (see
src/data_adapters.py), so MATC/MATD/MATDL are NaN for every CALCE cycle.
This is logged, not silently dropped.
"""

from __future__ import annotations

import numpy as np

HI_NAMES = [
    "CDECT", "ICHV", "UVP", "SCV", "VDEDT", "VIECT", "LVP",
    "MATC", "MATD", "MATDL", "MET",
    "TCCC", "TCVC", "TECD", "TEVD", "TEVI",
]


def _cc_phase_start_end_idx(I: np.ndarray, tol_frac: float = 0.05) -> tuple[int, int]:
    """
    (start, end) indices of the constant-current plateau. Uses the median
    of the *middle* half of the array as the CC set-point reference (robust
    to the brief current-ramp/sensor-settling transient often seen in the
    first 1-2 samples, and to the CV taper at the very end), then walks
    outward from the middle to find where the signal leaves that band.
    """
    n = len(I)
    if n < 4:
        return 0, max(n - 1, 0)
    mid_lo, mid_hi = n // 4, 3 * n // 4
    ref = np.median(I[mid_lo:mid_hi])
    if ref == 0:
        return 0, n - 1
    band = np.abs(I - ref) < tol_frac * np.abs(ref)

    mid = n // 2
    start = mid
    while start > 0 and band[start - 1]:
        start -= 1
    end = mid
    while end < n - 1 and band[end + 1]:
        end += 1
    return start, end


def _trapz_energy_wh(V: np.ndarray, I: np.ndarray, t: np.ndarray) -> float:
    """Energy = integral of |V*I| dt, converted from J (V*A*s) to Wh."""
    if len(t) < 2:
        return np.nan
    power = np.abs(V * I)
    energy_ws = np.trapezoid(power, t)
    return energy_ws / 3600.0


def compute_health_indicators(cycle: dict) -> dict:
    ch, dc = cycle["charge"], cycle["discharge"]
    Vc, Ic, Tc, tc = ch["V"], ch["I"], ch["T"], ch["t"]
    Vd, Id, Td, td = dc["V"], dc["I"], dc["T"], dc["t"]

    out = {}

    # --- Group C: time-based (well-established CC/CV timing HIs) ---
    cc_start_c, cc_end_c = _cc_phase_start_end_idx(Ic)
    # TCCC = Time of Constant-Current Charging: duration of the flat-
    # current plateau segment of the charge curve (excludes the brief
    # startup transient before the CC plateau begins).
    out["TCCC"] = float(tc[cc_end_c] - tc[cc_start_c]) if len(tc) > 1 else np.nan
    # TCVC = Time of Constant-Voltage Charging: remainder of the charge
    # duration after the CC plateau ends (tapering-current CV tail).
    out["TCVC"] = float(tc[-1] - tc[cc_end_c]) if len(tc) > 1 else np.nan

    cc_start_d, cc_end_d = _cc_phase_start_end_idx(Id)
    # TECD = Time Elapsed during Constant-current Discharge: duration the
    # discharge current stays within its CC plateau band (usually ~the
    # whole discharge for these single-rate cyclers).
    out["TECD"] = float(td[cc_end_d] - td[cc_start_d]) if len(td) > 1 else np.nan

    if len(Vd) > 1:
        v_range = Vd.max() - Vd.min()
        # TEVD = Time Elapsed for Voltage Drop: time from discharge start
        # until voltage first falls to 50% of this cycle's own
        # (max-min) discharge voltage range — a relative, chemistry-
        # agnostic drop threshold (rather than a hardcoded absolute volt
        # value, since this must work across NASA/CALCE/MIT chemistries).
        thresh_50 = Vd.max() - 0.5 * v_range if v_range > 0 else Vd.max()
        below = np.where(Vd <= thresh_50)[0]
        out["TEVD"] = float(td[below[0]] - td[0]) if len(below) else np.nan

        # TEVI = Time Elapsed for Voltage Interval: duration spent
        # traversing the 80%->20% band of the discharge voltage range
        # (distinct from TEVD's single-threshold crossing time).
        thresh_80 = Vd.max() - 0.2 * v_range
        thresh_20 = Vd.max() - 0.8 * v_range
        idx80 = np.where(Vd <= thresh_80)[0]
        idx20 = np.where(Vd <= thresh_20)[0]
        if len(idx80) and len(idx20):
            out["TEVI"] = float(td[idx20[0]] - td[idx80[0]])
        else:
            out["TEVI"] = np.nan
    else:
        out["TEVD"] = out["TEVI"] = np.nan

    # --- Group A: voltage/capacity based ---
    # CDECT = Constant-Discharge Energy at Cutoff Time: total discharge
    # energy delivered by end of discharge (Wh).
    out["CDECT"] = _trapz_energy_wh(Vd, Id, td)

    if len(Vc) > 1:
        vc_max = Vc.max()
        # ICHV = Interval of Charging at High Voltage: time spent with
        # charge voltage above 90% of this cycle's peak charge voltage
        # (proxy for how long the high-voltage/CV tail persists).
        high_v_mask = Vc >= 0.9 * vc_max
        out["ICHV"] = float(tc[high_v_mask][-1] - tc[high_v_mask][0]) if high_v_mask.sum() > 1 else 0.0
    else:
        out["ICHV"] = np.nan

    if len(Vd) > 1:
        v_range = Vd.max() - Vd.min()
        low_thresh = Vd.min() + 0.1 * v_range if v_range > 0 else Vd.min()
        low_mask = Vd <= low_thresh
        # UVP = Under-Voltage Point: total discharge time spent below the
        # bottom 10% of this cycle's discharge voltage range.
        out["UVP"] = float(td[low_mask][-1] - td[low_mask][0]) if low_mask.sum() > 1 else 0.0
        # LVP = Low-Voltage Point: elapsed time (from discharge start)
        # until voltage FIRST crosses into that bottom-10% band — a
        # timestamp, as opposed to UVP's duration.
        low_idx = np.where(low_mask)[0]
        out["LVP"] = float(td[low_idx[0]] - td[0]) if len(low_idx) else np.nan
        # SCV = Slope of Capacity vs Voltage: coarse average dQ/dV over
        # the discharge, using this cycle's total discharge capacity over
        # its voltage range as a single average-slope scalar.
        out["SCV"] = float(cycle["discharge_capacity"] / v_range) if v_range > 0 else np.nan
        # VDEDT = Voltage Drop rate at End of Discharge Time: mean dV/dt
        # over the last 10% of discharge samples (rate of voltage
        # collapse approaching cutoff — sensitive to internal resistance).
        tail_n = max(2, len(Vd) // 10)
        out["VDEDT"] = float(np.mean(np.diff(Vd[-tail_n:]) / np.diff(td[-tail_n:]))) if tail_n > 1 else np.nan
    else:
        out["UVP"] = out["LVP"] = out["SCV"] = out["VDEDT"] = np.nan

    if len(Vc) > 1 and len(tc) > 1:
        # VIECT = Voltage Increment at Equal Charge Time: charge voltage
        # sampled at a fixed elapsed time into charge (capped at half this
        # cycle's own charge duration, since absolute cutoff times differ
        # across cyclers/chemistries) — a classic equal-time HI that
        # tracks voltage rising faster as internal resistance grows.
        fixed_t = min(300.0, 0.5 * (tc[-1] - tc[0]))
        sample_idx = np.searchsorted(tc - tc[0], fixed_t)
        sample_idx = min(sample_idx, len(Vc) - 1)
        out["VIECT"] = float(Vc[sample_idx])
    else:
        out["VIECT"] = np.nan

    # --- Group B: temperature / energy based ---
    if Tc is not None and len(Tc) > 0:
        out["MATC"] = float(np.mean(np.abs(Tc)))
    else:
        out["MATC"] = np.nan
    if Td is not None and len(Td) > 0:
        out["MATD"] = float(np.mean(np.abs(Td)))
    else:
        out["MATD"] = np.nan
    # MATDL = Mean Absolute Temperature Difference across Load: |mean
    # charge temp - mean discharge temp| for this cycle (thermal
    # asymmetry between charge and discharge loads). This is the single
    # most uncertain acronym expansion in this module.
    if not np.isnan(out["MATC"]) and not np.isnan(out["MATD"]):
        out["MATDL"] = float(abs(out["MATC"] - out["MATD"]))
    else:
        out["MATDL"] = np.nan
    # MET = Mean Energy during Test: average of charge and discharge
    # energy throughput for the full cycle (Wh).
    charge_energy = _trapz_energy_wh(Vc, Ic, tc)
    out["MET"] = float(np.nanmean([charge_energy, out["CDECT"]]))

    return out
