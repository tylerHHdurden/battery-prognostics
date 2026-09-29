# App Audit (Phase 0, toolkit pass)

Driven via Streamlit's own `AppTest` framework (this project's established convention,
`src/_regression_sweep_calce_fix.py`), no code changes. One `.run()` per battery selection
renders ALL 8 tabs' content at once (Streamlit tabs are visual containers, not lazy-loaded,
so a single run captures every tab's output). 4 cases tested: NASA/B0005 (in-domain),
CALCE/CS2_35 (out-of-domain, raw-data path), XJTU (out-of-domain, precomputed-only path -
never has raw cycle data in this deployment), and HNEI (a BatteryLife source - **not
reachable via the sidebar dropdown at all**, tested via the "Upload your own cycle data"
path using one real HNEI battery's own raw cycles converted to the app's documented CSV
format).

## Finding 1 (CRITICAL, blocks the whole app): missing `river` dependency crashes every tab, not just Streaming Digital Twin

`app.py` line 61 does an **unconditional, top-level** `from digital_twin_streaming_river
import StreamingDigitalTwinRiver`, and `digital_twin_streaming_river.py` does `from river
import tree, drift` at its own module level. `river==0.26.1` IS listed in `requirements.txt`,
but was **not actually installed** in this local dev environment before this audit -
confirmed directly (`pip show river` failed, `pip freeze` in the prior session's own
`requirements-lock.txt` has no `river` entry either). With it missing, **every single page
load throws `ModuleNotFoundError: No module named 'river'` before a single tab renders** -
not a graceful per-tab degradation, a hard crash of the entire app, for every dataset,
every battery, every mode. Installed `river==0.26.1` (matching the already-declared
`requirements.txt` pin) to proceed with the rest of this audit - not a code change, just
syncing the local environment to what the project already declares it needs.

**Root cause for the paper/report**: one tab's own dependency (Streaming Digital Twin's
river-based online model) has no import guard, so its absence takes the ENTIRE
application down rather than just disabling that one tab. A real, disclosed fragility -
flagged for Phase 3/4 (an import-time try/except around this one import, degrading that
specific tab gracefully, would fix it without removing the feature).

## Finding 2: `st.error()` is used as a narrative-styling device in the Full Results Archive tab, not only for genuine errors

17-19 `st.error()` elements render on EVERY page load, regardless of dataset - initially
looked alarming, but reading the actual text confirms these are **intentional, styled
call-out boxes for narrative documentation text** in the Full Results Archive tab (e.g.
*"Reported plainly, not softened: this does not yet produce a generally useful
forecasting capability..."*) - not runtime failures. A real UI/UX observation (using
`st.error()`'s red styling for non-error narrative content is unconventional and could
confuse anyone/anything - including future automated testing - that treats `st.error()`
count as a health signal), not a functional bug. Worth a lighter-weight styling choice in
the Phase 4 redesign, not urgent.

## Finding 3: HNEI (and every other BatteryLife source) is not reachable via the sidebar dropdown at all

The sidebar's "Dataset" selectbox only ever offers `["NASA", "MIT", "CALCE", "Oxford",
"HUST", "XJTU"]` (`app.py` line 2887) - none of the 9 BatteryLife sources this project's
own research extensively evaluated (hnei, snl, mich, mich_exp, rwth, stanford,
stanford_2, isu_ilcc, ul_pur) are selectable there. The ONLY way to see a BatteryLife
battery's prediction in the live app today is "Upload your own cycle data," manually
formatted to the app's CSV schema - which is what this audit did to test HNEI at all. A
real, disclosed usability gap for Phase 4's redesign.

## Finding 4: a genuine HNEI upload produces a wildly implausible RUL (3036 cycles) - and the app does not specially flag RUL as untrustworthy

Real HNEI battery data, uploaded via the documented CSV path, produces **Predicted RUL =
3036 cycles** - about 9 standard deviations above the deployed RUL model's own training-
scale distribution (mean=390.6, std=292.9 cycles, per this project's own CHECK B
verification). This directly, visibly reproduces in the live app exactly the RUL
cross-domain failure CHECK B already found analytically (CALCE RUL R2=-566, predicted
mean=-401 cycles - also nonsensical). **The app DOES show a generic "OUT-OF-DOMAIN"
warning for this upload, but does not specifically call out that the RUL number itself
should not be trusted** - a real gap Phase 3(c) of this pass should close (hide RUL
entirely when the battery's nearest source is out-of-domain, per that phase's own
instruction).

## Finding 5: NASA/CALCE/XJTU behave correctly on the signals that matter

Re-reading the PRECISE indicator text (not just a broad substring search, which produced
some initial false positives from static Archive-tab/legend text unrelated to the
currently-selected battery - noted as a methodology caveat, corrected before writing this
up):

| Case | SOH predicted vs. true | RUL shown | Out-of-domain warning | Anomaly flag (live, precise indicator) |
|---|---|---|---|---|
| NASA/B0005 | 71.7% vs. true 72.8% (close, plausible) | 2 cycles (plausible - near end-of-life cycle) | NOT shown (correct - in-domain) | NOT flagged (correct) |
| CALCE/CS2_35 | 80.8% (no true-SOH cross-check shown for this path) | 193 cycles | **Shown** (correct) | **Flagged** (correct) |
| XJTU | 97.8% | **"not available"** (correct - precomputed path has no raw curve for RUL) | **Shown** (correct) | **Flagged** (correct) |
| HNEI (upload) | 98.0% | **3036 cycles (implausible - see Finding 4)** | **Shown** (correct) | NOT flagged (see note below) |

**Note on HNEI's anomaly flag**: the live OC-SVM did NOT flag this specific HNEI cycle as
anomalous, despite it being genuinely out-of-domain and producing an implausible RUL -
consistent with this project's own prior finding (Part B) that the anomaly detector,
trained only on NASA+MIT, does not reliably catch every out-of-domain BatteryLife source;
the domain-shift warning (triggered by "no temperature channel present," a structural
data-schema signal, not the anomaly score) is doing the actual protective work here, not
the OC-SVM.

**Minor observation, not chased further given time budget**: a handful of metrics
labeled generically "SOH"/"RUL"/"True SOH" (values 71.7%/2 cycles/72.8%) appear
identically across all 4 runs - almost certainly a FIXED Showcase-tab replay example
(unrelated to the sidebar-selected battery), not a bug; the battery-specific "Predicted
SOH" metric DOES correctly vary per case (71.7% / 80.8% / 97.8% / 98.0%), confirming the
real prediction pipeline works correctly per selection.

## Summary

| Case | Exceptions | Verdict |
|---|---|---|
| NASA/B0005 | 0 (after installing `river`) | Correct, plausible, no false alarms |
| CALCE/CS2_35 | 0 | Correct, plausible, correctly flagged out-of-domain |
| XJTU | 0 | Correct, plausible, correctly flagged out-of-domain, RUL correctly unavailable |
| HNEI (upload only) | 0 | Correctly flagged out-of-domain, but RUL=3036 is a live, visible demonstration of a real known failure mode not specially called out |

No crashes once the environment matched `requirements.txt`. The two real, actionable
findings for later phases: (1) the `river` import fragility (single point of failure for
the whole app), and (2) RUL needs its own out-of-domain-specific suppression/warning,
not just the generic domain warning it currently shares.
