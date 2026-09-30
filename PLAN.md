# PLAN.md — toolkit expansion pass (multi-source candidate → federated → label-efficient → passport → site → release)

This file is the durable source of truth for the current multi-phase plan.
It exists specifically so the plan survives conversation-context compaction
(the addendum below was lost once already and had to be re-supplied by the
user on 2026-09-29). **Read this file at the start of any session resuming
this work, before re-deriving scope from `DEVELOPMENT_LOG.md` alone.**

Full narrative detail, numbers, and verification evidence for every
completed phase live in `DEVELOPMENT_LOG.md` (chronological) and
`PAPER_RESULTS.md` (results-focused). This file is the checklist/spec, not
the log — update status markers here, write the actual narrative there.

Process rules for every phase below (per explicit user instruction,
2026-09-29): log everything in `DEVELOPMENT_LOG.md` and `PAPER_RESULTS.md`,
**commit and push per phase**, give the user a progress update after each
phase.

## Status summary (as of 2026-09-29)

- **Phase 0** (live app audit): COMPLETE. 4 findings, see DEVELOPMENT_LOG.md
  "Phase 0: live app audit".
- **Phase 1** (data completion): COMPLETE. BatteryLife sources confirmed
  never subsampled (nothing to rebuild); Tongji integrated (130 batteries,
  59,028 rows, rest-period check inconclusive - disclosed, not forced);
  EVBattery ruled NOT FEASIBLE (no discharge data + wrong label type).
- **Phase 2** (multi-source promotion candidate): COMPLETE. Gate table
  built and corrected (a NaN-encoder bug was caught and fixed mid-phase -
  see "CRITICAL correction" / "Corrected Phase 2 re-run" in the log).
  Source-balanced retrain tried and FAILED decisively (equal-total-weight
  collapses 15/16 sources to negative R2 - 153x source-size variance). OC-
  SVM sanity check run. **Decision adopted and WIRED**:
  `USE_MULTISOURCE_CANDIDATE = True` in `src/live_inference.py` - NASA/MIT
  stay on the deployed base model, every other known source + every
  uploaded/unknown battery routes to `_candidate_multisource.json` /
  `_candidate_ica_encoder.pt`. Verified via
  `src/verify_candidate_routing_production_code.py` against the real
  production function (100% correct routing, 0 exceptions, plausible R2
  per dataset). No deployed model file was touched by any of this -
  confirmed by md5 against the pre-pass baseline commit.
- **OC-SVM decision — CORRECTED 2026-09-29** (supersedes the "move to
  Research" recommendation recorded in the log): the candidate OC-SVM is
  NOT being retired. See "OC-SVM correction" below and the Phase 3 item
  that wires it.
- **`build_report_context.py` fix** (2026-09-29, out-of-band bugfix, not a
  phase of this plan): the LLM report generator's TreeSHAP feature
  construction had gone stale (23-dim, pre-reformulation) vs the deployed
  model's real 25-dim input. Fixed to reuse
  `live_inference.build_reformulated_hi_vector` directly. See
  DEVELOPMENT_LOG.md "LLM health-report re-check...".
- **Phase 2B** (federated multi-source learning): COMPLETE, 2026-09-30.
  Result: NEGATIVE, not adopted. Federated (Flower's `FedXgbBagging`,
  native XGBoost federated confirmed unavailable in this environment)
  never clearly beats the Phase 2 candidate's centralized pooling (wins
  on 13/16 sources outright) and does NOT fix NASA/MIT's crowded-out
  problem - MIT's federated R2 (-33.9) is far worse than its already-bad
  centralized R2 (-4.6). A companion "tempered centralized" experiment
  (n^0.5 group weighting) also failed to fix NASA/MIT and collapsed
  14/16 other sources - closes the open question from Phase 2d's
  balanced-retrain collapse: any departure from natural row-count
  weighting tested so far causes collapse, not just the extreme
  equal-weight case. **Decision: the existing Phase 2 routing (unweighted
  centralized candidate + dataset-identity routing) stands unchanged -
  no code change from this phase.** Two real implementation bugs were
  found and fixed via direct verification before trusting any result
  (Flower's own `aggregate()` silently drops multi-tree client payloads
  down to 1 tree; a self-built off-by-one in the fix's own batched
  version) - see DEVELOPMENT_LOG.md for both. Full results:
  `outputs/toolkit_phase2b_federated_results.csv`.
- **Phase 2C** (label-efficient checkpoints): COMPLETE, 2026-09-30.
  Result: at the 4 tested budgets (5/10/20/40 labels/battery), NO
  budget reaches a usable pooled coverage target (best case, 40 labels:
  51% mean coverage, still far below the 90% target). Highly variable
  by source (XJTU reaches 76%/92% late-life at budget=40; hnei/rwth/
  mich stay near-zero late-life coverage even at 40). Life-stage-
  weighted checkpointing consistently underperforms even spacing.
  Uncertainty-triggered ~ties fixed-every-n throughout - the adaptive
  trigger doesn't meaningfully help at these budgets. No forced
  minimum-labels recommendation given - reported honestly as "none of
  these budgets are enough." `online_conformal.py` extended with the
  3 schedule implementations + a real no-lookahead test for the
  schedules themselves (distinct from the recursion's own existing
  test). Full results: `outputs/toolkit_phase2c_label_efficient.csv`.
- **Phase 3: PARTIAL, 2026-09-30.** Done and directly verified (not
  just code-reviewed): the `river` import guard (moved from app.py's
  top level into `render_streaming_twin_tab`, confirmed via a real
  simulated-absence AppTest run that the rest of the app is
  unaffected); RUL hidden (not just captioned) for any out-of-domain
  battery in 4 places in app.py, verified against the exact Phase 0
  Finding 4 HNEI-upload scenario (was "3036 cycles", now "not
  available" with a reason). **NOT done, explicitly**: the OC-SVM
  rewiring (malformed-data check on uploads from `_candidate_
  ocsvm.pkl`; retire the deployed OC-SVM's 100%-flagging warning) and
  the nearest-source trust report it depends on for the "unfamiliar
  battery" message - no trust-report module exists in this codebase
  yet (checked directly), it needs to be built as new infrastructure,
  not just wired. Digital-twin next-measurement UI (from Phase 2C's
  schedule) also deferred with it. Pick up next: build the
  nearest-source trust report first (a lookup against each known
  source's own measured transfer error, e.g. from the Phase 2 gate
  table / LODO results), then wire the OC-SVM correction and the
  digital-twin UI on top of it.
- **Phase 3B, 4, 5**: NOT STARTED. Specs below.

## OC-SVM correction (2026-09-29, supersedes the Phase 2e "REPLACE...move to Research" recommendation)

The original recommendation (see DEVELOPMENT_LOG.md "OC-SVM sanity check,
rerun with the candidate's OWN exact normalization stats") read the 0%
Gaussian-noise detection rate as a failure. **Corrected framing**: 1x BMS-
level sensor noise is normal, expected data - NOT flagging it is correct
behavior, not a miss. Read against what it's actually good at: the
candidate OC-SVM catches swapped V/I columns, capacity-scaled-x10, and
truncated cycles at 100%, with only a 0.6% in-domain false-flag rate. That
combination makes it a good **input-sanity check** (malformed/corrupted
upload detector), not a domain/anomaly detector (it was never well-suited
to the latter - that job belongs to the nearest-source trust report).

**Action (Phase 3 scope, not yet implemented)**:
- Wire `models/_candidate_ocsvm.pkl` (+ its scaler) as a "data looks
  malformed" check on uploads specifically - plain-language message
  listing likely causes: swapped columns, wrong units, incomplete cycle.
- Use the nearest-source trust report (not OC-SVM) for the "this is an
  unfamiliar battery/source" message - a different question than "is this
  data malformed."
- Retire the currently-deployed OC-SVM's 100%-flagging out-of-domain
  warning (`models/ocsvm_model.pkl`/`ocsvm_scaler.pkl` - the one that
  flags everything non-NASA/MIT, uninformatively) - superseded by the two
  items above.

## Phase 2B — Federated multi-source learning

**Goal**: does a federated formulation (no raw rows ever leaving their
source) recover the Phase 2 candidate's pooled-centralized performance, or
close to it - and does it fix the NASA/MIT "crowded out" problem Phase 2
found, without the catastrophic collapse the naive equal-weight balanced
retrain produced?

**Clients**: one client per source, with sibling families grouped into a
single client: {stanford, stanford_2}, {mich, mich_exp}, and NASA grouped
with "NASA-derived" data. Same 16-source pool as Phase 2 (14 client groups
total after grouping).

**Method**: XGBoost's native federated training first choice; fall back to
Flower's XGBoost strategy (`flwr.server.strategy` federated-XGBoost, e.g.
bagging/cyclic) only if native federated XGBoost proves impractical for
this project's setup - **the choice must be justified in
`DEVELOPMENT_LOG.md` either way** (native XGBoost federated support is
newer/more limited than Flower's wrapper; document which constraint, if
any, forced the fallback). Same feature set and hyperparameters as the
Phase 2 candidate (no re-tuning - isolates the federated-vs-centralized
question from a hyperparameter question).

**Evaluation protocol**: LODO family-holdout (this project's existing,
established LODO methodology - NOT the Phase 2 gate's own weaker
"battery-level split within every source" protocol). Per target metric,
report three columns: **federated**, **centralized pooled** (the Phase 2
candidate), **NASA+MIT-only** (the original deployed model) - R2, RMSE,
MAE, each with battery-level bootstrap CIs, plus the **gap to centralized**
stated explicitly per source.

**Client weighting - three variants compared**:
1. Sample-weighted (proportional to each client's row count - the Phase 2
   candidate's own default).
2. Uniform (every client equal weight regardless of size).
3. **Tempered**: client weight proportional to `n^0.5` (a middle ground
   between the two above).

Additionally: run the tempered (`n^0.5`) weighting **once in the
centralized model** too, and report explicitly whether it fixes NASA/MIT's
loss-to-routed problem (Phase 2's finding) **without** reproducing the
catastrophic collapse the equal-total-weight (Phase 2d) balanced retrain
produced. This is a direct, disclosed test of whether the collapse was
specific to equal-TOTAL-weighting's extremeness or a more general property
of any non-uniform-by-row-count weighting.

**Hard requirement**: assert in code, not just claim in prose, that no raw
row ever leaves its client boundary during federated training (this is the
entire point of the federated formulation - verify it, don't just
architect for it).

**Deliverables**: a new `src/run_toolkit_phase2b_federated.py` (or
similarly named script(s)), a results table/CSV in `outputs/`, a
`DEVELOPMENT_LOG.md` entry with the full comparison + the native-vs-Flower
justification + the tempered-weighting finding, a `PAPER_RESULTS.md`
update, commit + push.

## Phase 2C — Label-efficient checkpoints

**Goal**: at a fixed, equal label budget per battery, which checkpoint-
selection *policy* gets the best coverage/width tradeoff - and what's a
defensible minimum-labels recommendation.

**Prerequisite**: `src/online_conformal.py` (PID + nexCP recursion,
already built per Phase 3(a)'s first half per the log) - if any gap is
found while building on it, extend it, and add **a no-lookahead shuffle
test as a real unit test** (shuffle the true-label revelation order and
assert the schedule/interval computed at cycle N never changes based on
labels revealed after cycle N - this is the property that makes the whole
label-efficient claim honest, not a nice-to-have).

**Setup**: the model predicts every cycle; true SOH is revealed only at
chosen checkpoints. Compare three checkpoint-selection policies at four
equal label budgets (5, 10, 20, 40 labels per battery):
1. **Fixed every-N** (N derived from the budget and that battery's own
   cycle-life length).
2. **Uncertainty-triggered**: reveal a label when the conformal interval
   width crosses a threshold, OR an ADWIN drift detector fires (whichever
   condition the schedule is built on - state explicitly which, and why,
   in the log).
3. **Life-stage-weighted**: denser sampling late in life (where SOH
   changes faster / matters more for RUL), sparser early.

**Report per dataset**: coverage, rolling-20 minimum coverage, late-life
coverage, and mean interval width, each as a function of label budget (4
points per policy per dataset) - then **derive a recommended minimum-
labels rule** from the results (not asserted a priori).

**Hard requirement**: a schedule may only use information available up to
the current cycle - **assert this in code** (mirrors the no-lookahead
shuffle test above, applied to the schedule/policy logic itself, not just
the underlying conformal recursion).

**Deliverables**: extended/new `src/online_conformal.py` tests, a new
`src/run_toolkit_phase2c_label_efficient.py` (or similarly named), results
CSV(s), `DEVELOPMENT_LOG.md` entry with the derived minimum-labels rule,
`PAPER_RESULTS.md` update, commit + push.

## Phase 3 — existing plan, plus these additions

(The "existing Phase 3 as planned" items referenced by the user are the
ones already visible in `DEVELOPMENT_LOG.md`'s existing forward references:
Phase 3(a) online-conformal minimum-checkpoints - already done; Phase 3(c)
hiding RUL for out-of-domain/unfamiliar batteries; the general "wire
Phase 2's/2e's recommendations into the app" scope.)

**Additions from the 2026-09-29 addendum**:
- Guard the top-level `from digital_twin_streaming_river import
  StreamingDigitalTwinRiver` (which itself imports `river` at module
  level) so a missing `river` package disables **only** the Streaming
  Digital Twin tab, never crashes the whole app on every page load - this
  is Phase 0's own Finding 1, now promoted to an explicit Phase 3 action
  item.
- Digital twin also surfaces **which cycles it recommends measuring next**
  (Phase 2C's derived checkpoint rule), not just the current prediction.
- **RUL hidden for unfamiliar batteries**, with the reason stated in the
  UI - the HNEI 3036-cycle case (Phase 0 Finding 4) must now show "not
  available" with an explicit reason, not a bare number that's ~9 std devs
  outside the training distribution.
- OC-SVM correction's two wiring actions (input-sanity check on uploads;
  nearest-source trust report for the "unfamiliar battery" message;
  retire the old 100%-flagging warning) - see "OC-SVM correction" above.

## Phase 3B — Passport-style output

**Research first, cite sources, never invent fields**: EU Battery
Regulation 2023/1542's digital battery passport requirements, and the
Battery Pass consortium's public data attribute list. Only include fields
this project genuinely computes - do not pad the JSON with plausible-
looking fields nobody asked for and this project can't actually populate.

**Per-battery JSON, fields limited to what's genuinely computed**: SOH
estimate; interval + coverage method (which conformal method, calibrated
how); number of measured checkpoints actually used for this battery;
model version (base vs multisource-candidate vs federated, whichever is
live); nearest training source and its own measured transfer error (from
the gate-table-style evaluation, not invented); trust flag; input-sanity
result (from the OC-SVM wiring above); timestamp.

**Labeling requirement**: every surface exposing this JSON must be
explicitly labeled **"passport-style, not a certified battery passport"**
- this is not a compliance claim.

**Deliverables**: a research citation list (URLs/sources for the EU
regulation + Battery Pass attribute list) recorded in
`DEVELOPMENT_LOG.md`, the JSON schema + generator code, `PAPER_RESULTS.md`
update if relevant, commit + push.

## Phase 4 — site redesign, as planned, plus these additions

- **All 15 sources selectable** in the sidebar/UI (not upload-only) - this
  directly closes Phase 0's Finding 3 gap (9 BatteryLife sources + Tongji
  currently reachable only via CSV upload).
- **"Check my battery"** offers the passport-style JSON (Phase 3B) as an
  export/download.
- **"Live monitoring"** shows the recommended next measurement (Phase 2C's
  checkpoint rule), consistent with the digital-twin addition in Phase 3.
- **"Benchmark explorer"** adds: federated vs centralized (Phase 2B), and
  coverage vs label budget (Phase 2C).

## Phase 5 — release, as planned, plus these additions

- Package exposes the checkpoint scheduler (Phase 2C) and the passport
  export (Phase 3B) as part of its public API, not just internal-only.
- The benchmark script reports coverage vs label budget (Phase 2C),
  alongside whatever it already reports.

## Notes on scope discipline (carried forward from this project's own established conventions)

- Every phase gets its own `DEVELOPMENT_LOG.md` entry, written the same
  way prior entries are: state the real result even when it's not the one
  hoped for (see Phase 2's balanced-retrain collapse, EVBattery's
  infeasibility, Tongji's inconclusive rest-period check - none of these
  were softened to sound more finished than they are).
- "No deployed file touched" (or the opposite, stated plainly) gets
  confirmed and stated explicitly for every phase that touches models/ or
  `live_inference.py`/`app.py`, the same discipline used for every prior
  phase in this pass.
- Assertions the plan calls "hard requirements" above (no-raw-row-leakage
  in Phase 2B, no-lookahead in Phase 2C) are real code-level `assert`s,
  verified to actually fire under a genuine violation during development
  (not just present and untested), not just prose claims.
