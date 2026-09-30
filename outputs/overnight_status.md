# Overnight status - FINAL (run finished; nothing started after the work below)
Branch: **site-redesign** (cut from master 1d3bb23). **Nothing was pushed** (no branch, no tag, nothing to master); bat-pro.streamlit.app and the staging app were not touched. report/ is git-ignored and untouched except the two new local PDFs below.

## What is done (all committed locally on site-redesign)
1. **MORNING_RUNBOOK.md** - review list, staging steps, merge/push commands, reboot + smoke test, rollback commands.
2. **Site redesign (Phase 4)** - first screen with one-sentence purpose and limits; three compact entry paths with step-by-step details; result card at the top (SOH + 90% interval, RUL only under the existing condition, trust state in plain words with 81.9% / 11.0%, passport pointer); upload flow with downloadable sample file, column guide and clear errors; neutral/amber only (every green success box -> neutral info); accessible contrast/focus CSS and mobile CSS; no external font request; use_container_width -> width; lazy tabs; World Model tab relabelled "(experiment)" (kept, not deleted); 47 one-line status banners + relabelled Showcase headline cards; benchmark explorer (federated vs centralized, coverage vs label budget); **the 9 BatteryLife sources and Tongji are selectable in Browse mode** (precomputed data, multisource candidate, RUL hidden, caution caption).
   - Cold render (AppTest): Oxford 36.2 s -> 7.7 s, NASA 45.3 s -> 9.1 s; warm Oxford 7.8 s -> 0.5 s, NASA 14.0 s -> 0.8 s.
   - Regression: 6-dataset + HNEI matrix gives the same SOH/RUL strings as before (local machine with raw data: NASA 71.7%/2 cycles, MIT 82.5%/0, CALCE 54.0%, Oxford 73.5%, HUST 79.3%, XJTU 95.2%, HNEI upload 97.3%; 0 exceptions) - identical after the redesign, after the layout change and after wiring the new sources. On a Unix-line-ending clone without raw data (as on Streamlit Cloud) the same app gives NASA 71.7%, MIT 95.9% (precomputed path, no RUL), other datasets identical - same as the pre-redesign clone smoke test.
   - Visual check on a Unix-line-ending clone at 1400x1000 and 390x844 (Oxford, flagged upload, unflagged upload; all 8 tabs opened): 0 exceptions, 0 green elements, no horizontal scroll, result card present (`outputs/site_redesign/site_visual_result.json` + screenshots).
   - `pytest tests -q`: 25 passed (about 8-10 s).
3. **Hardening** - README rewritten; tests/ (25 quick tests); DATA_AVAILABILITY.md HUST fixed (Energy & Environmental Science 2022, vol. 15, Ma G. et al.); REPRODUCIBILITY.md corrected and requirements.txt checked (pyarrow==24.0.0 added); PAPER_RESULTS.md: all 16 tables carry a status label and a source column (1 conflict, 1 unsourced item left, see `outputs/site_redesign/paper_results_labelling_note.md`); open-release drafts in `release/` (LICENSE, CITATION.cff, MODEL_CARD, DATA_CARD, KNOWN_LIMITATIONS, RELEASE_CHECKLIST, PACKAGING_PLAN) - nothing published.
4. **PLAN.md leftovers** - Phase 3B polish (passport adds interval method, measured checkpoints used, input-sanity result; research citations logged in DEVELOPMENT_LOG.md); Phase 5 preparation (toolkit table set incl. coverage-vs-label-budget in scripts/reproduce_paper.sh; packaging plan draft).
5. **Walkthrough PDF (10 pages)** - `report/walkthrough/Project_Walkthrough.pdf` (+ .docx, build_compact.py, compact/*.md). I corrected one error in it: the old trust rule's 47.5% is VERIFIED on the corrected profiles (superseded by the ROC rule), not INVALIDATED. Figures take about 1.3 pages (you asked 2-3).
6. **Addendum** - `report/Addendum_Post_Submission.pdf` (2 pages) + .md + build_addendum.py. **No six-dataset number in the submitted report is SUPERSEDED.** It also lists the wording in the submitted report that is now out of date (Abstract last sentence, Sections 5.9, 6.2, 6.3).

## Decisions I made for you (conservative defaults)
- **Did not push site-redesign.** Only one tracked file on the branch contains names (release/CITATION.cff, a draft; no register numbers anywhere), and you excluded names from public pushes earlier. Push it yourself when happy: `git push origin site-redesign`.
- Kept the World Model tab (relabelled) rather than deleting it; kept runtime.txt (python-3.11, ignored by Cloud) and the unused packages in requirements.txt so the deployed environment does not change; added only the pyarrow pin.
- Archive text that conflicts with PAPER_RESULTS (map items C1-C10, e.g. R2 0.917 vs 0.978, CALCE coverage 6.1% vs 4.3%) was labelled, not rewritten.
- Replaced every green success box with a neutral info box (even in the static research narrative).
- Extension sources in Browse mode: per-source flagged counts at the last cycle - ul_pur 0/10, hnei 0/14, snl 18/55, mich 1/40, mich_exp 0/18, rwth 1/10, stanford 0/6, stanford_2 0/8, isu_ilcc 9/9, tongji 3/130. isu_ilcc flags everything, consistent with its known false-alarm behaviour in validation. Their streaming-twin and comparison views use the base model, not the candidate.

## Things you should know / not finished
- **Hot-update ImportError reproduced locally**: after I pulled new code into a running local server it raised `cannot import name 'EXTENSION_SOURCES'` until I restarted it - same cause as the live incident. Reboot after every push stays the rule (PLAN.md); no reload guard added.
- Not done (needs you or is out of scope): staging/live deployment and smoke tests of the redesign; a from-scratch install on Python 3.14 (tests ran on the existing 3.14.2 venv against a clean clone); the packaging refactor; everything public (GitHub release, DOI, PyPI, new tags); the informational "no temperature channel" note is keyed to dataset != HUST and may be wrong for some extension sources; Tongji is missing from DATA_AVAILABILITY.md.
- Open small items: PAPER_RESULTS rwth old-table conflict (0.482 vs 0.540, superseded table), stanford pre-rerun family R2 0.889 (PAPER_RESULTS) vs 0.888 (CSV), OC-SVM pickles saved with scikit-learn 1.8.0 (warn under 1.9.0, still work).

## Commits on site-redesign (master..site-redesign, oldest first)
- 6794a93 Overnight status file
- bd55da9 Site redesign: first screen, three entry paths, result card, upload help, neutral colours, lazy tabs, no external font, width API, mobile/contrast CSS, status banners (47) and labelled headline cards
- b436604 Add site visual check and benchmark scripts
- 3b779e8 status update
- 519a5f8 Site redesign: compact entry paths, result card slot above the raw-curve plot, step-by-step details expander
- 658c911 Hardening: README rewrite, tests/ (25 quick tests, ~10 s), HUST journal fixed in DATA_AVAILABILITY, REPRODUCIBILITY/requirements checked (pyarrow added), PAPER_RESULTS tables labelled with status and source, open-release drafts in release/ (nothing published), status-label map
- b8237e4 Phase 3B polish: passport adds interval method, checkpoints used, input-sanity result; research citations logged
- 1ab6b6f Benchmark explorer: federated vs centralized (Phase 2B) and coverage vs label budget (Phase 2C) from result CSVs with status labels
- 92fa716 Morning runbook
- 6b76d39 DEVELOPMENT_LOG: overnight site redesign entry
- c2c1bec status update
- 3def164 Phase 5 preparation: toolkit table set in reproduce_paper.sh (coverage vs label budget), packaging plan draft (nothing published)
- c5b39a2 Phase 4: the 9 BatteryLife sources and Tongji are selectable in Browse mode (precomputed data, multisource candidate, RUL hidden, caution caption); six validated datasets unchanged (regression identical)
- ed053cb Site: caption contrast reaches nested text; entry-path copy no longer claims every number is labelled

## Files to review first
- outputs/MORNING_RUNBOOK.md, outputs/site_redesign/*.png (screenshots), outputs/site_redesign/site_visual_result.json, outputs/site_redesign/status_labels_map.md, outputs/site_redesign/sources_selectable_report.md
- app.py (diff vs master), src/live_inference.py, src/battery_passport.py, README.md, release/, PAPER_RESULTS.md, report/Addendum_Post_Submission.pdf, report/walkthrough/Project_Walkthrough.pdf
