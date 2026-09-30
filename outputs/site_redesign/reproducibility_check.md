# Reproducibility check (2026-10-01)

Method: `git clone --shared -c core.autocrlf=false -b site-redesign . <scratchpad>/cleanclone` (committed state only,
LF line endings; 0 CR characters in app.py, src/live_inference.py, requirements.txt), then scripted checks.

## Passed
- Every path named in REPRODUCIBILITY.md exists in the clone (scripts/reproduce_paper.sh and every
  `src/run_finalpass*.py` it calls, split_utils.py, run_conformal.py, run_groupkfold_cv.py, battery_split.json,
  bfa_selected_features_nasa_mit_only.txt, requirements-lock.txt, DATA_AVAILABILITY.md, verify/smoke scripts).
- `python -m pytest tests -q` passes in the clone: 25 passed in 9.6 s (32 s cold in the working tree). The sidecar
  check passes against committed blobs.
- Import closure (AST) from app.py, src/live_inference.py and src/battery_passport.py reaches 29 local modules; every
  third-party import in it is in requirements.txt. The one apparent miss, `flwr`, appears only in a docstring.
- `pip check` on the project venv (Python 3.14.2): no broken requirements.

## Problems found and fixed
1. `pyarrow` (needed by `pandas.read_parquet`, used by app.py and live_inference.py) was missing from
   requirements.txt; it is present only transitively via Streamlit. Added `pyarrow==24.0.0` (the version in
   requirements-lock.txt).
2. REPRODUCIBILITY.md said `scripts/reproduce_paper.sh` regenerates every PAPER_RESULTS.md table. It does not cover
   the toolkit tables (trust threshold, label-efficient and minimum checkpoints). Statement corrected; Python version,
   tests and quick-start added.
3. DATA_AVAILABILITY.md: HUST citation corrected to Ma, G. et al., Energy & Environmental Science 2022, vol. 15.

## Not fixed / for the main session
- `runtime.txt` says `python-3.11`; the project runs on 3.14 (Streamlit Cloud uses the app-settings Python, not this
  file). Left unchanged; flagged in REPRODUCIBILITY.md and the release checklist.
- DATA_AVAILABILITY.md does not list Tongji (130 batteries, part of the 16-source candidate and trust profiles).
- requirements.txt lists packages the app never imports (lime, dice_ml, gplearn, tabpfn, beep, ...); harmless but
  they lengthen the install. Not removed, to avoid changing the deployed environment.
- Not verified: a from-scratch install on Python 3.14 (torch is large); tests were run with the existing 3.14.2 venv
  against the clean clone.
- Pickled OC-SVM and scaler were saved with scikit-learn 1.8.0; they load under the pinned 1.9.0 with an
  InconsistentVersionWarning (works; re-saving would be cleaner).
- The clone has no `data/raw/`; the reproduce script needs it (documented).
