# MORNING RUNBOOK (written overnight, 2026-10-01)

Nothing was pushed to `master` overnight and bat-pro.streamlit.app was not touched. The work is on branch **site-redesign** (cut from master 1d3bb23) and, if you ask, pushed as a branch only.
Read `outputs/overnight_status.md` first (what was done, every commit, every decision, what is unfinished).

## 1. Review (10 minutes)
- Screenshots: `outputs/site_redesign/` (desktop_* and mobile_* top-of-page and tab screenshots, `site_visual_result.json`, `bench_*.json` for cold/warm render times).
- Regression: `outputs/toolkit_ood_matrix_redesign2.json` vs `outputs/toolkit_ood_matrix_routed.json` (same SOH/RUL strings, 0 exceptions).
- Docs: README.md, release/ (drafts only), tests/ (`.venv\Scripts\python.exe -m pytest tests -q`, about 10 s), PAPER_RESULTS.md (every table has a status + source), `report/Addendum_Post_Submission.pdf` (2 pages), `report/walkthrough/Project_Walkthrough.pdf` (10 pages). Note: `report/` is git-ignored on purpose (names and register numbers), so those PDFs exist only on this machine.

## 2. Try the redesign on STAGING first
1. Streamlit Cloud dashboard -> the staging app (bat-pro-stage) -> Settings -> General -> **Branch: `site-redesign`** (or create a third app: repo `tylerHHdurden/battery-prognostics`, branch `site-redesign`, main file `app.py`, Python 3.14). The branch must be pushed first: `git push origin site-redesign` (I did not push it unless the status file says so).
2. After a branch change or a push, staging may show a transient `ImportError` (hot update). Fix: Manage app -> three dots -> **Reboot app**, wait 1-3 minutes.
3. Test (from the repo root, Windows, `.venv`): `PYTHONIOENCODING=utf8 .venv\Scripts\python.exe src/live_app_smoke_test.py https://bat-pro-stage.streamlit.app staging_site new` (about 12 minutes; six datasets, flagged and unflagged uploads, both passport downloads, no exceptions, no green/"trusted" wording). Screenshots and JSON land in `outputs/live_smoke/staging_site_*`. Also run `src/site_visual_check.py <url> outputs/site_redesign_staging` for desktop and mobile views.

## 3. Merge to master and go live (only after staging passes)
```
git fetch origin
git checkout master
git tag pre-site-redesign-rollback master          # rollback point
git push origin pre-site-redesign-rollback         # (a new tag on origin - done by you, not overnight)
git merge --ff-only site-redesign                  # master is an ancestor of site-redesign, so this is a fast-forward
git push origin master
```
Then: dashboard -> bat-pro -> Manage app -> three dots -> **Reboot app** -> paste the log text here with the word "rebooted". Do not test before the reboot (hot-update ImportError). After "rebooted" I run `src/live_app_smoke_test.py https://bat-pro.streamlit.app live_site new`.

## 4. Rollback (if the live smoke test fails after the reboot)
Preferred (keeps history, no force-push), from a clean checkout of master:
```
tree=$(git rev-parse pre-site-redesign-rollback^{tree})
c=$(git commit-tree "$tree" -p HEAD -m "REVERT: restore pre-site-redesign-rollback (live smoke test failed)")
git update-ref refs/heads/master "$c"
git push origin master
```
then Reboot app again and confirm with `src/live_app_smoke_test.py https://bat-pro.streamlit.app live_after_revert old`. The older rollback tags `pre-step3-rollback` (f49493e) and `pre-step3b-rollback` (5e92ae6) also exist.

## 5. Things to know
- The redesign changes the layout, not the numbers: SOH/RUL strings for the six datasets plus the HNEI upload are identical to before (see the regression files).
- Tabs now load lazily (only the open tab runs), so the smoke test clicks the Prediction tab before looking for the passport.
- `runtime.txt` says python-3.11 but the apps run 3.14 (the dashboard setting wins); I left the file alone.
- `requirements.txt` gained `pyarrow==24.0.0` (it was missing; it was already installed transitively). A push that changes requirements triggers a dependency reinstall, which is normal.
- Nothing public was created: no GitHub release, DOI, PyPI package or new tag on origin. `release/` holds drafts and a checklist of what a human must do to publish.
