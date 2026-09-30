# Tests

Quick smoke tests for the CellSense pieces that ship with the app. They check that things import and run
(models load, one prediction, passport JSON/PDF export, the trust flag rule, the RUL-hiding rule, encoder
provenance sidecars). They do not re-validate model accuracy; the results are in `PAPER_RESULTS.md`.

```bash
pip install pytest
python -m pytest tests -q
```

Needs the repository's tracked model and data files (no raw datasets). The sidecar test needs `git` and a
committed `HEAD`. Expected run time: well under two minutes (loading the models takes about 10 s).
