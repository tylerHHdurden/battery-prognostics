"""Precomputed RUL for NASA/MIT browse batteries (offline, from raw curves): file, lookup, and the rules for when it is shown."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import live_inference as li  # noqa: E402

PARQ = ROOT / "data" / "processed" / "precomputed_rul_nasa_mit.parquet"


def test_file_is_small_and_covers_every_nasa_mit_browse_cycle():
    assert PARQ.exists() and PARQ.stat().st_size < 1_000_000
    df = pd.read_parquet(PARQ)
    assert set(df.dataset) == {"NASA", "MIT"} and df.rul_pred.notna().all()
    for ds in ("NASA", "MIT"):
        expected = sum(len(v) for v in li.available_precomputed_cycles(ds).values())
        assert (df.dataset == ds).sum() == expected


def test_lookup_other_datasets_and_missing_cycles_return_none():
    assert li.lookup_precomputed_rul("Oxford", "Cell1", 100) is None
    assert li.lookup_precomputed_rul("NASA", "B0005", 10 ** 6) is None
    row = pd.read_parquet(PARQ).iloc[0]
    assert li.lookup_precomputed_rul(row.dataset, row.battery_id, int(row.cycle_idx)) == pytest.approx(float(row.rul_pred))


@pytest.fixture(scope="module")
def res():
    return li.load_resources()


def _ctx(res, ds, trust=None):
    avail = li.available_precomputed_cycles(ds)
    bid = sorted(avail)[0]
    return li.predict_and_explain_precomputed(ds, bid, int(avail[bid][-1]), res, trust=trust), bid


def test_rul_shown_for_unflagged_nasa_mit_only(res):
    for ds in ("NASA", "MIT"):
        ctx, bid = _ctx(res, ds)
        assert ctx["out_of_domain"] is False and ctx["rul_hidden"] is False
        assert isinstance(ctx["rul_pred"], int) and ctx["rul_conformal_lo"] <= ctx["rul_pred"] <= ctx["rul_conformal_hi"]
    for ds in ("CALCE", "Oxford", "HUST", "XJTU"):
        ctx, _ = _ctx(res, ds)
        assert ctx["rul_pred"] is None and ctx["rul_hidden"] is True


def test_rul_hidden_when_a_nasa_battery_is_flagged(res):
    avail = li.available_precomputed_cycles("NASA"); bid = sorted(avail)[0]
    flagged = {"nearest_source": "NASA", "nll_min": 10.0, "gate_table_mae": 1.0, "lodo_family_mae": 5.0}
    ctx = li.predict_and_explain_precomputed("NASA", bid, int(avail[bid][-1]), res, trust=flagged)
    assert ctx["out_of_domain"] is True and ctx["rul_pred"] is None and ctx["rul_hidden"] is True
