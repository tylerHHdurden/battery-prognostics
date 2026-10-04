"""RUL wording (shown / not shown / flagged / upload) and the corrected Streaming Digital Twin and Health Report text."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import live_inference as li  # noqa: E402
import rul_messages as rm  # noqa: E402

APP = (ROOT / "app.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def res():
    return li.load_resources()


def _ctx(res, ds, trust=None):
    avail = li.available_precomputed_cycles(ds)
    bid = sorted(avail)[0]
    return li.predict_and_explain_precomputed(ds, bid, int(avail[bid][-1]), res, trust=trust)


def test_shown_caption_for_unflagged_nasa_mit(res):
    for ds in ("NASA", "MIT"):
        ctx = _ctx(res, ds)
        assert rm.rul_state(ctx) == "shown"
        cap = rm.rul_shown_caption(ctx)
        assert "offline" in cap and "max difference 0 cycles over 28 pairs" in cap and "0.37" in cap


def test_not_shown_message_for_other_sources(res):
    for ds in ("CALCE", "Oxford", "HUST", "XJTU"):
        ctx = _ctx(res, ds)
        assert rm.rul_state(ctx) != "shown"
        msg = rm.rul_not_shown_message(rm.rul_state(ctx))
        assert "421" in msg and "-566" in msg and "SOH is the headline output" in msg


def test_flagged_nasa_battery_not_shown(res):
    avail = li.available_precomputed_cycles("NASA"); bid = sorted(avail)[0]
    flagged = {"nearest_source": "NASA", "nll_min": 10.0, "gate_table_mae": 1.0, "lodo_family_mae": 5.0}
    ctx = li.predict_and_explain_precomputed("NASA", bid, int(avail[bid][-1]), res, trust=flagged)
    assert rm.rul_state(ctx) == "flagged"
    assert "flagged as unfamiliar" in rm.rul_not_shown_message("flagged")


def test_upload_not_shown_even_if_context_says_visible():
    ctx = {"is_upload": True, "rul_hidden": False, "rul_pred": 100, "out_of_domain": False}
    assert rm.rul_state(ctx) == "upload"
    assert "uploaded data" in rm.rul_not_shown_message("upload")


def test_no_green_or_trusted_wording_in_rul_messages():
    for text in (rm.RUL_SHOWN_CAPTION, rm.RUL_LIVE_CAPTION, rm.RUL_NOT_SHOWN_TEXT):
        assert "trusted" not in text.lower() and ":green" not in text


def test_twin_text_states_what_data_is_used():
    assert "exact " "same raw data" not in APP
    assert "hi_table.parquet" in APP and "fusion_embeddings.csv" in APP
    assert "This twin does not use the RUL model" in APP
    assert "~51%" not in APP


def test_upload_forced_hidden_in_app():
    assert 'ctx.update({"is_upload": True, "rul_hidden": True' in APP
