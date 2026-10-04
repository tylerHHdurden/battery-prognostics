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


# ---- "Why isn't RUL shown here?" expander ----
def _flat(content):
    return " ".join(content["paragraphs"])


def test_expander_absent_when_rul_shown(res):
    assert rm.why_not_shown_content(_ctx(res, "NASA"), "NASA") is None


def test_expander_for_each_evaluated_external_source_highlights_it(res):
    for ds in ("CALCE", "Oxford", "HUST", "XJTU"):
        c = rm.why_not_shown_content(_ctx(res, ds), ds)
        assert c["highlight"] == ds and rm.WHY_TRAINED in c["paragraphs"] and rm.WHY_INSTEAD in c["paragraphs"]
        assert rm.WHY_NOT_EVALUATED not in c["paragraphs"]
        assert "-401 cycles" in _flat(c) and "no raw curves" in _flat(c) and "b1c4" in _flat(c)


def test_expander_for_unevaluated_sources_and_uploads(res):
    c = rm.why_not_shown_content(_ctx(res, "tongji"), "tongji")
    assert c["highlight"] is None and rm.WHY_NOT_EVALUATED in c["paragraphs"]
    up = rm.why_not_shown_content({"is_upload": True, "rul_hidden": True, "out_of_domain": False}, "Uploaded")
    assert up["highlight"] is None and rm.WHY_NOT_EVALUATED in up["paragraphs"]


def test_expander_for_flagged_nasa_battery(res):
    avail = li.available_precomputed_cycles("NASA"); bid = sorted(avail)[0]
    flagged = {"nearest_source": "NASA", "nll_min": 10.0, "gate_table_mae": 1.0, "lodo_family_mae": 5.0}
    ctx = li.predict_and_explain_precomputed("NASA", bid, int(avail[bid][-1]), res, trust=flagged)
    c = rm.why_not_shown_content(ctx, "NASA")
    assert rm.WHY_FLAGGED in c["paragraphs"] and rm.WHY_NOT_EVALUATED not in c["paragraphs"] and c["highlight"] is None


def test_crossdomain_table_values_come_from_the_csvs():
    import pandas as pd
    df = rm.rul_crossdomain_table(ROOT / "outputs")
    x = pd.read_csv(ROOT / "outputs" / "finalpass_item5c_rul_crossdomain.csv").set_index("dataset")
    assert list(df["Dataset"])[1:] == list(rm.EVALUATED_EXTERNAL)
    assert df.loc[df.Dataset == "CALCE", "MAE (cycles)"].iloc[0] == pytest.approx(x.loc["CALCE", "rul_mae"])
    assert df.loc[df.Dataset == "XJTU", "R2"].iloc[0] == pytest.approx(x.loc["XJTU", "rul_r2"])
    assert df["R2"].iloc[0] == pytest.approx(0.374, abs=1e-3)


def test_expander_wired_into_result_card():
    assert "render_rul_why_expander(ctx, dataset)" in APP and "WHY_TITLE" in APP
