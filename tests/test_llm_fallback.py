"""Health Report LLM chain: Gemini (Flash-Lite) -> Groq -> structured-data display. Provider calls are mocked (no network, no real keys needed); the
tests also check that no API key can leak through an error message, which `requests` otherwise includes in HTTPError text (the Gemini URL carries ?key=...)."""
import os
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import generate_health_report as g  # noqa: E402

FAKE_GEMINI = "FAKEGEMINIKEY0123456789abcdefghij"
FAKE_GROQ = "FAKEGROQKEY0123456789abcdefghijklmnopqrstuvwxyz0123"


class Resp:
    def __init__(self, status=200, payload=None, url=""):
        self.status_code, self._payload, self.url = status, payload or {}, url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error: error for url: {self.url}")

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ)


def post_factory(gemini_ok=True, groq_ok=True, seen=None):
    def post(url, headers=None, params=None, json=None, timeout=None):
        if seen is not None:
            seen.append(url)
        if "generativelanguage" in url:
            full = url + f"?key={(params or {}).get('key')}"
            return Resp(200, {"candidates": [{"content": {"parts": [{"text": "gemini text"}]}}]}) if gemini_ok else Resp(429, url=full)
        return Resp(200, {"choices": [{"message": {"content": "groq text"}}]}) if groq_ok else Resp(500, url=url)
    return post


def test_primary_model_is_flash_lite_and_groq_model_unchanged():
    assert "flash-lite" in g.GEMINI_MODEL and g.GEMINI_MODEL.startswith("gemini-")
    import inspect
    assert inspect.signature(g.call_llm).parameters["gemini_model"].default == g.GEMINI_MODEL
    assert inspect.signature(g.call_llm).parameters["groq_model"].default == "openai/gpt-oss-120b"


def test_gemini_used_first(monkeypatch):
    seen = []
    monkeypatch.setattr(g.requests, "post", post_factory(seen=seen))
    text, provider = g.call_llm("p")
    assert (text, provider) == ("gemini text", "Gemini") and any(g.GEMINI_MODEL in u for u in seen)


def test_falls_back_to_groq_when_gemini_fails(monkeypatch):
    monkeypatch.setattr(g.requests, "post", post_factory(gemini_ok=False))
    text, provider = g.call_llm("p")
    assert (text, provider) == ("groq text", "Groq")


def test_both_fail_gives_error_sentinel_without_keys(monkeypatch):
    monkeypatch.setattr(g.requests, "post", post_factory(gemini_ok=False, groq_ok=False))
    text, provider = g.call_llm("p")
    assert provider is None and text.startswith("API_ERROR")        # the app shows the structured-data display on this sentinel
    assert FAKE_GEMINI not in text and FAKE_GROQ not in text and "key=AIza" not in text


def test_no_keys_gives_no_api_key_sentinel(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY"), monkeypatch.delenv("GROQ_API_KEY")
    text, provider = g.call_llm("p")
    assert provider is None and text.startswith("NO_API_KEY")


def test_scrub_removes_key_from_urls_and_values():
    msg = f"429 Client Error: for url: https://x/y:generateContent?key={FAKE_GEMINI}&alt=sse and again {FAKE_GROQ}"
    out = g._scrub(msg)
    assert FAKE_GEMINI not in out and FAKE_GROQ not in out and "key=<redacted>" in out


def test_keys_come_from_dotenv_or_streamlit_secrets():
    src = Path(g.__file__).read_text(encoding="utf-8")
    assert "_load_dotenv()" in src and "_load_streamlit_secrets()" in src and "st.secrets" in src


def _ctx_precomputed_oxford():
    import live_inference as li
    res = li.load_resources()
    avail = li.available_precomputed_cycles("Oxford")
    bid = sorted(avail)[0]
    ctx = dict(li.predict_and_explain_precomputed("Oxford", bid, int(avail[bid][-1]), res))
    ctx["battery_id"], ctx["dataset"] = bid, "Oxford"
    return ctx


def test_prompts_never_contain_unknownV_on_the_precomputed_path():
    ctx = _ctx_precomputed_oxford()
    assert ctx["voltage_region"] is None                      # the precomputed path has no raw curve, so no voltage region
    for prompt in (g.build_prompt(ctx), g.build_qa_prompt(ctx, "How confident should I be?")):
        assert "unknownV" not in prompt and "unknown%" not in prompt
        assert "not available in this deployment" in prompt   # said plainly instead of inventing a region


def test_prompts_keep_the_voltage_region_sentence_when_there_is_one():
    ctx = _ctx_precomputed_oxford()
    ctx["voltage_region"] = {"v_lo": 3.6, "v_hi": 3.8, "frac_of_attribution": 0.42}
    p1, p2 = g.build_prompt(ctx), g.build_qa_prompt(ctx, "q")
    assert "3.6V - 3.8V" in p1 and "42%" in p1 and "3.6V - 3.8V" in p2 and "42%" in p2
    assert "unknownV" not in p1 + p2 and "not available in this deployment (it needs" not in p1
