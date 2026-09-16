"""
Plain-English battery health report generator: takes the structured
context from build_report_context.py (fusion-ensemble SOH prediction +
conformal interval, joint-adaptive-model RUL prediction + conformal
interval, per-instance top SHAP features, per-instance voltage-region
localization) and turns it into a short natural-language report via an
LLM API call.

API: Google Gemini (`gemini-2.5-flash`) via the Generative Language REST
API, called directly via `requests` rather than the `google-genai` SDK,
to avoid adding a dependency for one HTTP call (same reasoning as the
project's other API integrations). Key is read from `GEMINI_API_KEY`,
loaded automatically from a `.env` file in the project root via a small
hand-rolled parser (`python-dotenv` isn't installed and this project
avoids adding dependencies for something this simple - see e.g. the
Anthropic-via-`requests` precedent this replaces). `.env` is
git-ignored; never commit it.

Fallback chain: `call_llm` tries Gemini first; if that fails for any
reason (missing key, rate limit, any other error), it tries Groq
(`openai/gpt-oss-120b` via Groq's OpenAI-compatible chat-completions
endpoint, same prompt template, key from `GROQ_API_KEY`); only if BOTH
fail does the caller fall back to displaying the structured data
instead of a narrative. `call_llm` returns `(report_text, provider)`
where `provider` is `"Gemini"`, `"Groq"`, or `None` (both failed) - the
caller uses `provider` to label which model actually answered, and
still checks the `NO_API_KEY`/`API_ERROR` prefix on `report_text` to
decide whether to show the structured-data fallback.
"""

import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report_context import get_report_context

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
ENV_PATH = ROOT / ".env"


def _load_dotenv(path: Path = ENV_PATH) -> None:
    """Minimal .env loader: KEY=VALUE per line, '#' comments, no quoting
    support needed for this project's single-key use case. Does not
    override a variable already set in the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

PROMPT_TEMPLATE = """You are writing a short, plain-English battery health report for a non-expert reader (e.g. a fleet operator or equipment owner), based on structured predictions from a trained machine-learning pipeline. Do not invent any numbers - use only the figures given below. Keep the report to 2-3 sentences, in the style of a concise diagnostic summary.

Battery: {battery_id} (dataset: {dataset}), cycle {cycle_idx}

State of Health (SOH) prediction: {soh_pred}% (90% confidence interval: {soh_conformal_lo}% - {soh_conformal_hi}%)

Remaining Useful Life (RUL) prediction: approximately {rul_pred} cycles remaining (90% confidence interval: {rul_conformal_lo} - {rul_conformal_hi} cycles)

Top contributing factors to this SOH prediction, most important first:
{top_features_text}

Voltage region most associated with this cell's degradation signature: {v_lo}V - {v_hi}V (this region accounts for about {frac_pct}% of the model's attention in explaining this cell's discharge behavior)

Write the report now. Mention the SOH percentage, the voltage region likely driving degradation, and the RUL estimate with its confidence range - in a natural, flowing style similar to this example: "This battery is at 84% health, likely due to degradation concentrated in the 3.6-3.8V region; expect approximately 120 cycles remaining, with 90% confidence between 95-145 cycles."
"""


def build_prompt(context: dict) -> str:
    top_features_text = "\n".join(
        f"  - {f['feature']}: {f['description']}" for f in context["top_features"]
    )
    vr = context["voltage_region"]
    return PROMPT_TEMPLATE.format(
        battery_id=context["battery_id"], dataset=context["dataset"], cycle_idx=context["cycle_idx"],
        soh_pred=context["soh_pred"], soh_conformal_lo=context["soh_conformal_lo"],
        soh_conformal_hi=context["soh_conformal_hi"],
        rul_pred=context["rul_pred"], rul_conformal_lo=context["rul_conformal_lo"],
        rul_conformal_hi=context["rul_conformal_hi"],
        top_features_text=top_features_text,
        v_lo=vr["v_lo"] if vr else "unknown", v_hi=vr["v_hi"] if vr else "unknown",
        frac_pct=round(vr["frac_of_attribution"] * 100) if vr else "unknown",
    )


QA_PROMPT_TEMPLATE = """You are answering ONE specific question about ONE specific battery cycle, using ONLY the structured data given below. Never invent a number, feature name, or claim that isn't given here - if you don't have the information to answer precisely, say so plainly instead of guessing or using generic outside knowledge about batteries.

Battery: {battery_id} (dataset: {dataset}), cycle {cycle_idx}

State of Health (SOH) prediction: {soh_pred}% (90% confidence interval: {soh_conformal_lo}% - {soh_conformal_hi}%)
Remaining Useful Life (RUL) prediction: approximately {rul_pred} cycles remaining (90% confidence interval: {rul_conformal_lo} - {rul_conformal_hi} cycles)

Top contributing factors to this SOH prediction, most important first:
{top_features_text}

Voltage region most associated with this cell's degradation signature: {v_lo}V - {v_hi}V ({frac_pct}% of the model's attribution)

Degradation-mode signature (from peak-tracking analysis): {degradation_mode}

Out-of-domain status: {domain_status}

KNOWN, DOCUMENTED LIMITATION - you MUST disclose this plainly whenever it is relevant to the question asked (e.g. any question about how much to trust the numbers above, or about a battery/dataset different from this model's own NASA/MIT training data): this project's own testing found that for data meaningfully different from the NASA/MIT training distribution - the clearest documented case being the CALCE dataset - the 90% confidence interval's ACTUAL real-world coverage collapses dramatically, to as low as 6-7% in the worst case tested, despite looking exactly as narrow and confident as a reliable in-domain interval. If this battery is flagged out-of-domain above, or the question is about a CALCE-like or otherwise different battery/dataset, you must say plainly that you cannot give a confidently reliable answer here and explain why, rather than answering as if the interval above can be trusted the way it can in-domain.

Question: {question}

Answer in 2-4 sentences, plain language, grounded strictly in the data above. If the honest answer is genuine uncertainty, say so directly (e.g. "I can't give you a confident answer here - this project's own testing found...") rather than sounding falsely authoritative.
"""


def build_qa_prompt(context: dict, question: str) -> str:
    top_features_text = "\n".join(
        f"  - {f['feature']}: {f['description']}" for f in context["top_features"]
    )
    vr = context.get("voltage_region")
    domain_status = "OUT-OF-DOMAIN - " + "; ".join(context.get("domain_reasons", [])) \
        if context.get("out_of_domain") else "in-domain (looks consistent with NASA/MIT training data)"
    return QA_PROMPT_TEMPLATE.format(
        battery_id=context["battery_id"], dataset=context["dataset"], cycle_idx=context["cycle_idx"],
        soh_pred=context["soh_pred"], soh_conformal_lo=context["soh_conformal_lo"],
        soh_conformal_hi=context["soh_conformal_hi"],
        rul_pred=context["rul_pred"], rul_conformal_lo=context["rul_conformal_lo"],
        rul_conformal_hi=context["rul_conformal_hi"],
        top_features_text=top_features_text,
        v_lo=vr["v_lo"] if vr else "unknown", v_hi=vr["v_hi"] if vr else "unknown",
        frac_pct=round(vr["frac_of_attribution"] * 100) if vr else "unknown",
        degradation_mode=context.get("degradation_mode") or "not available for this battery",
        domain_status=domain_status,
        question=question,
    )


def _is_error(text: str) -> bool:
    return text.startswith("NO_API_KEY") or text.startswith("API_ERROR")


def call_gemini(prompt: str, model: str = "gemini-flash-latest") -> str:
    """
    ASSUMPTION/deviation, disclosed: requested model was "gemini-2.5-flash",
    but the provided GEMINI_API_KEY's account gets a 404
    ("This model models/gemini-2.5-flash is no longer available to new
    users") on that exact model, even though it's listed as available in
    /v1beta/models for this same key - an account/tier restriction, not a
    bug in this code (confirmed via a direct curl to the generateContent
    endpoint, independent of this script). "gemini-2.0-flash" hit a
    separate free-tier rate-limit (429) on first test. "gemini-flash-latest"
    is the model alias confirmed working with this key (HTTP 200) - it
    currently resolves to "gemini-3.6-flash" per the response's own
    modelVersion field, not literally 2.5. If the account's access changes
    (e.g. billing/tier upgrade), swap the default back to
    "gemini-2.5-flash" - the rest of this function is unaffected either way.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "NO_API_KEY: set GEMINI_API_KEY (in .env) to enable live report generation."
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"content-type": "application/json"},
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"API_ERROR: Gemini call failed ({type(e).__name__}: {e})"


def call_groq(prompt: str, model: str = "openai/gpt-oss-120b") -> str:
    """Groq's OpenAI-compatible chat-completions endpoint - same prompt
    template as Gemini, used only as a fallback when Gemini fails.

    MODEL FIX (found and fixed during the website-overhaul Phase 3 chat
    work, disclosed here not silently patched): the originally-used
    `llama-3.3-70b-versatile` has been REMOVED from Groq's model catalog
    since this project last used it - confirmed directly via Groq's own
    `/v1/models` endpoint (still a 200/valid API key, the specific model
    id is simply gone, hence every fallback call was failing with a 404).
    This is a real, pre-existing reliability gap this Phase 3 testing
    surfaced (the SAME failure would have silently affected the existing
    Health Report tab's own Groq fallback too, not just the new chat
    feature) - `openai/gpt-oss-120b` (confirmed present and working via
    a live test call) is the new default, the largest general-purpose
    chat model in the current catalog."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "NO_API_KEY: set GROQ_API_KEY (in .env) to enable the Groq fallback."
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"API_ERROR: Groq call failed ({type(e).__name__}: {e})"


def call_llm(prompt: str, gemini_model: str = "gemini-flash-latest",
             groq_model: str = "openai/gpt-oss-120b") -> tuple[str, str | None]:
    """Try Gemini first; on any failure (missing key, rate limit, or any
    other error) fall through to Groq; only if both fail does the caller
    see an error sentinel. Returns (report_text, provider) where
    provider is "Gemini", "Groq", or None if both failed."""
    gemini_result = call_gemini(prompt, gemini_model)
    if not _is_error(gemini_result):
        return gemini_result, "Gemini"

    groq_result = call_groq(prompt, groq_model)
    if not _is_error(groq_result):
        return groq_result, "Groq"

    combined = ("NO_API_KEY" if gemini_result.startswith("NO_API_KEY")
                and groq_result.startswith("NO_API_KEY") else "API_ERROR")
    return f"{combined}: both providers failed - Gemini: {gemini_result}; Groq: {groq_result}", None


def generate_report(dataset: str, battery_id: str, cycle_idx: int) -> dict:
    context = get_report_context(dataset, battery_id, cycle_idx)
    prompt = build_prompt(context)
    report_text, provider = call_llm(prompt)
    return {"context": context, "prompt": prompt, "report": report_text, "provider": provider}


if __name__ == "__main__":
    EXAMPLES = [
        ("MIT", "b1c4", 67),
        ("MIT", "b4c38", 250),
        ("MIT", "b1c4", 674),
        ("MIT", "b3c0", 747),
        ("MIT", "b4c38", 1096),
    ]
    results = []
    for dataset, battery_id, cycle_idx in EXAMPLES:
        print(f"\n=== {dataset}/{battery_id} cycle {cycle_idx} ===")
        r = generate_report(dataset, battery_id, cycle_idx)
        print("PROMPT:\n", r["prompt"])
        print(f"REPORT (via {r['provider']}):\n", r["report"])
        results.append(r)

    with open(OUT_DIR / "health_reports_examples.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} examples to outputs/health_reports_examples.json")
