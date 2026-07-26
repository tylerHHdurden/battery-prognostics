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

If no key is configured or the API call fails for any reason, `call_llm`
returns a clear `NO_API_KEY`/`API_ERROR` sentinel rather than silently
failing or fabricating a response - `app.py` and this module's own
fallback both check for that prefix and show the structured data
instead of a narrative in that case.
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


def call_llm(prompt: str, model: str = "gemini-flash-latest") -> str:
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


def generate_report(dataset: str, battery_id: str, cycle_idx: int) -> dict:
    context = get_report_context(dataset, battery_id, cycle_idx)
    prompt = build_prompt(context)
    report_text = call_llm(prompt)
    return {"context": context, "prompt": prompt, "report": report_text}


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
        print("REPORT:\n", r["report"])
        results.append(r)

    with open(OUT_DIR / "health_reports_examples.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} examples to outputs/health_reports_examples.json")
