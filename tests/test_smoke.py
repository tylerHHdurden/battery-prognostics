"""Quick smoke tests (whole file runs in well under 2 minutes).

Run from the repository root:  python -m pytest tests -q
They check that the shipped pieces import and run; they do not re-validate model accuracy
(see PAPER_RESULTS.md for that).
"""
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", [
    "live_inference", "battery_passport", "trust_report", "encoder_provenance", "data_adapters",
    "generate_health_report", "prescriptive_decision_layer", "run_second_life_grading",
    "run_degradation_mode_analysis", "ica_dv_dc",
])
def test_src_module_imports(name):
    assert importlib.import_module(name) is not None


def test_load_resources(res):
    assert isinstance(res, dict) and len(res) > 0
    assert "train_medians" in res


def test_one_prediction_on_oxford_cycle(oxford_ctx):
    ctx = oxford_ctx["ctx"]
    assert math.isfinite(ctx["soh_pred"])
    assert 0 < ctx["soh_pred"] < 130          # SOH is on a 0-100 scale
    assert ctx["soh_conformal_lo"] <= ctx["soh_pred"] <= ctx["soh_conformal_hi"]
    assert ctx["top_features"]                # SHAP explanation present
    assert ctx["rul_pred"] is None            # precomputed path never fabricates an RUL


def test_passport_json_and_pdf(li, oxford_ctx):
    import battery_passport as bp
    p = bp.build_passport(oxford_ctx["ctx"], "Oxford", oxford_ctx["battery_id"], oxford_ctx["n_cycles"])
    assert "NOT A CERTIFIED BATTERY PASSPORT" in p["document"]["disclaimer"]
    assert p["trust_status"]["guarantee"].startswith("none")
    assert p["trust_status"]["stated_detection_rate"] == li.OOD_NOVEL_DETECTED
    back = json.loads(bp.passport_json(p).decode("utf-8"))
    assert back["battery"]["battery_id"] == oxford_ctx["battery_id"]
    pdf = bp.passport_pdf(p)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 5000


def _trust(nearest, nll):
    return {"nearest_source": nearest, "nll_min": nll, "gate_table_mae": 1.0, "lodo_family_mae": 8.0}


def test_trust_flag_high_score_is_flagged(li):
    v = li.domain_verdict(_trust("NASA", 10.0))
    assert v["flagged"] and v["out_of_domain"] and v["rul_hidden"]


def test_trust_flag_low_score_not_flagged(li):
    v = li.domain_verdict(_trust("NASA", -40.0))
    assert not v["flagged"] and not v["out_of_domain"]
    assert not v["rul_hidden"]                # nearest source NASA, not flagged -> RUL may be shown


def test_missing_or_nan_trust_is_flagged(li):
    assert li.domain_verdict(None)["flagged"]
    assert li.domain_verdict(_trust("NASA", float("nan")))["flagged"]


@pytest.mark.parametrize("src", ["MIT", "NASA"])
def test_rul_shown_only_for_nasa_mit_nearest(li, src):
    assert li.domain_verdict(_trust(src, -40.0))["rul_hidden"] is False


@pytest.mark.parametrize("src", ["CALCE", "Oxford", "HUST", "XJTU", "snl"])
def test_rul_hidden_for_other_nearest_sources(li, src):
    v = li.domain_verdict(_trust(src, -40.0))
    assert not v["flagged"] and v["rul_hidden"] and v["rul_hidden_reason"]


def test_stated_trust_figures(li):
    assert li.OOD_NOVEL_DETECTED == 0.819 and li.OOD_KNOWN_FALSE_ALARM == 0.110
    assert set(li.OOD_WEAK_SOURCES) == {"mich", "NASA", "snl"}


def test_encoder_sidecars_match_committed_blobs():
    """Runs src/verify_sidecars_vs_git_blobs.py (needs git and a committed HEAD)."""
    try:
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    r = subprocess.run([sys.executable, str(ROOT / "src" / "verify_sidecars_vs_git_blobs.py")],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "invalid: []" in r.stdout
