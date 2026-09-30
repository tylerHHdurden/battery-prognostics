"""Shared fixtures. Models are loaded once per test session (about 10 s)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))


@pytest.fixture(scope="session")
def li():
    import live_inference
    return live_inference


@pytest.fixture(scope="session")
def res(li):
    return li.load_resources()


@pytest.fixture(scope="session")
def oxford_ctx(li, res):
    cycles = li.available_precomputed_cycles("Oxford")
    bid = sorted(cycles)[0]
    cyc = cycles[bid][len(cycles[bid]) // 2]
    ctx = li.predict_and_explain_precomputed("Oxford", bid, cyc, res)
    return {"ctx": ctx, "battery_id": bid, "n_cycles": len(cycles[bid])}
