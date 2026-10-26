"""S1–S5 directional EV sanity checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_ev.ev import load_engine
from nfl_ev.sanity import hard_failures, run_sanity_suite

ART = ROOT / "artifacts"
REQUIRED = [
    "conversion_table.json",
    "fg_table.json",
    "punt_ep_table.json",
    "ep_by_yardline.json",
    "convert_logit.joblib",
    "fg_logit.joblib",
]


@pytest.fixture(scope="module")
def engine():
    missing = [f for f in REQUIRED if not (ART / f).exists()]
    if missing:
        pytest.skip(
            f"Missing artifacts {missing}. Run download → build_mart → train_models first."
        )
    return load_engine(ART)


def test_sanity_hard_cases(engine):
    results = run_sanity_suite(engine)
    fails = hard_failures(results)
    detail = "\n".join(
        f"{r['id']}: ok={r['ok']} soft={r['soft']} rec={r['recommendation']} "
        f"Δ={r['delta_ev']:.2f} msg={r['message']}"
        for r in results
    )
    assert not fails, f"Hard sanity failures:\n{detail}"


def test_s2_punt_not_winner(engine):
    results = {r["id"]: r for r in run_sanity_suite(engine)}
    assert results["S2"]["recommendation"] != "punt"


def test_s1_not_go_with_huge_delta(engine):
    results = {r["id"]: r for r in run_sanity_suite(engine)}
    r = results["S1"]
    assert not (r["recommendation"] == "go" and r["delta_ev"] > 1.5)


def test_fg_make_near_empirical_in_range(engine):
    """FG logistic must stay near empirical make rates in typical kick range (P0)."""
    from nfl_ev.ev import GameState

    # Opp ~25 → ~42-yard kick; thick empirical cells exist.
    r = engine.score(GameState(yardline_100=25, ydstogo=5, qtr=2, game_seconds_remaining=1800))
    assert r.sample_n_fg >= 30
    assert abs(r.p_make - r.p_make_baseline) < 0.20
    assert r.p_make > 0.55  # must not collapse toward coin-flip / balanced-weight failure
