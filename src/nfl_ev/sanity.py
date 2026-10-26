"""Extreme-situation presets and directional checks (S1–S5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from nfl_ev.ev import EvEngine, EvResult, GameState


@dataclass
class SanityCase:
    id: str
    label: str
    state: GameState
    check: Callable[[EvResult], tuple[bool, str]]
    soft: bool = False  # soft-warn only for borderline S3/S4


def _s1(r: EvResult) -> tuple[bool, str]:
    # Prefer punt; fail if go is clear #1 with huge ΔEV
    if r.recommendation == "go" and r.delta_ev > 1.5:
        return False, f"S1 fail: go recommended with huge ΔEV={r.delta_ev:.2f}"
    if r.ev_punt >= r.ev_go - 0.25 and r.ev_punt >= r.ev_fg - 0.5:
        return True, "punt competitive / preferred in own territory long yardage"
    if r.recommendation == "punt":
        return True, "punt recommended"
    # Soft pass if punt is close second
    if r.second_best == "punt" or r.ev_punt > max(r.ev_go, r.ev_fg) - 1.0:
        return True, "punt EV near top (directional ok)"
    return False, f"S1 unexpected: rec={r.recommendation} ΔEV={r.delta_ev:.2f}"


def _s2(r: EvResult) -> tuple[bool, str]:
    if r.recommendation == "punt":
        return False, "S2 fail: punt must not win at opponent 1, 4th & goal"
    if r.recommendation in {"go", "fg"}:
        return True, f"{r.recommendation} preferred at goal line"
    return False, f"S2 unexpected: {r.recommendation}"


def _s3(r: EvResult) -> tuple[bool, str]:
    # Go competitive; FG possible — document which wins
    if r.recommendation == "punt" and r.delta_ev > 1.0:
        return False, "S3 soft-fail: punt clearly ahead at opp 15 & 1"
    return True, f"S3: {r.recommendation} wins (ΔEV={r.delta_ev:.2f})"


def _s4(r: EvResult) -> tuple[bool, str]:
    # FG or punt ahead of go
    if r.recommendation == "go" and r.delta_ev > 1.0:
        return False, "S4 soft-fail: go clearly ahead on 4th & 12 from opp 30"
    if r.ev_go > max(r.ev_fg, r.ev_punt) + 1.0:
        return False, "S4: go EV unexpectedly dominant"
    return True, f"S4: {r.recommendation} ahead of go (directional)"


def _s5(r: EvResult) -> tuple[bool, str]:
    # Go favored vs punt
    if r.recommendation == "punt" and r.delta_ev > 0.5:
        return False, "S5 fail: punt favored while trailing late on 4th & 1"
    if r.ev_go >= r.ev_punt - 0.1:
        return True, "go favored or tied vs punt (clock/score caveat expected)"
    return False, f"S5: go EV {r.ev_go:.2f} << punt {r.ev_punt:.2f}"


SANITY_CASES: list[SanityCase] = [
    SanityCase(
        id="S1",
        label="Own 5, 4th & 10, Q2, tied → prefer punt",
        state=GameState(
            yardline_100=95,
            ydstogo=10,
            score_differential=0,
            qtr=2,
            game_seconds_remaining=1800,
        ),
        check=_s1,
    ),
    SanityCase(
        id="S2",
        label="Opp 1, 4th & goal, Q2, tied → prefer go (punt must not win)",
        state=GameState(
            yardline_100=1,
            ydstogo=1,
            score_differential=0,
            qtr=2,
            game_seconds_remaining=1800,
            goal_to_go=1,
        ),
        check=_s2,
    ),
    SanityCase(
        id="S3",
        label="Opp 15, 4th & 1, Q2, tied → go competitive / FG possible",
        state=GameState(
            yardline_100=15,
            ydstogo=1,
            score_differential=0,
            qtr=2,
            game_seconds_remaining=1800,
        ),
        check=_s3,
        soft=True,
    ),
    SanityCase(
        id="S4",
        label="Opp 30, 4th & 12, Q2, tied → FG or punt ahead of go",
        state=GameState(
            yardline_100=30,
            ydstogo=12,
            score_differential=0,
            qtr=2,
            game_seconds_remaining=1800,
        ),
        check=_s4,
        soft=True,
    ),
    SanityCase(
        id="S5",
        label="Own 40, 4th & 1, Q4, trailing by 4, <1:00 → go vs punt",
        state=GameState(
            yardline_100=60,
            ydstogo=1,
            score_differential=-4,
            qtr=4,
            game_seconds_remaining=50,
            posteam_timeouts_remaining=2,
            defteam_timeouts_remaining=2,
        ),
        check=_s5,
    ),
]


def run_sanity_suite(engine: EvEngine) -> list[dict]:
    results = []
    for case in SANITY_CASES:
        ev = engine.score(case.state)
        ok, msg = case.check(ev)
        results.append(
            {
                "id": case.id,
                "label": case.label,
                "ok": ok,
                "soft": case.soft,
                "message": msg,
                "recommendation": ev.recommendation,
                "delta_ev": ev.delta_ev,
                "ev_go": ev.ev_go,
                "ev_fg": ev.ev_fg,
                "ev_punt": ev.ev_punt,
            }
        )
    return results


def hard_failures(results: list[dict]) -> list[dict]:
    """Hard CI failures: non-soft cases that failed."""
    return [r for r in results if (not r["ok"]) and (not r["soft"])]
