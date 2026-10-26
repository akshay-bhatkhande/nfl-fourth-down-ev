"""EV(go / FG / punt), recommendation, and caveats."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nfl_ev.baseline import (
    load_json,
    lookup_conversion_p,
    lookup_ep,
    lookup_fg_p,
    lookup_punt_ep,
)
from nfl_ev.features import MIN_CELL_N, kick_distance_proxy
from nfl_ev.model import load_model, predict_proba_row


@dataclass
class GameState:
    yardline_100: float = 65.0  # own 35 → 65 yards to opp EZ
    ydstogo: float = 2.0
    score_differential: float = 0.0
    qtr: int = 2
    game_seconds_remaining: float = 1800.0
    posteam_timeouts_remaining: int = 3
    defteam_timeouts_remaining: int = 3
    roof: str = "outdoors"
    goal_to_go: int = 0

    def as_dict(self) -> dict[str, Any]:
        gtg = 1 if self.yardline_100 <= self.ydstogo else int(self.goal_to_go)
        return {
            "yardline_100": float(self.yardline_100),
            "ydstogo": float(self.ydstogo),
            "goal_to_go": int(gtg),
            "score_differential": float(self.score_differential),
            "game_seconds_remaining": float(self.game_seconds_remaining),
            "qtr": int(self.qtr),
            "posteam_timeouts_remaining": int(self.posteam_timeouts_remaining),
            "defteam_timeouts_remaining": int(self.defteam_timeouts_remaining),
            "roof": self.roof,
        }


@dataclass
class EvResult:
    ev_go: float
    ev_fg: float
    ev_punt: float
    p_conv: float
    p_make: float
    p_conv_baseline: float
    p_make_baseline: float
    recommendation: str
    delta_ev: float
    second_best: str
    caveats: list[str] = field(default_factory=list)
    reason: str = ""
    low_confidence: bool = False
    sample_n_conv: int = 0
    sample_n_fg: int = 0


class EvEngine:
    """Load artifacts and score a game state."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.conv_table = load_json(self.artifacts_dir / "conversion_table.json")
        self.fg_table = load_json(self.artifacts_dir / "fg_table.json")
        self.punt_table = load_json(self.artifacts_dir / "punt_ep_table.json")
        self.ep_by_yl = load_json(self.artifacts_dir / "ep_by_yardline.json")
        self.metrics = load_json(self.artifacts_dir / "metrics.json")
        self.convert_pipe, self.convert_cols = load_model(
            self.artifacts_dir / "convert_logit.joblib"
        )
        self.fg_pipe, self.fg_cols = load_model(self.artifacts_dir / "fg_logit.joblib")

    def score(self, state: GameState, use_model: bool = True) -> EvResult:
        d = state.as_dict()
        yl = d["yardline_100"]
        ytg = d["ydstogo"]

        p_conv_b, n_conv, back_conv = lookup_conversion_p(self.conv_table, yl, ytg)
        p_make_b, n_fg, back_fg = lookup_fg_p(self.fg_table, yl)
        punt_ep, n_punt, back_punt = lookup_punt_ep(self.punt_table, yl)

        kick_dist = float(kick_distance_proxy(yl))
        if use_model:
            p_conv = predict_proba_row(self.convert_pipe, self.convert_cols, d)
            p_make = predict_proba_row(self.fg_pipe, self.fg_cols, d)
            # Blend toward baseline when cell is thin
            if back_conv or n_conv < MIN_CELL_N:
                p_conv = 0.5 * p_conv + 0.5 * p_conv_b
            if back_fg or n_fg < MIN_CELL_N:
                p_make = 0.5 * p_make + 0.5 * p_make_b
            # Guardrail: never let FG logistic drift far below a thick empirical
            # distance cell (historical failure mode when class_weight=balanced).
            if n_fg >= MIN_CELL_N and not back_fg and p_make + 0.15 < p_make_b:
                p_make = 0.5 * p_make + 0.5 * p_make_b
            # Model was not trained on absurd distances — hard-cap long FG probs.
            if kick_dist >= 70:
                p_make = min(p_make, 0.01)
            elif kick_dist >= 65:
                p_make = min(p_make, 0.08)
            elif kick_dist >= 60:
                p_make = min(p_make, max(p_make_b, 0.15))
        else:
            p_conv, p_make = p_conv_b, p_make_b

        ep_now = lookup_ep(self.ep_by_yl, yl)
        # After convert: roughly at yardline_100 - ydstogo (new 1st down)
        yl_after_conv = max(1.0, yl - ytg)
        ep_after_conv = lookup_ep(self.ep_by_yl, yl_after_conv)
        # Turnover on downs: opponent at 100 - yardline_100
        opp_yl = max(1.0, min(99.0, 100.0 - yl))
        ep_tod = -lookup_ep(self.ep_by_yl, opp_yl)

        ev_go = p_conv * ep_after_conv + (1.0 - p_conv) * ep_tod

        # FG convention (frozen): make = +3; miss = −EP_opp at miss spot
        # Miss spot heuristic: ≈ 100 - (yardline_100 + 8)
        miss_opp_yl = max(1.0, min(99.0, 100.0 - (yl + 8)))
        ep_miss = -lookup_ep(self.ep_by_yl, miss_opp_yl)
        ev_fg = p_make * 3.0 + (1.0 - p_make) * ep_miss

        ev_punt = punt_ep

        scores = {"go": ev_go, "fg": ev_fg, "punt": ev_punt}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_ev = ranked[0]
        second, second_ev = ranked[1]
        delta = best_ev - second_ev

        caveats: list[str] = []
        low = False
        if n_conv < MIN_CELL_N or back_conv:
            caveats.append(
                f"Low conversion sample for this cell (N={n_conv} < {MIN_CELL_N}) — confidence reduced."
            )
            low = True
        if n_fg < MIN_CELL_N or back_fg:
            caveats.append(
                f"Thin FG sample for this kick distance (N={n_fg}) — blended with baseline."
            )
            low = True
        if d["game_seconds_remaining"] < 120 or (d["qtr"] >= 4 and abs(d["score_differential"]) >= 3):
            caveats.append(
                "Late game / score pressure: expected points ≠ win value (WP out of MVP scope)."
            )
        if kick_distance_proxy(yl) > 55:
            caveats.append("Long FG distance uses a heuristic make curve — treat as proxy.")
        caveats.append("League-average model — not team- or QB-specific.")
        caveats.append(
            "FG/punt field position uses documented heuristics (see docs/decisions.md)."
        )

        reasons = {
            "go": f"Going for it maximizes proxy EP ({best_ev:.2f}) vs {second} ({second_ev:.2f}).",
            "fg": f"Field goal maximizes proxy EP ({best_ev:.2f}) vs {second} ({second_ev:.2f}).",
            "punt": f"Punting maximizes proxy EP ({best_ev:.2f}) vs {second} ({second_ev:.2f}).",
        }

        return EvResult(
            ev_go=float(ev_go),
            ev_fg=float(ev_fg),
            ev_punt=float(ev_punt),
            p_conv=float(p_conv),
            p_make=float(p_make),
            p_conv_baseline=float(p_conv_b),
            p_make_baseline=float(p_make_b),
            recommendation=best,
            delta_ev=float(delta),
            second_best=second,
            caveats=caveats,
            reason=reasons[best],
            low_confidence=low,
            sample_n_conv=int(n_conv),
            sample_n_fg=int(n_fg),
        )


def load_engine(artifacts_dir: str | Path | None = None) -> EvEngine:
    root = Path(__file__).resolve().parents[2]
    path = Path(artifacts_dir) if artifacts_dir else root / "artifacts"
    return EvEngine(path)
