"""Pre-play feature allowlist and banned leakage columns."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

# Columns usable at decision time (pre-snap / pre-play).
ALLOWED_FEATURES: list[str] = [
    "yardline_100",
    "ydstogo",
    "goal_to_go",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "roof",  # optional; one-hot later if present
]

# Banned as model inputs (leakage / post-decision / circular).
BANNED_FEATURES: list[str] = [
    "epa",
    "ep",  # banned as model *feature*; used only for EP scale tables
    "wpa",
    "wp",
    "vegas_wp",
    "success",
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "complete_pass",
    "interception",
    "fumble",
    "fumble_lost",
    "fumble_forced",
    "fumble_not_forced",
    "fumble_out_of_bounds",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_id",
    "desc",
    "pass_length",
    "run_location",
    "first_down",
    "touchdown",
    "field_goal_result",
    "kick_distance",
    "return_yards",
    "penalty",
    "cp",
    "cpoe",
    "fg_prob",
    "td_prob",
    "safety_prob",
]

# Labels / filters (not model features).
LABEL_COLUMNS: list[str] = [
    "play_type",
    "first_down",
    "touchdown",
    "field_goal_result",
    "season",
    "game_id",
    "play_id",
]

PLAY_TYPES_GO = {"pass", "run"}
PLAY_TYPES_FG = {"field_goal"}
PLAY_TYPES_PUNT = {"punt"}
PLAY_TYPES_KEEP = PLAY_TYPES_GO | PLAY_TYPES_FG | PLAY_TYPES_PUNT

MIN_CELL_N = 30


def assert_no_banned(columns: Iterable[str]) -> None:
    """Raise if any banned column appears in a feature matrix column set."""
    banned_hit = sorted(set(columns) & set(BANNED_FEATURES))
    if banned_hit:
        raise ValueError(f"Banned leakage columns in feature matrix: {banned_hit}")


def feature_matrix(
    df: pd.DataFrame,
    *,
    include_roof: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Build a numeric feature matrix from pre-play columns only.

    Roof is one-hot encoded when present and include_roof is True.
    Season is intentionally excluded from the coach-facing feature matrix
    (used only for train/holdout splits).
    """
    cols = [c for c in ALLOWED_FEATURES if c != "roof" and c in df.columns]
    out = df[cols].copy()

    if include_roof and "roof" in df.columns:
        roof = df["roof"].fillna("unknown").astype(str).str.lower()
        dummies = pd.get_dummies(roof, prefix="roof")
        out = pd.concat([out, dummies], axis=1)

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    assert_no_banned(out.columns)
    if dropna:
        out = out.dropna()
    return out


def go_convert_label(df: pd.DataFrame) -> pd.Series:
    """Convert if first_down == 1 or touchdown == 1 on go attempts."""
    fd = pd.to_numeric(df["first_down"], errors="coerce").fillna(0).astype(int)
    td = pd.to_numeric(df["touchdown"], errors="coerce").fillna(0).astype(int)
    return ((fd == 1) | (td == 1)).astype(int)


def fg_make_label(df: pd.DataFrame) -> pd.Series:
    """FG make if field_goal_result == 'made'."""
    return (df["field_goal_result"].astype(str).str.lower() == "made").astype(int)


def kick_distance_proxy(yardline_100: pd.Series | float) -> pd.Series | float:
    """Decision-time kick distance heuristic: LOS to uprights ≈ yardline_100 + 17."""
    return yardline_100 + 17


def yardline_bucket(yardline_100: pd.Series, edges: list[int] | None = None) -> pd.Series:
    """Coarse field-position buckets for empirical tables."""
    if edges is None:
        edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
    labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(edges) - 1)]
    return pd.cut(
        pd.to_numeric(yardline_100, errors="coerce"),
        bins=edges,
        right=False,
        labels=labels,
    )


def ydstogo_bucket(ydstogo: pd.Series) -> pd.Series:
    """ydstogo ∈ {1, 2, 3–5, 6+}."""
    y = pd.to_numeric(ydstogo, errors="coerce")
    out = pd.Series(index=y.index, dtype="object")
    out[y == 1] = "1"
    out[y == 2] = "2"
    out[(y >= 3) & (y <= 5)] = "3-5"
    out[y >= 6] = "6+"
    return out
