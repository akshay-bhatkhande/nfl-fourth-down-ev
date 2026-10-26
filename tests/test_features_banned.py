"""Banned leakage columns must never enter feature_matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_ev.features import BANNED_FEATURES, BANNED_PREFIXES, assert_no_banned, feature_matrix


def test_banned_list_nonempty():
    assert "epa" in BANNED_FEATURES
    assert "wp" in BANNED_FEATURES
    assert "first_down" in BANNED_FEATURES
    assert "fg_prob" in BANNED_FEATURES
    assert "xyac_epa" in BANNED_FEATURES
    assert "xyac_" in BANNED_PREFIXES
    assert "fumble_" in BANNED_PREFIXES


def test_feature_matrix_excludes_banned():
    df = pd.DataFrame(
        {
            "yardline_100": [65, 20],
            "ydstogo": [2, 1],
            "goal_to_go": [0, 1],
            "score_differential": [0, -3],
            "game_seconds_remaining": [1800, 90],
            "qtr": [2, 4],
            "posteam_timeouts_remaining": [3, 1],
            "defteam_timeouts_remaining": [3, 2],
            "roof": ["outdoors", "dome"],
            "epa": [0.5, -0.2],
            "wp": [0.6, 0.4],
            "yards_gained": [5, 0],
            "first_down": [1, 0],
        }
    )
    X = feature_matrix(df)
    for col in BANNED_FEATURES:
        assert col not in X.columns
    assert "yardline_100" in X.columns
    assert "ydstogo" in X.columns


def test_assert_no_banned_raises():
    with pytest.raises(ValueError, match="Banned"):
        assert_no_banned(["yardline_100", "epa"])


def test_assert_no_banned_xyac_prefix():
    with pytest.raises(ValueError, match="xyac"):
        assert_no_banned(["yardline_100", "xyac_mean_yardage"])
