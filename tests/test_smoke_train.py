"""Tiny synthetic frame fits logistic; artifacts writable."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_ev.baseline import build_conversion_table, save_json, split_by_play_type
from nfl_ev.features import go_convert_label
from nfl_ev.model import save_model, train_binary_logit


def _synth(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yl = rng.integers(1, 99, size=n)
    ytg = rng.integers(1, 10, size=n)
    convert = ((yl < 40) & (ytg <= 2)).astype(int)
    # flip some noise
    flip = rng.random(n) < 0.15
    convert = np.where(flip, 1 - convert, convert)
    return pd.DataFrame(
        {
            "yardline_100": yl,
            "ydstogo": ytg,
            "goal_to_go": (yl <= ytg).astype(int),
            "score_differential": rng.integers(-14, 15, size=n),
            "game_seconds_remaining": rng.integers(60, 3600, size=n),
            "qtr": rng.integers(1, 5, size=n),
            "posteam_timeouts_remaining": rng.integers(0, 4, size=n),
            "defteam_timeouts_remaining": rng.integers(0, 4, size=n),
            "roof": rng.choice(["outdoors", "dome"], size=n),
            "play_type": "pass",
            "first_down": convert,
            "touchdown": 0,
            "field_goal_result": None,
            "season": rng.choice([2023, 2024], size=n),
        }
    )


def test_smoke_logit_and_artifacts(tmp_path: Path):
    df = _synth()
    go, _, _ = split_by_play_type(df)
    table = build_conversion_table(go)
    assert "cells" in table
    save_json(table, tmp_path / "conversion_table.json")
    assert (tmp_path / "conversion_table.json").exists()

    tr = go[go["season"] == 2023]
    te = go[go["season"] == 2024]
    if len(tr) < 30 or len(te) < 10:
        tr, te = go.iloc[:150], go.iloc[150:]
    result = train_binary_logit(tr, te, go_convert_label, "convert")
    assert result["pipeline"] is not None
    save_model(result["pipeline"], result["columns"], tmp_path / "convert_logit.joblib")
    assert (tmp_path / "convert_logit.joblib").exists()
    assert result["metrics"]["n_train"] > 0
