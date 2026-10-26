#!/usr/bin/env python3
"""Filter nflverse PBP into a 4th-down mart (DuckDB + parquet)."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SEASONS = [2021, 2022, 2023, 2024, 2025]

KEEP_COLS = [
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "posteam",
    "defteam",
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "roof",
    "play_type",
    "first_down",
    "touchdown",
    "field_goal_result",
    "kick_distance",
    "punt_net_yards",
    "return_yards",
    "ep",
    "epa",
    "desc",
]


def find_season_files() -> list[tuple[int, Path]]:
    found = []
    for s in SEASONS:
        pq = RAW / f"play_by_play_{s}.parquet"
        csv = RAW / f"play_by_play_{s}.csv.gz"
        if pq.exists() and pq.stat().st_size > 1_000_000:
            found.append((s, pq))
        elif csv.exists() and csv.stat().st_size > 1_000_000:
            found.append((s, csv))
    return found


def load_season(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def main() -> int:
    files = find_season_files()
    if len(files) < 3:
        print(
            f"Need ≥3 season files in {RAW}; found {[s for s, _ in files]}. "
            "Run: python scripts/download_pbp.py",
            file=sys.stderr,
        )
        return 1

    frames = []
    for season, path in files:
        print(f"loading {path.name}…")
        df = load_season(path)
        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        if "season" not in df.columns:
            df["season"] = season
        frames.append(df)

    all_pbp = pd.concat(frames, ignore_index=True)
    print(f"raw rows: {len(all_pbp):,} seasons={sorted(all_pbp['season'].unique().tolist())}")

    PROC.mkdir(parents=True, exist_ok=True)
    all_path = PROC / "pbp_reg_subset.parquet"

    con = duckdb.connect()
    con.register("pbp", all_pbp)
    # Persist a lean all-downs subset for EP curve (REG only)
    lean = con.execute(
        """
        SELECT *
        FROM pbp
        WHERE season_type = 'REG'
        """
    ).df()
    lean.to_parquet(all_path, index=False)
    print(f"wrote {all_path} rows={len(lean):,}")

    fourth = con.execute(
        """
        SELECT *
        FROM pbp
        WHERE season_type = 'REG'
          AND down = 4
          AND play_type IN ('pass', 'run', 'field_goal', 'punt')
        """
    ).df()
    out = PROC / "fourth_downs.parquet"
    fourth.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(fourth):,}")
    print(fourth.groupby(["season", "play_type"]).size().unstack(fill_value=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
