#!/usr/bin/env python3
"""Download nflverse play-by-play parquet for seasons 2021–2025."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests

SEASONS = [2021, 2022, 2023, 2024, 2025]
BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
CSV_BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
MIN_BYTES = 1_000_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 — surface clear offline error
            last_err = e
            wait = 2 ** attempt
            print(f"  attempt {attempt}/{retries} failed: {e}; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(
        f"Failed to download {url}. "
        f"If GitHub is unreachable, place the file manually at {dest}. Last error: {last_err}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
    )
    parser.add_argument("--csv-fallback", action="store_true", help="Prefer csv.gz if parquet fails")
    args = parser.parse_args()

    ok_seasons: list[int] = []
    for season in SEASONS:
        parquet = args.outdir / f"play_by_play_{season}.parquet"
        csv_gz = args.outdir / f"play_by_play_{season}.csv.gz"
        url = BASE.format(season=season)

        if parquet.exists() and parquet.stat().st_size > MIN_BYTES and not args.force:
            digest = sha256_file(parquet)
            print(f"skip {parquet.name} size={parquet.stat().st_size} sha256={digest[:12]}…")
            ok_seasons.append(season)
            continue
        if csv_gz.exists() and csv_gz.stat().st_size > MIN_BYTES and not args.force:
            digest = sha256_file(csv_gz)
            print(f"skip {csv_gz.name} size={csv_gz.stat().st_size} sha256={digest[:12]}…")
            ok_seasons.append(season)
            continue

        print(f"download season={season}")
        try:
            download(url, parquet)
            digest = sha256_file(parquet)
            print(f"  wrote {parquet} size={parquet.stat().st_size} sha256={digest}")
            ok_seasons.append(season)
        except Exception as e:
            print(f"  parquet failed: {e}", file=sys.stderr)
            if args.csv_fallback or True:
                csv_url = CSV_BASE.format(season=season)
                try:
                    print(f"  trying csv.gz fallback…")
                    download(csv_url, csv_gz)
                    digest = sha256_file(csv_gz)
                    print(f"  wrote {csv_gz} size={csv_gz.stat().st_size} sha256={digest}")
                    ok_seasons.append(season)
                except Exception as e2:
                    print(f"  season {season} unavailable: {e2}", file=sys.stderr)

    if len(ok_seasons) < 3:
        print(
            f"ERROR: need ≥3 seasons; got {ok_seasons}. Place files manually in {args.outdir}.",
            file=sys.stderr,
        )
        return 1

    # Prefer consecutive seasons from the frozen set
    print(f"OK seasons available: {ok_seasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
