"""Empirical conversion / FG / punt / EP tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ev.features import (
    MIN_CELL_N,
    PLAY_TYPES_FG,
    PLAY_TYPES_GO,
    PLAY_TYPES_PUNT,
    fg_make_label,
    go_convert_label,
    kick_distance_proxy,
    yardline_bucket,
    ydstogo_bucket,
)


def _safe_rate(success: int, n: int, fallback: float) -> float:
    if n >= MIN_CELL_N:
        return float(success) / float(n)
    return float(fallback)


def build_conversion_table(go_df: pd.DataFrame) -> dict[str, Any]:
    """Empirical P(convert | yardline bucket × ydstogo) with min-N backoff."""
    df = go_df.copy()
    df["yl_b"] = yardline_bucket(df["yardline_100"])
    df["ytg_b"] = ydstogo_bucket(df["ydstogo"])
    df["y"] = go_convert_label(df)

    global_rate = float(df["y"].mean()) if len(df) else 0.4
    by_ytg = df.groupby("ytg_b", observed=True)["y"].agg(["sum", "count", "mean"])
    ytg_rates = {
        str(i): float(r["mean"]) if r["count"] >= MIN_CELL_N else global_rate
        for i, r in by_ytg.iterrows()
    }

    cells: dict[str, dict[str, Any]] = {}
    for (yl, yt), g in df.groupby(["yl_b", "ytg_b"], observed=True):
        n = int(len(g))
        s = int(g["y"].sum())
        key = f"{yl}|{yt}"
        fb = ytg_rates.get(str(yt), global_rate)
        cells[key] = {
            "n": n,
            "success": s,
            "p": _safe_rate(s, n, fb),
            "backed_off": n < MIN_CELL_N,
            "yardline_bucket": str(yl),
            "ydstogo_bucket": str(yt),
        }

    return {
        "global_p": global_rate,
        "by_ydstogo": ytg_rates,
        "cells": cells,
        "min_n": MIN_CELL_N,
    }


def build_fg_table(fg_df: pd.DataFrame) -> dict[str, Any]:
    """Empirical P(make | kick-distance bucket) using historical kick_distance when present."""
    df = fg_df.copy()
    if "kick_distance" in df.columns and df["kick_distance"].notna().sum() > 50:
        dist = pd.to_numeric(df["kick_distance"], errors="coerce")
    else:
        dist = kick_distance_proxy(pd.to_numeric(df["yardline_100"], errors="coerce"))
    df["dist"] = dist
    df["y"] = fg_make_label(df)
    # Include an explicit long/impossible bin so own-territory never inherits league make rate.
    edges = [0, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120]
    labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(edges) - 1)]
    df["dist_b"] = pd.cut(df["dist"], bins=edges, right=False, labels=labels)

    global_rate = float(df["y"].mean()) if len(df) else 0.8
    # Long-kick backoff: prefer declining rates, never the short-kick global average.
    long_fallback = 0.05
    cells: dict[str, dict[str, Any]] = {}
    for b, g in df.groupby("dist_b", observed=True):
        n = int(len(g))
        s = int(g["y"].sum())
        label = str(b)
        # Buckets starting at 60+ yards: back off toward near-zero, not global_p.
        try:
            lo = int(label.split("-")[0])
        except ValueError:
            lo = 0
        fb = long_fallback if lo >= 60 else global_rate
        if n == 0:
            p = fb
            backed = True
        elif n < MIN_CELL_N:
            p = float(s) / float(n) if lo >= 60 else _safe_rate(s, n, fb)
            # Shrink sparse long-kick rates toward long_fallback
            if lo >= 60:
                p = 0.3 * (float(s) / float(n)) + 0.7 * long_fallback
            backed = True
        else:
            p = float(s) / float(n)
            backed = False
        cells[label] = {
            "n": n,
            "success": s,
            "p": float(p),
            "backed_off": backed,
        }
    # Ensure ultra-long bin exists even if empty in training data
    if "70-119" not in cells:
        cells["70-119"] = {"n": 0, "success": 0, "p": 0.01, "backed_off": True}
    if "65-69" not in cells:
        cells["65-69"] = {"n": 0, "success": 0, "p": 0.08, "backed_off": True}
    return {"global_p": global_rate, "cells": cells, "min_n": MIN_CELL_N, "edges": edges}


def build_ep_by_yardline(pbp: pd.DataFrame) -> dict[str, float]:
    """Coarse EP-by-yardline curve from nflverse pre-play `ep` (reference scale)."""
    df = pbp.copy()
    if "ep" not in df.columns:
        # Fallback linear heuristic if ep missing
        return {str(y): round(5.0 - 0.07 * y, 4) for y in range(1, 100)}
    df["yl"] = pd.to_numeric(df["yardline_100"], errors="coerce").round().astype("Int64")
    df["ep_num"] = pd.to_numeric(df["ep"], errors="coerce")
    g = df.dropna(subset=["yl", "ep_num"]).groupby("yl")["ep_num"].mean()
    out = {str(int(k)): float(v) for k, v in g.items() if 1 <= int(k) <= 99}
    # Fill gaps by interpolation
    for y in range(1, 100):
        if str(y) not in out:
            lo = next((out[str(i)] for i in range(y - 1, 0, -1) if str(i) in out), None)
            hi = next((out[str(i)] for i in range(y + 1, 100) if str(i) in out), None)
            if lo is not None and hi is not None:
                out[str(y)] = (lo + hi) / 2.0
            elif lo is not None:
                out[str(y)] = lo
            elif hi is not None:
                out[str(y)] = hi
            else:
                out[str(y)] = 5.0 - 0.07 * y
    return out


def build_punt_ep_table(punt_df: pd.DataFrame, ep_by_yl: dict[str, float]) -> dict[str, Any]:
    """
    Empirical expected EP after punt by yardline bucket.

    Prefer net yards → opponent yardline heuristic when next-possession join is heavy.
    Convention: after punt, offense perspective EP ≈ −EP_opp at resulting opp yardline_100.
    """
    df = punt_df.copy()
    df["yl"] = pd.to_numeric(df["yardline_100"], errors="coerce")
    # net_punt yards if present; else typical ~40
    if "punt_net_yards" in df.columns:
        net = pd.to_numeric(df["punt_net_yards"], errors="coerce")
    elif "kick_distance" in df.columns and "return_yards" in df.columns:
        net = pd.to_numeric(df["kick_distance"], errors="coerce") - pd.to_numeric(
            df["return_yards"], errors="coerce"
        ).fillna(0)
    else:
        net = pd.Series(40.0, index=df.index)
    net = net.fillna(40.0)
    # Opponent yardline_100 after punt ≈ 100 - (our_yl - net) wait:
    # Our yardline_100 is yards to opponent EZ. After net punt of N yards,
    # ball is ~ (yardline_100 - N) from our perspective toward their EZ, so
    # opponent's yardline_100 ≈ 100 - (yardline_100 - N) = 100 - yardline_100 + N.
    opp_yl = (100 - df["yl"] + net).clip(1, 99)
    df["opp_yl"] = opp_yl
    df["ep_after"] = df["opp_yl"].map(lambda y: -float(ep_by_yl.get(str(int(round(y))), 0.0)))
    df["yl_b"] = yardline_bucket(df["yl"])

    cells: dict[str, dict[str, Any]] = {}
    for b, g in df.groupby("yl_b", observed=True):
        n = int(len(g))
        cells[str(b)] = {
            "n": n,
            "mean_ep_after": float(g["ep_after"].mean()) if n else -1.5,
            "mean_opp_yl": float(g["opp_yl"].mean()) if n else 70.0,
            "backed_off": n < MIN_CELL_N,
        }
    global_ep = float(df["ep_after"].mean()) if len(df) else -1.5
    return {"global_ep_after": global_ep, "cells": cells, "min_n": MIN_CELL_N}


def lookup_conversion_p(table: dict[str, Any], yardline_100: float, ydstogo: float) -> tuple[float, int, bool]:
    yl_b = str(yardline_bucket(pd.Series([yardline_100])).iloc[0])
    yt_b = str(ydstogo_bucket(pd.Series([ydstogo])).iloc[0])
    cell = table["cells"].get(f"{yl_b}|{yt_b}")
    if cell:
        return float(cell["p"]), int(cell["n"]), bool(cell["backed_off"])
    p = float(table["by_ydstogo"].get(yt_b, table["global_p"]))
    return p, 0, True


def lookup_fg_p(table: dict[str, Any], yardline_100: float) -> tuple[float, int, bool]:
    dist = float(kick_distance_proxy(yardline_100))
    # Hard ceiling: kicks beyond ~65 yards are rare; beyond 70 treat as ~impossible.
    if dist >= 70:
        return 0.01, 0, True
    if dist >= 65:
        return 0.08, 0, True
    edges = table.get("edges", [0, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120])
    labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(edges) - 1)]
    bucket = None
    for i in range(len(edges) - 1):
        if edges[i] <= dist < edges[i + 1]:
            bucket = labels[i]
            break
    if bucket is None:
        return 0.01, 0, True
    cell = table["cells"].get(bucket)
    if cell:
        return float(cell["p"]), int(cell["n"]), bool(cell["backed_off"])
    # Missing mid-range cell: mild backoff, not short-kick global rate for long attempts
    try:
        lo = int(bucket.split("-")[0])
    except ValueError:
        lo = 0
    if lo >= 60:
        return 0.05, 0, True
    return float(table["global_p"]), 0, True


def lookup_punt_ep(table: dict[str, Any], yardline_100: float) -> tuple[float, int, bool]:
    yl_b = str(yardline_bucket(pd.Series([yardline_100])).iloc[0])
    cell = table["cells"].get(yl_b)
    if cell:
        return float(cell["mean_ep_after"]), int(cell["n"]), bool(cell["backed_off"])
    return float(table["global_ep_after"]), 0, True


def lookup_ep(ep_by_yl: dict[str, float], yardline_100: float) -> float:
    y = int(np.clip(round(yardline_100), 1, 99))
    return float(ep_by_yl.get(str(y), 5.0 - 0.07 * y))


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def split_by_play_type(fourth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pt = fourth["play_type"].astype(str)
    go = fourth[pt.isin(PLAY_TYPES_GO)].copy()
    fg = fourth[pt.isin(PLAY_TYPES_FG)].copy()
    punt = fourth[pt.isin(PLAY_TYPES_PUNT)].copy()
    return go, fg, punt
