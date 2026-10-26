#!/usr/bin/env python3
"""Fit empirical tables + logistic models; write artifacts/."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_ev.baseline import (  # noqa: E402
    build_conversion_table,
    build_ep_by_yardline,
    build_fg_table,
    build_punt_ep_table,
    save_json,
    split_by_play_type,
)
from nfl_ev.features import go_convert_label, fg_make_label  # noqa: E402
from nfl_ev.model import (  # noqa: E402
    empirical_vs_model_mae,
    save_metrics,
    save_model,
    train_binary_logit,
)

PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"


def calibration_plot(cal_bins: list[dict], title: str, path: Path) -> None:
    if not cal_bins:
        return
    pred = [b["pred_mean"] for b in cal_bins]
    obs = [b["obs_rate"] for b in cal_bins]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    ax.plot(pred, obs, "o-", color="#0B6E4F", label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    fourth_path = PROC / "fourth_downs.parquet"
    lean_path = PROC / "pbp_reg_subset.parquet"
    if not fourth_path.exists():
        print("Missing mart. Run: python scripts/build_mart.py", file=sys.stderr)
        return 1

    fourth = pd.read_parquet(fourth_path)
    lean = pd.read_parquet(lean_path) if lean_path.exists() else fourth
    seasons = sorted(int(s) for s in fourth["season"].unique())
    print(f"seasons in mart: {seasons}")
    holdout = seasons[-1]
    train_seasons = [s for s in seasons if s < holdout]
    if not train_seasons:
        train_seasons = seasons[:-1] or seasons
        holdout = seasons[-1]

    go, fg, punt = split_by_play_type(fourth)
    print(f"go={len(go)} fg={len(fg)} punt={len(punt)}")

    ep_by_yl = build_ep_by_yardline(lean)
    conv_table = build_conversion_table(go[go["season"].isin(train_seasons)])
    fg_table = build_fg_table(fg[fg["season"].isin(train_seasons)])
    punt_table = build_punt_ep_table(punt[punt["season"].isin(train_seasons)], ep_by_yl)

    ART.mkdir(parents=True, exist_ok=True)
    save_json(conv_table, ART / "conversion_table.json")
    save_json(fg_table, ART / "fg_table.json")
    save_json(punt_table, ART / "punt_ep_table.json")
    save_json(ep_by_yl, ART / "ep_by_yardline.json")

    go_tr = go[go["season"].isin(train_seasons)]
    go_te = go[go["season"] == holdout]
    fg_tr = fg[fg["season"].isin(train_seasons)]
    fg_te = fg[fg["season"] == holdout]

    convert = train_binary_logit(go_tr, go_te, go_convert_label, "convert")
    fg_model = train_binary_logit(fg_tr, fg_te, fg_make_label, "fg_make")

    save_model(convert["pipeline"], convert["columns"], ART / "convert_logit.joblib")
    save_model(fg_model["pipeline"], fg_model["columns"], ART / "fg_logit.joblib")

    mae = empirical_vs_model_mae(go_te, conv_table, convert["pipeline"], convert["columns"])
    metrics = {
        "seasons": seasons,
        "train_seasons": train_seasons,
        "holdout_season": holdout,
        "convert": convert["metrics"],
        "fg_make": fg_model["metrics"],
        "empirical_vs_model_mae_convert": mae,
        "baseline_global_p_convert": conv_table["global_p"],
        "baseline_global_p_fg": fg_table["global_p"],
        "n_fourth_downs": int(len(fourth)),
    }
    save_metrics(metrics, ART / "metrics.json")

    calibration_plot(
        convert["metrics"].get("calibration") or [],
        f"Convert calibration (holdout {holdout})",
        ART / "calibration_convert.png",
    )
    calibration_plot(
        fg_model["metrics"].get("calibration") or [],
        f"FG make calibration (holdout {holdout})",
        ART / "calibration_fg.png",
    )

    print("metrics:")
    print(
        f"  convert log_loss={convert['metrics'].get('log_loss')} "
        f"brier={convert['metrics'].get('brier')}"
    )
    print(
        f"  fg     log_loss={fg_model['metrics'].get('log_loss')} "
        f"brier={fg_model['metrics'].get('brier')}"
    )
    print(f"  empirical vs model MAE (convert)={mae:.4f}")
    print(f"artifacts → {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
