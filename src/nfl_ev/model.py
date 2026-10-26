"""Train / load logistic models for convert and FG make."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ev.features import (
    feature_matrix,
    fg_make_label,
    go_convert_label,
    yardline_bucket,
    ydstogo_bucket,
)


FEATURE_COLS_CORE = [
    "yardline_100",
    "ydstogo",
    "goal_to_go",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]


def _align_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = X.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = 0.0
    return out[columns]


def make_logit_pipeline(*, class_weight: str | None = None) -> Pipeline:
    """
    Logistic with optional class_weight.

    Default is None: calibrated probabilities matter for EV. Using
    class_weight='balanced' on FG (~86% make) collapsed p_make toward 0.5
    and systematically undervalued field goals in the product EV.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _prepare_xy(df: pd.DataFrame, label_fn) -> tuple[pd.DataFrame, np.ndarray, pd.Index]:
    X_full = feature_matrix(df, include_roof=True, dropna=False)
    y = label_fn(df)
    mask = X_full[FEATURE_COLS_CORE].notna().all(axis=1) & y.notna()
    X = X_full.loc[mask].fillna(0.0)
    # Keep only numeric; drop all-zero rare dummies later via column list
    return X, y.loc[mask].astype(int).to_numpy(), X.index


def train_binary_logit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_fn,
    name: str,
    *,
    class_weight: str | None = None,
) -> dict[str, Any]:
    X_tr, y_tr, _ = _prepare_xy(train_df, label_fn)
    X_te, y_te, te_idx = _prepare_xy(test_df, label_fn)

    columns = list(X_tr.columns)
    X_te = _align_columns(X_te, columns)

    pipe = make_logit_pipeline(class_weight=class_weight)
    pipe.fit(X_tr[columns], y_tr)

    metrics: dict[str, Any] = {"name": name, "n_train": int(len(y_tr)), "n_test": int(len(y_te))}
    if len(y_te) and len(np.unique(y_te)) > 1:
        proba = pipe.predict_proba(X_te[columns])[:, 1]
        metrics["log_loss"] = float(log_loss(y_te, proba, labels=[0, 1]))
        metrics["brier"] = float(brier_score_loss(y_te, proba))
        # Calibration bins
        bins = np.linspace(0, 1, 11)
        digit = np.digitize(proba, bins) - 1
        cal = []
        for b in range(10):
            m = digit == b
            if m.sum() == 0:
                continue
            cal.append(
                {
                    "bin": b,
                    "pred_mean": float(proba[m].mean()),
                    "obs_rate": float(y_te[m].mean()),
                    "n": int(m.sum()),
                }
            )
        metrics["calibration"] = cal
        # Slice metrics
        te = test_df.loc[te_idx].copy()
        te["_p"] = proba
        te["_y"] = y_te
        te["_yl_third"] = pd.cut(
            pd.to_numeric(te["yardline_100"], errors="coerce"),
            bins=[0, 33, 67, 101],
            labels=["own", "mid", "opp"],
            right=False,
        )
        te["_ytg"] = ydstogo_bucket(te["ydstogo"])
        slices = []
        for (yl, yt), g in te.groupby(["_yl_third", "_ytg"], observed=True):
            if len(g) < 5:
                continue
            slices.append(
                {
                    "yardline_third": str(yl),
                    "ydstogo": str(yt),
                    "n": int(len(g)),
                    "brier": float(brier_score_loss(g["_y"], g["_p"])),
                    "obs_rate": float(g["_y"].mean()),
                    "pred_mean": float(g["_p"].mean()),
                }
            )
        metrics["slices"] = slices
    else:
        metrics["log_loss"] = None
        metrics["brier"] = None
        metrics["calibration"] = []
        metrics["slices"] = []
        metrics["note"] = "insufficient holdout class diversity"

    return {"pipeline": pipe, "columns": columns, "metrics": metrics}


def predict_proba_row(pipe: Pipeline, columns: list[str], state: dict[str, Any]) -> float:
    row = {c: 0.0 for c in columns}
    for k, v in state.items():
        if k in row:
            row[k] = float(v) if v is not None else 0.0
    # roof one-hots
    roof = str(state.get("roof", "outdoors") or "outdoors").lower()
    key = f"roof_{roof}"
    if key in row:
        row[key] = 1.0
    X = pd.DataFrame([row])[columns]
    return float(pipe.predict_proba(X)[0, 1])


def save_model(pipe: Pipeline, columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "columns": columns}, path)


def load_model(path: Path) -> tuple[Pipeline, list[str]]:
    obj = joblib.load(path)
    return obj["pipeline"], obj["columns"]


def empirical_vs_model_mae(
    go_test: pd.DataFrame,
    conv_table: dict,
    pipe: Pipeline,
    columns: list[str],
) -> float:
    """MAE between empirical cell rate and model mean p within bucket."""
    from nfl_ev.baseline import lookup_conversion_p

    X, y, idx = _prepare_xy(go_test, go_convert_label)
    if len(y) == 0:
        return float("nan")
    X = _align_columns(X, columns)
    p_model = pipe.predict_proba(X[columns])[:, 1]
    te = go_test.loc[idx].copy()
    te["_pm"] = p_model
    te["yl_b"] = yardline_bucket(te["yardline_100"])
    te["ytg_b"] = ydstogo_bucket(te["ydstogo"])
    errs = []
    for (yl, yt), g in te.groupby(["yl_b", "ytg_b"], observed=True):
        emp, n, _ = lookup_conversion_p(conv_table, float(g["yardline_100"].iloc[0]), float(g["ydstogo"].iloc[0]))
        # use cell p from table if available
        key = f"{yl}|{yt}"
        if key in conv_table["cells"]:
            emp = conv_table["cells"][key]["p"]
        errs.append(abs(emp - float(g["_pm"].mean())))
    return float(np.mean(errs)) if errs else float("nan")


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
