# Build log — nfl-fourth-down-ev-lab

AI-assisted iterate log. **Human owns metrics, cut list, and EV convention** (`docs/decisions.md`).

## Timeline

1. **Scaffold** — Repo layout per frozen spec; `.gitignore` for raw parquet; public GitHub `akshay-bhatkhande/nfl-fourth-down-ev-lab`.
2. **Ingest** — `scripts/download_pbp.py` pulls nflverse `pbp` parquet 2021–2025 with retries + SHA256; offline path documented.
3. **Mart** — DuckDB/pandas filter: REG, down==4, play_type ∈ {pass, run, field_goal, punt}.
4. **Baseline** — Empirical conversion / FG / punt-EP tables with min-N=30 backoff; EP-by-yardline from nflverse `ep`.
5. **Model** — Primary `LogisticRegression` for convert + FG make; season holdout; log-loss, Brier, calibration bins, yardline×ydstogo slices. No model zoo.
6. **EV + sanity** — Frozen FG +3 / miss heuristic; S1–S5 presets + pytest hard fails.
7. **Streamlit** — Single-page console: controls, presets, recommendation + ΔEV, Plotly bars, caveats, sidebar metrics.
8. **Polish** — Interview README, decisions (leaky rejects), demo screenshots/GIF to Project media store.

## AI role

- Accelerated boilerplate (download retries, Streamlit layout CSS, test stubs).
- Did **not** choose hire-bar metric (ΔEV vs next best under proxy EP) or expand into WP/odds.

## Commands used (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_pbp.py
python scripts/build_mart.py
python scripts/train_models.py
pytest -q
streamlit run app.py
```
