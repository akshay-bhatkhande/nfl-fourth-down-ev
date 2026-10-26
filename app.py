"""Fourth-down go / FG / punt expected-points console."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_ev.ev import EvEngine, GameState, load_engine
from nfl_ev.sanity import SANITY_CASES

st.set_page_config(
    page_title="NFL Fourth-Down EV Lab",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual direction: turf green + chalk white (not purple / cream / newspaper)
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"] {
  font-family: 'Source Sans 3', sans-serif;
}
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, #1a5c3a 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #0d3d28 0%, transparent 50%),
    linear-gradient(180deg, #0a2218 0%, #122a1f 40%, #0e1f18 100%);
  color: #f2f5f1;
}
h1, h2, h3, .brand {
  font-family: 'Archivo', sans-serif;
  letter-spacing: -0.02em;
}
.brand {
  font-size: 2.4rem;
  font-weight: 800;
  color: #e8f5e9;
  margin-bottom: 0.15rem;
}
.tagline {
  color: #b7c9bb;
  font-size: 1.05rem;
  margin-bottom: 1.2rem;
}
.rec-badge {
  display: inline-block;
  padding: 0.55rem 1.1rem;
  font-family: 'Archivo', sans-serif;
  font-weight: 800;
  font-size: 1.35rem;
  border-radius: 4px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.rec-go { background: #c8f560; color: #0a2218; }
.rec-fg { background: #f0c75e; color: #1a1505; }
.rec-punt { background: #7eb6ff; color: #061428; }
.delta {
  font-size: 1.15rem;
  color: #d7e8db;
  margin-top: 0.4rem;
}
.reason { color: #a8bbae; margin-top: 0.35rem; }
.caveat-warn {
  border-left: 3px solid #f0c75e;
  padding-left: 0.75rem;
  color: #f5e6b8;
}
footer, .footer-note {
  color: #7f9486;
  font-size: 0.85rem;
}
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0c1f17 0%, #10261c 100%);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_engine() -> EvEngine:
    return load_engine(ROOT / "artifacts")


def badge_class(rec: str) -> str:
    return {"go": "rec-go", "fg": "rec-fg", "punt": "rec-punt"}.get(rec, "rec-go")


def label_action(rec: str) -> str:
    return {"go": "GO FOR IT", "fg": "FIELD GOAL", "punt": "PUNT"}.get(rec, rec.upper())


def ev_figure(ev_go: float, ev_fg: float, ev_punt: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Go", "Field goal", "Punt"],
                y=[ev_go, ev_fg, ev_punt],
                marker_color=["#c8f560", "#f0c75e", "#7eb6ff"],
                text=[f"{ev_go:.2f}", f"{ev_fg:.2f}", f"{ev_punt:.2f}"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Proxy expected points by action",
        yaxis_title="EV (offense perspective)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,40,28,0.65)",
        font=dict(color="#e8f5e9", family="Source Sans 3"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=340,
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def main() -> None:
    st.markdown('<div class="brand">NFL Fourth-Down EV Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">On 4th down, should we go, kick, or punt? '
        "Answered with expected-points deltas — not win probability or betting odds.</div>",
        unsafe_allow_html=True,
    )

    art = ROOT / "artifacts"
    needed = [
        "conversion_table.json",
        "fg_table.json",
        "punt_ep_table.json",
        "ep_by_yardline.json",
        "metrics.json",
        "convert_logit.joblib",
        "fg_logit.joblib",
    ]
    missing = [f for f in needed if not (art / f).exists()]
    if missing:
        st.error(
            "Model artifacts missing: "
            + ", ".join(missing)
            + ". From repo root run: `python scripts/download_pbp.py` → "
            "`python scripts/build_mart.py` → `python scripts/train_models.py`."
        )
        st.stop()

    try:
        engine = get_engine()
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

    metrics = engine.metrics
    seasons = metrics.get("seasons", [])
    holdout = metrics.get("holdout_season", "?")

    with st.sidebar:
        st.markdown("### Diagnostics")
        st.caption(f"Seasons: {seasons}")
        st.caption(f"Holdout: {holdout}")
        conv_m = metrics.get("convert", {})
        fg_m = metrics.get("fg_make", {})
        st.markdown("**Holdout metrics**")
        st.write(
            {
                "convert_log_loss": conv_m.get("log_loss"),
                "convert_brier": conv_m.get("brier"),
                "fg_log_loss": fg_m.get("log_loss"),
                "fg_brier": fg_m.get("brier"),
                "emp_vs_model_mae": metrics.get("empirical_vs_model_mae_convert"),
            }
        )
        use_model = st.toggle("Use logistic model (off = empirical only)", value=True)
        st.markdown("---")
        st.markdown(
            '<p class="footer-note">Not betting advice. Free nflverse PBP only — '
            "no paid odds APIs.</p>",
            unsafe_allow_html=True,
        )

    # Defaults: own 35, 4th & 2, tied, Q2 mid
    if "preset_state" not in st.session_state:
        st.session_state.preset_state = None

    st.markdown("#### Extreme-situation presets")
    cols = st.columns(len(SANITY_CASES))
    for col, case in zip(cols, SANITY_CASES):
        if col.button(case.id, help=case.label, use_container_width=True):
            st.session_state.preset_state = case.state

    preset = st.session_state.preset_state
    defaults = preset or GameState()

    c1, c2, c3 = st.columns(3)
    with c1:
        yardline_100 = st.slider(
            "Yards to opponent end zone (`yardline_100`)",
            1,
            99,
            int(defaults.yardline_100),
            help="Own 35 ≈ 65; opponent 1 ≈ 1.",
        )
        ydstogo = st.slider("Yards to go", 1, 20, int(defaults.ydstogo))
        score_differential = st.number_input(
            "Score differential (posteam − defteam)",
            -28,
            28,
            int(defaults.score_differential),
        )
    with c2:
        qtr = st.selectbox("Quarter", [1, 2, 3, 4], index=int(defaults.qtr) - 1)
        game_seconds_remaining = st.slider(
            "Game seconds remaining",
            0,
            3600,
            int(defaults.game_seconds_remaining),
        )
        roof = st.selectbox(
            "Roof",
            ["outdoors", "dome", "open", "closed"],
            index=["outdoors", "dome", "open", "closed"].index(
                defaults.roof if defaults.roof in ["outdoors", "dome", "open", "closed"] else "outdoors"
            ),
        )
    with c3:
        posteam_to = st.slider(
            "Offense timeouts", 0, 3, int(defaults.posteam_timeouts_remaining)
        )
        defteam_to = st.slider(
            "Defense timeouts", 0, 3, int(defaults.defteam_timeouts_remaining)
        )
        st.caption("Goal-to-go is inferred when yardline ≤ yards to go.")

    state = GameState(
        yardline_100=yardline_100,
        ydstogo=ydstogo,
        score_differential=score_differential,
        qtr=qtr,
        game_seconds_remaining=game_seconds_remaining,
        posteam_timeouts_remaining=posteam_to,
        defteam_timeouts_remaining=defteam_to,
        roof=roof,
    )

    result = engine.score(state, use_model=use_model)

    left, right = st.columns([1.05, 1.2])
    with left:
        st.markdown("#### Recommendation")
        st.markdown(
            f'<span class="rec-badge {badge_class(result.recommendation)}">'
            f"{label_action(result.recommendation)}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="delta">ΔEV vs {result.second_best}: '
            f"<strong>{result.delta_ev:+.2f}</strong> proxy EP</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="reason">{result.reason}</div>', unsafe_allow_html=True)
        st.markdown(
            f"p(convert) **{result.p_conv:.1%}** "
            f"(baseline {result.p_conv_baseline:.1%}, N={result.sample_n_conv}) · "
            f"p(FG make) **{result.p_make:.1%}** "
            f"(baseline {result.p_make_baseline:.1%}, N={result.sample_n_fg})"
        )
        with st.expander("Caveats & confidence", expanded=result.low_confidence):
            for c in result.caveats:
                st.markdown(f'<div class="caveat-warn">{c}</div>', unsafe_allow_html=True)
            st.markdown(
                "Methodology: see README and [`docs/decisions.md`](docs/decisions.md)."
            )

    with right:
        st.plotly_chart(
            ev_figure(result.ev_go, result.ev_fg, result.ev_punt),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown(
        '<p class="footer-note">Data: nflverse / nflfastR play-by-play (REG seasons). '
        "EP scale uses nflverse pre-play `ep` as a published reference — not as a leaked "
        "feature of the conversion model. Human-owned metrics; AI used as build accelerator.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
