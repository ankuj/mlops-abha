"""CupGuard Streamlit agent — World Cup match prediction with ML lifecycle controls."""

import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import (
    load_team_stats,
    predict_fixture,
    run_lifecycle,
)

st.set_page_config(page_title="CupGuard — 2026 World Cup Predictor", page_icon="⚽", layout="wide")
st.title("⚽ CupGuard — 2026 FIFA World Cup Match Predictor")
st.caption(
    "Predict match outcomes from historical international results. "
    "Correct the agent to improve future versions."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Pick two teams and I'll estimate **home win / draw / away win** probabilities. "
                "After a match, tell me the actual result so the feedback loop can improve the model."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    st.header("ML lifecycle")
    st.markdown(
        "Run the full pipeline: `ingest()` → `transform()` → `train()` → "
        "`evaluate()` → `package()`"
    )

    if st.button("Run full pipeline", type="primary"):
        with st.spinner("Running lifecycle..."):
            try:
                result = run_lifecycle()
                metrics = result["evaluate"]["metrics"]
                st.session_state.pipeline_result = result
                st.success(
                    f"Pipeline complete — accuracy {metrics['accuracy']:.1%}, "
                    f"f1_macro {metrics['f1_macro']:.3f}"
                )
            except Exception as exc:
                st.error(str(exc))

    st.markdown("---")
    st.header("Fixture setup")

    try:
        stats = load_team_stats()
        teams = sorted(stats["team"].tolist())
    except FileNotFoundError:
        st.warning("Run the pipeline first to build team stats.")
        teams = []

    if teams:
        home = st.selectbox(
            "Home team",
            teams,
            index=teams.index("Brazil") if "Brazil" in teams else 0,
        )
        away_options = [t for t in teams if t != home]
        away = st.selectbox("Away team", away_options, index=0)
        neutral = st.checkbox("Neutral venue", value=True)
    else:
        home = away = None
        neutral = True


if teams and st.button("Predict outcome"):
    try:
        probs = predict_fixture(home, away, neutral=int(neutral), is_world_cup=1)
        best = max(probs, key=probs.get)
        lines = "\n".join(
            f"- {k.replace('_', ' ')}: **{v:.1%}**"
            for k, v in sorted(probs.items(), key=lambda x: -x[1])
        )
        reply = (
            f"**{home}** vs **{away}** ({'neutral' if neutral else 'home venue'})\n\n"
            f"{lines}\n\nMost likely: **{best.replace('_', ' ')}**"
        )
        st.session_state.last_prediction = {
            "home": home,
            "away": away,
            "neutral": neutral,
            "predicted": best,
            "probs": probs,
        }
    except Exception as exc:
        reply = f"Could not predict: `{exc}`. Run **Run full pipeline** in the sidebar first."
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
