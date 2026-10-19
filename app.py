"""Web chat for the agent:  streamlit run app.py

Left: the conversation (type, click a sample, or ask by voice). Right: "How I got this answer" - every tool the agent called, with its
raw output, and a score table whenever it ranked or compared airports. The table comes straight
from the deterministic scoring code, so the analyst can check the agent's explanation against it.
"""

import json

import streamlit as st

from agent import Agent
from voice import transcribe

SAMPLE_QUESTIONS = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]

st.set_page_config(page_title="Airport Investment Agent", page_icon="✈️", layout="wide")


def new_conversation():
    st.session_state.agent = Agent()
    st.session_state.messages = []      # [{"role", "content"}] shown in the chat
    st.session_state.turn_traces = []   # tool calls made for each user question
    st.session_state.voice_key = 0      # bumped after each recording so the recorder resets


if "agent" not in st.session_state:
    new_conversation()
agent = st.session_state.agent
as_of = agent.backend.as_of

st.title("✈️ Airport Investment Intelligence Agent")
st.caption(f"Data: BTS T-100 {as_of['t100']} · BTS On-Time {as_of['ontime']} · "
           f"FAA live status · Model: {agent.llm.model} · Screening tool, not investment advice")


def score_rows(tool, output):
    """Flatten rank/compare output into table rows."""
    items = output.get("ranking") if tool == "rank_airports" else output.get("airports", [])
    rows = []
    for a in items or []:
        c, k = a.get("components") or {}, a.get("kpis") or {}
        rows.append({
            "airport": a["code"], "state": a["state"], "score": a.get("score"), "label": a.get("label"),
            "confidence": a.get("confidence"),
            "growth pct": c.get("growth"), "capacity pct": c.get("capacity"),
            "congestion pct": c.get("congestion"), "scale pct": c.get("scale"),
            "CAGR": round(k["growth_cagr"], 3) if k.get("growth_cagr") is not None else None,
            "load factor": round(k["load_factor"], 3) if k.get("load_factor") is not None else None,
            "delayed 15+": round(k["delay_rate"], 3) if k.get("delay_rate") is not None else None,
            "taxi-out min": round(k["taxi_out_min"], 1) if k.get("taxi_out_min") is not None else None,
        })
    return rows


def ask(question, shown=None):
    st.session_state.messages.append({"role": "user", "content": shown or question})
    start = len(agent.trace)
    try:
        with st.spinner("Analyzing..."):
            answer = agent.reply(question)
    except Exception as e:  # LLM provider down, rate limit, missing key...
        answer = f"Sorry, the language model is unavailable right now ({type(e).__name__}). Please try again."
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.turn_traces.append(agent.trace[start:])


chat_col, trace_col = st.columns([3, 2], gap="large")

with chat_col:
    if not st.session_state.messages:
        st.markdown("**Try a question:**")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True):
                ask(q)
                st.rerun()
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    audio = st.audio_input("🎤 Or ask by voice", key=f"voice_{st.session_state.voice_key}")
    if audio is not None:
        st.session_state.voice_key += 1  # new recorder next run, so the same clip isn't sent twice
        try:
            with st.spinner("Transcribing..."):
                spoken = transcribe(audio.getvalue())
        except Exception as e:  # Whisper unavailable: typing still works
            spoken = None
            st.warning(f"Couldn't transcribe the recording ({type(e).__name__}). Please type your question.")
        if spoken:
            ask(spoken, shown=f"🎤 {spoken}")
            st.rerun()
        elif spoken == "":
            st.warning("I didn't catch anything in that recording. Please try again.")
    if st.session_state.messages and st.button("New conversation"):
        new_conversation()
        st.rerun()

question = st.chat_input("Ask about US airports, e.g. 'Rank Pacific airports by expansion potential'")
if question:
    ask(question)
    st.rerun()

with trace_col:
    st.subheader("How I got this answer")
    traces = st.session_state.turn_traces
    if not traces or not traces[-1]:
        st.caption("Tool calls and scores for the latest question appear here.")
    for call in traces[-1] if traces else []:
        args = ", ".join(f"{k}={v!r}" for k, v in call["input"].items())
        output = json.loads(call["output"]) if call.get("output") else {}
        if call["tool"] in ("rank_airports", "compare_airports") and not call["error"]:
            st.markdown(f"**Scores from `{call['tool']}`** (percentiles 0-1, national peers)")
            st.dataframe(score_rows(call["tool"], output), hide_index=True, use_container_width=True)
        with st.expander(f"{'⚠️ ' if call['error'] else ''}{call['tool']}({args})"):
            st.json(output)
