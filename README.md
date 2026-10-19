# Airport Investment Intelligence Agent

A chat agent that helps analysts find US airports where terminal expansion is most likely to pay off.
It uses public aviation data (BTS, FAA, OurAirports), ranks airports with a deterministic
**Expansion Opportunity Score**, explains its reasoning and states its assumptions.

Design, scoring methodology and tradeoffs: **[DESIGN.md](DESIGN.md)**

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "GROQ_API_KEY=<your key>" > .env      # free key at https://console.groq.com
```

Optional: `LLM_PROVIDER=claude` with `ANTHROPIC_API_KEY` uses Claude instead of Groq.

## Run

```bash
.venv/bin/streamlit run app.py          # web chat + "how I got this answer" panel + voice input
.venv/bin/python chat.py --debug        # terminal chat, prints every tool call
```

Try:
- Which airports in New England are strong candidates for terminal expansion?
- Compare LA and Santa Ana airport congestion levels.
- What is the percentage of long haul flights out of Anchorage airport?
- What is the unmet flight demand in SFO airport and why?
- Or click **🎤 Or ask by voice** and speak the question (transcribed by Whisper on Groq; the browser asks for microphone access once).
- Follow-ups such as "What about Boston?", "Rank them again with growth weighted 50%", or "Use 2,000 miles as long-haul".

## Data

`data/airports.json` is a snapshot (T-100: May 2025–Apr 2026, On-Time: Aug 2025–Jul 2026) built from public
APIs. To refresh it (about 15 minutes; monthly files are cached in `data/raw/`):

```bash
.venv/bin/python -m scripts.fetch_data
```

## Tests and evals

```bash
.venv/bin/python -m unittest discover tests      # 55 offline tests, no network or API key
.venv/bin/python evals/run_evals.py --repeat 2   # 8 behavioral evals against the real model
```

## Project layout

| Path | What it is |
|---|---|
| `app.py`, `chat.py` | Web (Streamlit) and terminal chat |
| `agent.py` | Tool-calling agent loop (step cap, trace, history compaction) |
| `llm.py` | Groq / Claude providers behind one interface, with model fallback |
| `tools.py` | The 7 tools the agent can call |
| `scoring.py` | Deterministic KPIs, scores, unmet demand, distance mix |
| `voice.py` | Voice input: speech-to-text with Whisper on Groq |
| `prompts.py`, `guardrails.py` | System prompt; prompt-injection check |
| `scripts/fetch_data.py` | Builds the data snapshot from public sources |
| `tests/`, `evals/` | Unit tests; behavioral evals |
| `docs/` | Design spec and implementation plan |
