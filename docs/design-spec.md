# Airport Investment Intelligence Agent — Design Spec

Date: 2026-10-19 · Deadline: 2026-10-20 09:00 · Budget: ~9–10 working hours

## 1. Goal

Help analysts at a US airport-modernization investment firm find airports where renovation is most
likely to pay off because flight/passenger demand is outgrowing capacity. The agent answers
conversational questions, ranks/compares airports with **deterministic** logic, explains its
reasoning, and states assumptions, uncertainty and scope.

Must handle the four sample questions well:

| Question | Capability it tests | Tool(s) |
|---|---|---|
| New England candidates for terminal expansion | regional ranking | `find_airports`, `rank_airports` |
| Compare LAX and SNA congestion | side-by-side KPIs | `compare_airports` |
| % long-haul flights out of ANC | route-distance mix | `distance_mix` |
| Unmet flight demand at SFO and why | capacity gap + constraint signals | `unmet_demand`, `get_airport_metrics` |

Out of scope: non-US airports, construction cost / ROI / regulation, investment advice.

## 2. Architecture

```
scripts/fetch_data.py ──► data/airports.json ──► scoring.py ──► tools.py ──► agent.py + llm.py ──► app.py (Streamlit)
  (run once, public APIs)   (small snapshot)      (pure code)    (7 tools)    (agent loop)          chat + trace panel
                                                                    ▲
                                                           FAA NAS status API (live)
```

- `agent.py`: provider-agnostic tool-calling loop with a 10-step cap and a trace of every tool call.
- `llm.py`: model providers behind one interface (Groq default `openai/gpt-oss-120b`, Claude optional).
- `tools.py` / `scoring.py` / `data/`: domain tools, deterministic scoring and the data snapshot.
- `prompts.py`: system prompt. `guardrails.py`: prompt-injection check on user input.

**Core principle:** the LLM never produces numbers. It interprets the question, chooses tools and
explains results. Every figure comes from code; every answer carries the data period and caveats.

## 3. Data (hybrid: snapshot via public APIs + one live source)

| Source | Access | Used for |
|---|---|---|
| BTS "AFF – T100 Segment Summary By Origin Airport" (`data.bts.gov/resource/r495-tyji.json`, Socrata API) | API, monthly per origin airport, 2014–2026 | passengers, departures, seats, load factor, avg distance; growth |
| BTS On-Time Reporting PREZIP (`transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_YYYY_M.zip`, ~27 MB/month) | public download | per-flight origin, destination, distance, delay ≥15 min, taxi-out, delay causes (carrier/weather/NAS/security/late aircraft), cancellations |
| OurAirports `airports.csv` | public CSV | name, state, region, coordinates |
| FAA NAS Status (`nasstatus.faa.gov/api/airport-status-information`, XML) | live API | current delays / closures |

`fetch_data.py` aggregates the latest 12 months (L12M) plus the same 12 months three years earlier,
and writes one small `data/airports.json` including an `as_of` period. Raw zips are not kept in
the repo. On failure it retries, then exits with a clear message without overwriting the old snapshot.

**Known data limits (to be stated by the agent):** On-Time covers domestic flights of reporting
carriers only (no international, no cargo); T-100 has no origin–destination breakdown via API;
snapshot, not real time (except FAA live status).

## 4. Scoring methodology (`scoring.py`, pure functions)

**Peer group:** US commercial airports with ≥ ~500k L12M passengers. Percentiles are computed
nationally, so "strong in New England" also means strong nationally.

**KPIs (L12M):**

| KPI | Formula | Weight |
|---|---|---|
| Growth | 3-year passenger CAGR = (pax_L12M / pax_3y_ago)^(1/3) − 1 | 0.35 |
| Capacity pressure | load factor = pax / seats | 0.30 |
| Congestion | 0.5·pct(share of departures delayed ≥15 min) + 0.5·pct(avg taxi-out min) | 0.20 |
| Scale | log(pax_L12M) | 0.15 |

**Expansion Opportunity Score** = 100 × Σ weight_k × percentile_k (0–100).
Labels: ≥70 strong, 50–70 moderate, <50 weak. Every result returns the per-KPI breakdown.

**Missing data:** if a KPI is unavailable (e.g. no On-Time data), its weight is redistributed
proportionally across available KPIs and the airport gets a confidence label (`high` / `medium` / `low`).

**Custom weights:** `rank_airports` accepts optional weights (normalized to sum to 1) so analysts
can ask "what if growth were 50%?". Defaults are a stated judgment call.

**Unmet demand:**
`seat_gap = max(0, pax / 0.85 − seats)` — extra annual seats needed for flights to run at a
practical 85% load factor. Reported with constraint signals: share of delay minutes caused by NAS
(airport/ATC capacity) and weather, avg taxi-out, growth trend. Stated as a **lower bound**: at a
constrained airport, airlines cannot add flights, so suppressed demand never appears in the data.

**Long-haul share:** `distance_mix(code, threshold_mi=3000)` = share of L12M departures with
distance ≥ threshold (default 3,000 mi ≈ 6 h; adjustable). Caveat: domestic reporting carriers only.

## 5. Tools (`tools.py`)

| Tool | Returns |
|---|---|
| `find_airports(region?, state?, query?)` | matching airports (code, name, state, pax) |
| `get_airport_metrics(code)` | all KPIs, raw values, as_of, confidence |
| `rank_airports(region?, states?, top_n=10, weights?)` | ranked list with scores and breakdown |
| `compare_airports(codes)` | raw KPIs, delay causes and score side by side |
| `distance_mix(code, threshold_mi=3000)` | long-haul %, distance buckets, flight count, caveat |
| `unmet_demand(code)` | seat_gap, load factor, delay-cause shares, growth, caveats |
| `live_status(code)` | current FAA delays/closures, or "live data unavailable" |

Unknown codes return an error with close suggestions. Regions: New England = CT, ME, MA, NH, RI, VT
(plus a small table of other census regions).

## 6. Agent behavior (`prompts.py`)

Answer structure: (1) direct answer, (2) numbers + reasoning from tools, (3) **Assumptions &
limitations** (data period, coverage, confidence). Rules: numbers only from tools, otherwise say
"I don't have data on that"; separate "the data shows" from "general knowledge (not from data)";
scope = US airports; it is a screening tool, not investment advice. Follow-ups work through the
conversation history kept by the agent loop.

## 7. Interface (`app.py`, Streamlit)

Left: chat (history in `st.session_state`). Right: "How I got this answer": tool calls from
`agent.trace` and a score table when ranking/compare ran. Top banner: data period + "screening tool,
not investment advice". Four buttons with the sample questions. A terminal `chat.py --debug` is also available.
Voice (Groq Whisper) is optional, only if time remains.

## 8. Error handling

FAA API down → tool reports unavailable, agent continues on snapshot. Unknown airport → error with
suggestions. Missing KPIs → confidence label. Fetch failure → retry, keep old snapshot. LLM failure →
friendly message.

## 9. Testing

- **Offline unit tests** on a small fixture JSON: weights sum to 1, percentile ranks, missing-data
  reweighting, seat_gap known example (pax 90, seats 95 → ≈10.9), distance_mix threshold, tool output
  shape, unknown-code error. No network, no LLM.
- **Behavioral evals** against the real model: the four sample questions call the right tools; the
  ANC long-haul % in the answer equals the tool value (no invented numbers); a follow-up ("what about
  Boston?") works; out-of-scope request is declined; answer includes an assumptions section. Run twice.

## 10. Key tradeoffs (for the design document)

- **Snapshot + live vs live-only:** reliability and speed for analysis, explicit data date; only
  FAA status is live. OpenSky rejected (historical data requires an account).
- **Weighted percentile score vs rule tiers vs forecast model:** transparent, deterministic and ranks
  within tiers; weights are a judgment call and are exposed/adjustable. Forecasting is out of a one-day scope.
- **Agent harness:** Considered a framework such as deepagents / LangGraph (planning, sub-agents,
  virtual filesystem). Chose a minimal custom loop because each question needs only 1–3 tool calls,
  and transparency/debuggability matter more than orchestration. If the product grows into long
  multi-step research tasks (e.g. full investment memos), a harness with planning and sub-agents
  would become worthwhile.
- **Safety:** read-only tools on public data for internal analysts → prompt scope rules, code-sourced
  numbers, injection check and step cap. For an external or broad deployment, add a guard model
  (e.g. Llama Guard) as an input/output filter. For off-topic detection, a calibrated classifier such as Jev
  (TypeSafe AI) would be used first in the evals, to measure scope adherence without adding per-message latency.
- **Proxy metrics:** load factor and delays are proxies for capacity constraint; seat_gap is a lower bound.

## 11. Where AI is used

LLM: understanding the question, choosing tools and parameters, explaining results, follow-ups.
Not used for: data, scoring, ranking, arithmetic.

**Models:**
- **Primary:** Groq `openai/gpt-oss-120b`.
- **Automatic fallback on rate limits:** Groq `openai/gpt-oss-20b`, then `qwen/qwen3.8-27b`. Each has its own
  free-tier quota. Every new question retries the primary model first, and the UI shows which model answered.
- **Speech-to-text:** Groq `whisper-large-v3-turbo` for voice questions.
- **Alternative provider:** Anthropic Claude (`claude-opus-5-5`, via `LLM_PROVIDER=claude`), behind the same
  interface in `llm.py`.

All of them play the same role and use the same tools, so the numbers are identical whichever model
answers; only the wording and reasoning quality differ.

## 12. Milestones

1. Fetch script + `airports.json`; sanity-check SFO/LAX/ANC (2 h)
2. `scoring.py` + tests (1.5 h)
3. Tools + prompt + loop → **end-to-end in terminal (submittable checkpoint)** (1.5 h)
4. Streamlit (1 h)
5. Evals, passing twice (1 h)
6. Design doc + README (1 h)
7. Voice if time + demo practice (1 h)
