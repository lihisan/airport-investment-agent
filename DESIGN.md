# Airport Investment Intelligence Agent: Design

## 1. Problem and scope

Analysts at a US airport-modernization investment firm need to find airports where terminal and capacity
expansion is most likely to pay off, because passenger and flight demand is outgrowing what the airport can
serve. The agent answers questions in a chat. It ranks and compares airports with **deterministic,
transparent scoring**, explains its reasoning, and states its assumptions and uncertainty.

**In scope:** US airports; demand growth, capacity pressure, congestion, route mix, unmet-demand signals.
**Out of scope:** construction cost, ROI, financing, regulation, non-US airports. This is a **screening
tool, not investment advice**, and the agent says so.

## 2. Architecture

```
scripts/fetch_data.py ──► data/airports.json ──► scoring.py ──► tools.py ──► agent.py + llm.py ──► app.py
 (public APIs, run once)   (12-month snapshot)    (pure code)    (7 tools)    (tool-calling loop)  Streamlit chat
                                                                    ▲                               + trace panel
                                                           FAA NAS Status API (live)
```

| Module | Responsibility |
|---|---|
| `scripts/fetch_data.py` | Pulls the public sources and aggregates them into one small snapshot (~0.5 MB, 1,045 airports). |
| `scoring.py` | KPIs, percentiles, Expansion Opportunity Score, seat gap, distance mix. Pure functions with no I/O and no LLM. |
| `tools.py` | 7 read-only tools over the snapshot, plus the live FAA feed. Every tool output carries its data period and caveats, plus a code-computed interpretation where misreading is likely. |
| `agent.py` | Provider-agnostic loop: model → tool calls → results → model. Caps each question at 10 steps and keeps a trace of every call. |
| `llm.py` | Groq (default `openai/gpt-oss-120b`, with automatic fallback models) or Claude, behind one interface. |
| `app.py` | Streamlit chat. The side panel shows every tool call and the score table behind the answer. |

**Core principle:** *the LLM decides what to ask, the code decides the answer, the LLM explains it.*
Every number in an answer comes from a tool result.

## 3. Data

| Source | Access | Used for |
|---|---|---|
| BTS T-100 Segment by Origin Airport (`data.bts.gov`, dataset `r495-tyji`) | Socrata REST API | Passengers, seats, departures; last 12 months (May 2025–Apr 2026) vs the same months 3 years earlier |
| BTS On-Time Reporting (monthly files on `transtats.bts.gov`) | Public download | Every domestic flight of the large US carriers (Aug 2025–Jul 2026): distance, delays, taxi-out, delay causes |
| OurAirports | Public CSV | Airport names, cities and states (for regions) |
| FAA NAS Status | Live REST API (XML) | Current delays and closures, queried at chat time |

**Why a snapshot?** The analysis questions are about structural trends, so data that is at most a few
months old is fine. The snapshot makes answers fast and reproducible, and the demo can't break because an
API is down. The snapshot dates are shown in the UI and in every answer. One source (FAA) stays live to
show the pattern. Refresh with `python -m scripts.fetch_data` (about 15 minutes, 12 monthly files).

**Known data limits.** The agent surfaces these as caveats:
- On-Time covers **domestic flights of US reporting carriers only**: no international and no cargo flights.
  This matters at hubs like ANC (a major transpacific cargo hub), where the true long-haul share is far
  higher than the domestic figure.
- T-100 passengers are **enplanements** (departing passengers). That is roughly half of the
  "total passengers" figure airports publish.
- The growth base period (May 2022–Apr 2023) still included **COVID recovery**, so growth is partly
  recovery, especially at international hubs.

## 4. Scoring methodology

### Expansion Opportunity Score (0–100)

| Component | KPI | Weight | Business question |
|---|---|---|---|
| Growth | 3-year passenger CAGR = (pax / pax₃ᵧ₋ₐ₉ₒ)^(1/3) − 1 | 35% | Is demand rising? Without that, expansion doesn't pay back. |
| Capacity pressure | Load factor = passengers / seats | 30% | Are planes already full? |
| Congestion | Mean of percentiles of (share of departures delayed ≥ 15 min) and (avg taxi-out min) | 20% | Is the airport itself the bottleneck? |
| Scale | log₁₀(annual passengers) | 15% | Is the market big enough to justify investment? |

1. **Peer group:** the 141 US airports with ≥ 500,000 annual passengers. Smaller fields are listed as
   "not scored" so they don't distort the percentiles.
2. **Each KPI becomes a national percentile** (0–1): the share of peers with a lower value. Percentiles
   make KPIs with different units comparable and are robust to outliers. They are **national**, so
   "strong in New England" also means strong in the US.
3. **Score** = 100 × Σ weight × percentile. Labels: ≥ 70 *strong*, 50–70 *moderate*, < 50 *weak*.
4. **Missing data:** a missing component's weight is redistributed proportionally across the others, and
   the airport's **confidence** drops: high when all 4 components are present, medium with 3, low with 2 or fewer.
5. **Custom weights:** analysts can ask, for example, "rank again with growth at 50%". The weights they
   give are kept exactly (growth = 0.50), the other components share the remaining 50% in their default
   ratio, and the tool returns the weights it used. The result is still deterministic.

**Worked example: Boston Logan (BOS), real data**

| Component | KPI value | National median | Percentile | Weight | Contribution |
|---|---|---|---|---|---|
| Growth | 3.87% CAGR | 3.77% | 0.521 | 0.35 | 18.2 |
| Capacity | 81.7% load factor | 78.4% | 0.807 | 0.30 | 24.2 |
| Congestion | 23.4% delayed, 21.4 min taxi-out | 19.9%, 16.5 min | 0.860 | 0.20 | 17.2 |
| Scale | 21.0M enplanements | 2.1M | 0.893 | 0.15 | 13.4 |
| **Score** | | | | | **73.0 → strong** |

### Unmet demand

The assignment asks about unmet demand at SFO, which isn't directly observable, so the tool reports two
signals plus a code-computed interpretation:
- **Seat-based:** `seat_gap = max(0, passengers / 0.85 − seats)`, the extra annual seats needed for flights to
  run at a practical 85% load factor.
- **Flight-based:** `excess_delayed_flights = delayed departures − departures × national median delay rate`.
  It counts the traffic the airport cannot handle on time.
- Context: growth, taxi-out time vs median, and **delay causes** (BTS categories; *NAS* = airport and
  air-traffic capacity).

At **SFO** the seat gap is **0** (load factor 82.6%), but there are **≈ 6,200 excess delayed departures a year**
(24.1% delayed vs a 19.9% median), and **46%** of arrival-delay minutes are NAS. The constraint is runway and
airspace capacity, not seats. Both signals are **lower bounds**: at a constrained airport, airlines can't add
flights, so suppressed demand never appears in the data.

### Long-haul share

`distance_mix` = share of departures with distance ≥ threshold (default **3,000 miles**, about 6 hours; adjustable
in the conversation), plus distance buckets and the top long-haul routes. ANC: **4.9%** of domestic departures,
with the caveat that international and cargo flights are excluded.

## 5. Where AI is used, and where it isn't

| LLM | Deterministic code |
|---|---|
| Understanding the question and follow-ups, using conversation history | Data collection and aggregation |
| Choosing tools and arguments (region, codes, weights, threshold) | KPIs, percentiles, scores, rankings |
| Explaining results, structuring the answer, stating caveats | Unmet-demand and long-haul calculations |
| Clearly labeled general context, such as runway layout | Interpretation of edge cases (e.g. "no seat gap"), benchmarks, data periods |

**Models:**
- **Primary:** Groq `openai/gpt-oss-120b`.
- **Automatic fallback on rate limits:** Groq `openai/gpt-oss-20b`, then `qwen/qwen3.8-27b`. Each has its own
  free-tier quota. Every new question retries the primary model first, and the UI shows which model answered.
- **Alternative provider:** Anthropic Claude (`claude-opus-5-5`, via `LLM_PROVIDER=claude`), behind the same
  interface in `llm.py`.

All of them play the same role and use the same tools, so the numbers are identical whichever model
answers; only the wording and reasoning quality differ.

**Guardrails in code, not only in the prompt:**
- Tools are read-only.
- Numbers come only from tools, and tool outputs carry their own interpretation and benchmarks so the model has nothing to guess.
- A prompt-injection check runs on user input.
- Each question is capped at 10 tool steps.
- Out-of-scope requests are declined.

## 6. Key tradeoffs

- **Weighted percentile score vs rule-based tiers vs a forecasting model:**
  - Percentiles are transparent, deterministic and rank within tiers.
  - Tiers are coarse.
  - Forecasting needs much more data and validation than a day allows.
  - The weights are a judgment call. They are stated, explained and adjustable per question.
- **Snapshot + one live source vs everything live:** we chose reliability and reproducibility over freshness,
  and the data period is always shown. OpenSky (live flights) was rejected because historical data needs an account.
- **National percentiles:** comparable across regions, but small fast-growing airports (e.g. TYS, SGF) can
  outrank big hubs nationally, because scale is only 15%. A fund with a minimum deal size would raise the
  peer threshold or the scale weight. Both are one-line changes.
- **Proxies:** load factor and delays are proxies for capacity constraint. Ground truth (gate counts, runway
  capacity) isn't available in public APIs.
- **Agent harness:** we considered a framework such as deepagents / LangGraph (planning, sub-agents, virtual
  filesystem). We chose a minimal custom loop because each question needs only 1–3 tool calls, and
  transparency and debuggability matter more than orchestration. If the product grows into long multi-step
  research tasks, such as full investment memos, a harness with planning and sub-agents becomes worthwhile.
- **Free-tier limits shaped the implementation:**
  - Groq caps a single request (8k tokens, including `max_tokens`) and daily usage per model.
  - Tool outputs are therefore compact, and raw tool results from earlier turns are truncated. The model's
    earlier answers are kept, so follow-ups still work.
  - On a rate limit, the agent falls back to `gpt-oss-20b` and then `qwen3.8-27b`, and the UI shows the
    active model. Fallback answers are less polished, but they are still built on tool numbers.
- **Safety:** these are read-only tools on public data for internal analysts, so the controls are prompt scope
  rules, code-sourced numbers, an injection check and a step cap. For an external or broad deployment we would
  add a guard model (e.g. Llama Guard or `gpt-oss-safeguard`, both on Groq) as an input/output filter, and an
  off-topic classifier in the evals.

## 7. Evaluation

- **52 offline unit tests** (`python -m unittest discover tests`) cover:
  - fetch aggregation
  - scoring math, including missing data, bad weights and edge cases
  - every tool, including unknown airports and FAA being down
  - the agent loop with a fake model, including history compaction and model fallback
- **8 behavioral evals** against the real model (`python evals/run_evals.py --repeat 2`):
  - the four assignment questions
  - a follow-up
  - custom weights
  - an out-of-scope request
  - an unknown airport

  Checks include "the number in the reply equals the tool's number" and "the reply has an assumptions
  section". Result: **8/8 in two consecutive runs.**
- **What evals caught:** reading real answers exposed the model inventing a seat gap when the tool said 0, and
  quoting a made-up "national average". Both were fixed in code, by adding a computed interpretation and real
  peer medians, not just in the prompt. The evals also caught the provider's request-size limit before the demo did.

## 8. Future work

- Origin–destination data including international and cargo traffic (BTS T-100 international segments) for a true long-haul share.
- Demand forecasting (population, income, airline schedules) to turn growth into a capacity-gap forecast.
- Infrastructure data (gates, runways, FAA capacity benchmarks) to replace the proxies.
- Voice input (Groq `whisper-large-v3`).
- A guard model and larger eval sets in production.
