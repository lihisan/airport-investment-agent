"""Deterministic scoring: the numbers behind every ranking. No LLM involved.

Expansion Opportunity Score (0-100) = 100 x weighted average of four KPI percentiles:
- growth      (35%): 3-year passenger CAGR. Demand that keeps rising pays back an expansion.
- capacity    (30%): load factor (passengers / seats). Full planes = demand pressing on capacity.
- congestion  (20%): share of flights delayed 15+ min and average taxi-out time.
- scale       (15%): log10 of annual passengers. A bigger market justifies a bigger investment.
Percentiles are national: each airport is compared with every US airport above the peer threshold,
so the same score means the same thing whatever region the analyst asks about.
"""

from __future__ import annotations

import math
import random

DEFAULT_WEIGHTS = {"growth": 0.35, "capacity": 0.30, "congestion": 0.20, "scale": 0.15}
PEER_MIN_PAX = 500_000
TARGET_LOAD_FACTOR = 0.85
LONG_HAUL_MI = 3000

# Which raw KPIs feed each scoring component (congestion averages two).
COMPONENT_KPIS = {"growth": ["growth_cagr"], "capacity": ["load_factor"],
                  "congestion": ["delay_rate", "taxi_out_min"], "scale": ["log_pax"]}


def kpis(a: dict) -> dict:
    """Raw KPIs for one airport. A KPI is None when its data is missing."""
    pax, base, o = a.get("pax"), a.get("pax_base"), a.get("ontime")
    has_ontime = bool(o and o.get("departures"))
    return {
        "growth_cagr": (pax / base) ** (1 / 3) - 1 if pax and base else None,
        "load_factor": pax / a["seats"] if pax and a.get("seats") else None,
        "delay_rate": o["delayed"] / o["departures"] if has_ontime else None,
        "taxi_out_min": o["taxi_out_sum"] / o["departures"] if has_ontime else None,
        "log_pax": math.log10(pax) if pax else None,
    }


def percentile(value: float, population: list) -> float:
    """Share of the *other* values in the population that are strictly below this value (0-1)."""
    if len(population) <= 1:
        return 1.0
    return sum(x < value for x in population) / (len(population) - 1)


def normalize_weights(weights: dict | None) -> dict:
    """Turn custom weights into a full set that sums to 1.

    All four given: rescaled proportionally. Some given: those are kept exactly (fractions, e.g.
    {"growth": 0.5} means 50%) and the remaining defaults share what is left, in their usual ratio.
    """
    given = {}
    for key, value in (weights or {}).items():
        if key not in DEFAULT_WEIGHTS:
            raise ValueError(f"Unknown weight {key!r}; allowed: {', '.join(DEFAULT_WEIGHTS)}")
        if value is None or value < 0:
            raise ValueError(f"Weight {key!r} must be a number >= 0")
        given[key] = float(value)
    if len(given) == len(DEFAULT_WEIGHTS) or not given:
        merged = {**DEFAULT_WEIGHTS, **given}
        total = sum(merged.values())
        if total <= 0:
            raise ValueError("At least one weight must be greater than 0")
        return {k: v / total for k, v in merged.items()}
    used = sum(given.values())
    if used > 1:
        raise ValueError("Partial weights are fractions of 1 (e.g. 0.5 for 50%) and must sum to at most 1")
    rest = {k: v for k, v in DEFAULT_WEIGHTS.items() if k not in given}
    rest_total = sum(rest.values())
    return {k: given[k] if k in given else (1 - used) * rest[k] / rest_total for k in DEFAULT_WEIGHTS}


def label(score: float) -> str:
    return "strong" if score >= 70 else "moderate" if score >= 50 else "weak"


def score_all(airports: dict, weights: dict | None = None) -> dict:
    """Score every airport. Non-peers (too small) get score None with a reason."""
    w = normalize_weights(weights)
    raw = {code: kpis(a) for code, a in airports.items()}
    peers = [c for c, a in airports.items() if (a.get("pax") or 0) >= PEER_MIN_PAX]
    populations = {k: [raw[c][k] for c in peers if raw[c][k] is not None]
                   for keys in COMPONENT_KPIS.values() for k in keys}

    results = {}
    for code in airports:
        if code not in peers:
            results[code] = {"score": None, "label": None, "confidence": None,
                             "components": {k: None for k in COMPONENT_KPIS}, "kpis": raw[code],
                             "reason": f"below peer threshold of {PEER_MIN_PAX:,} annual passengers"}
            continue
        components = {}
        for comp, keys in COMPONENT_KPIS.items():
            pcts = [percentile(raw[code][k], populations[k]) for k in keys if raw[code][k] is not None]
            components[comp] = round(sum(pcts) / len(pcts), 3) if pcts else None
        available = {k: w[k] for k, v in components.items() if v is not None}
        # Missing components hand their weight to the available ones, proportionally.
        total = sum(available.values())
        score = round(100 * sum(components[k] * wk / total for k, wk in available.items()), 1) if total else 0.0
        n = len(available)
        results[code] = {"score": score, "label": label(score),
                         "confidence": "high" if n == 4 else "medium" if n == 3 else "low",
                         "components": components, "kpis": raw[code], "reason": None}
    return results


def rank_stability(results: dict, codes: list, top_k: int = 3, samples: int = 500,
                   concentration: float = 20.0, seed: int = 42) -> dict:
    """How robust is the ranking to the choice of weights?

    Percentiles don't depend on the weights, so we reuse the components from score_all and
    re-weight them `samples` times. Weights are drawn from a Dirichlet distribution centred on the
    defaults (higher concentration = smaller changes; 20 moves each weight by roughly ±10 points).
    Fixed seed, so the result is deterministic. Returns, per scored airport, the share of scenarios
    in which it ranks in the top_k and its best/worst rank.
    """
    rng = random.Random(seed)
    scored = [c for c in codes if c in results and results[c]["score"] is not None]
    stats = {c: {"top": 0, "min": len(scored), "max": 1} for c in scored}
    keys = list(DEFAULT_WEIGHTS)
    for _ in range(samples):
        draws = [rng.gammavariate(DEFAULT_WEIGHTS[k] * concentration, 1) for k in keys]
        weights = dict(zip(keys, (d / sum(draws) for d in draws)))
        scores = {}
        for c in scored:
            comps = {k: v for k, v in results[c]["components"].items() if v is not None}
            total = sum(weights[k] for k in comps)
            scores[c] = sum(comps[k] * weights[k] for k in comps) / total
        for rank, c in enumerate(sorted(scored, key=lambda c: (-scores[c], c)), 1):
            stats[c]["top"] += rank <= top_k
            stats[c]["min"] = min(stats[c]["min"], rank)
            stats[c]["max"] = max(stats[c]["max"], rank)
    out = {}
    for c, s in stats.items():
        share = round(s["top"] / samples, 3)
        verdict = "robust" if share >= 0.8 else "sensitive" if share > 0.2 else "rarely in top"
        out[c] = {"top_k_share": share, "rank_min": s["min"], "rank_max": s["max"], "verdict": verdict}
    return out


def seat_gap(pax: float, seats: float, target: float = TARGET_LOAD_FACTOR) -> int:
    """Extra seats a year needed for flights to run at the target load factor (0 if none)."""
    return max(0, round(pax / target - seats))


def years_to_ceiling(load_factor, growth, target: float = TARGET_LOAD_FACTOR):
    """Capacity runway: years until load factor reaches the target if demand keeps growing at
    `growth` a year and seats stay flat. A scenario, not a forecast. 0 if already there,
    None if there is no growth (or no data) to get there."""
    if load_factor is None or growth is None:
        return None
    if load_factor >= target:
        return 0.0
    if growth <= 0:
        return None
    return round(math.log(target / load_factor) / math.log(1 + growth), 1)


def distance_mix(routes: dict, threshold_mi: float = LONG_HAUL_MI) -> dict:
    """Share of departures flying at least threshold_mi, plus a distance breakdown."""
    flights = sum(r["flights"] for r in routes.values())
    long_haul = {d: r for d, r in routes.items() if r["distance"] >= threshold_mi}
    long_flights = sum(r["flights"] for r in long_haul.values())
    buckets = {"<800": 0, "800-1799": 0, "1800-2999": 0, "3000+": 0}
    for r in routes.values():
        d = r["distance"]
        key = "<800" if d < 800 else "800-1799" if d < 1800 else "1800-2999" if d < 3000 else "3000+"
        buckets[key] += r["flights"]
    top = sorted(long_haul.items(), key=lambda kv: -kv[1]["flights"])[:5]
    return {
        "threshold_mi": threshold_mi,
        "flights": flights,
        "long_haul_flights": long_flights,
        "long_haul_pct": round(100 * long_flights / flights, 1) if flights else None,
        "buckets": buckets,
        "top_long_haul_routes": [{"dest": d, "distance": r["distance"], "flights": r["flights"]} for d, r in top],
    }
