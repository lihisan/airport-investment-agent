"""The agent's tools: what it is allowed to *do* in the world.

Each tool has two parts:
1. A definition (name, description, input_schema) that we send to the model so it
   knows the tool exists and what arguments it takes.
2. A Python method that actually runs when the model asks for it.

All numbers come from here, never from the model: the tools read the data snapshot
(data/airports.json, built by scripts/fetch_data.py) and compute with scoring.py.
Only live_status calls an API at chat time (FAA's live airport status).
"""

from __future__ import annotations

import json
import statistics
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import scoring

DATA_PATH = Path(__file__).parent / "data" / "airports.json"
FAA_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"

REGIONS = {
    "new england": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "mid-atlantic": ["NJ", "NY", "PA"],
    "east north central": ["IL", "IN", "MI", "OH", "WI"],
    "west north central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "south atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "east south central": ["AL", "KY", "MS", "TN"],
    "west south central": ["AR", "LA", "OK", "TX"],
    "mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "pacific": ["AK", "CA", "HI", "OR", "WA"],
}

ONTIME_CAVEAT = ("Delay, taxi and route data cover domestic flights of US reporting carriers only "
                 "(BTS On-Time); international and cargo flights are not included.")


GROWTH_CAVEAT = ("Growth compares with the base period (as_of.t100_base), when some airports - especially "
                 "international hubs - were still recovering from COVID; part of their growth is recovery.")


class ToolError(Exception):
    """A tool failed in a way the model should hear about (and can recover from)."""


def _compact(value):
    """Round floats so tool outputs stay small: 0.06283594 -> 0.0628, 26593024.0 -> 26593024."""
    if isinstance(value, float):
        return int(value) if value.is_integer() else round(value, 4)
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def _fetch_faa_status() -> str:
    with urllib.request.urlopen(FAA_STATUS_URL, timeout=10) as resp:
        return resp.read().decode("utf-8")


class Backend:
    """Read-only access to the airport snapshot, plus the live FAA status feed."""

    def __init__(self, data_path=DATA_PATH, fetch_live=None):
        with open(data_path, encoding="utf-8") as f:
            self.snapshot = json.load(f)
        self.airports = self.snapshot["airports"]
        self.as_of = self.snapshot["as_of"]
        self.fetch_live = fetch_live or _fetch_faa_status
        self.scores = scoring.score_all(self.airports)
        # National benchmarks, so the agent compares with real medians instead of guessing "the average".
        peers = [c for c, s in self.scores.items() if s["score"] is not None]
        values = {k: [self.scores[c]["kpis"][k] for c in peers if self.scores[c]["kpis"][k] is not None]
                  for k in ("growth_cagr", "load_factor", "delay_rate", "taxi_out_min")}
        values["pax"] = [self.airports[c]["pax"] for c in peers]
        self.peer_median = {k: statistics.median(v) if v else None for k, v in values.items()}

    # ---------- helpers ----------

    def resolve(self, code_or_name: str) -> str:
        """Turn 'sfo', 'KSFO' or 'San Francisco' into an IATA code."""
        q = (code_or_name or "").strip()
        code = q.upper()
        if code in self.airports:
            return code
        if len(code) == 4 and code.startswith("K") and code[1:] in self.airports:
            return code[1:]
        matches = [c for c, a in self.airports.items()
                   if q and (q.lower() in a["name"].lower() or q.lower() == (a["city"] or "").lower())]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Several airports in one city (e.g. Chicago): pick the busiest, the usual meaning.
            return max(matches, key=lambda c: self.airports[c]["pax"])
        suggestions = sorted((c for c, a in self.airports.items()
                              if q and (q[:3].upper() == c[:len(q[:3])] or q[:4].lower() in a["name"].lower())),
                             key=lambda c: -self.airports[c]["pax"])[:5]
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else " Try find_airports."
        raise ToolError(f"Unknown airport {code_or_name!r}.{hint}")

    def _states(self, region=None, states=None):
        if region:
            key = region.strip().lower()
            if key not in REGIONS:
                raise ToolError(f"Unknown region {region!r}. Known regions: {', '.join(REGIONS)}.")
            return set(REGIONS[key])
        return {s.upper() for s in states} if states else None

    def _info(self, code):
        a = self.airports[code]
        return {"code": code, "name": a["name"], "city": a["city"], "state": a["state"], "pax": a["pax"]}

    def _delay_causes_pct(self, code):
        o = self.airports[code]["ontime"]
        minutes = (o or {}).get("arr_delay_minutes") or {}
        total = sum(minutes.values())
        return {k: round(100 * v / total, 1) for k, v in minutes.items()} if total else None

    def _caveats(self, code):
        caveats = []
        if not self.airports[code]["ontime"]:
            caveats.append("No On-Time data for this airport: congestion is not scored.")
        if not self.airports[code]["pax_base"]:
            caveats.append("No passenger data for the base period: growth is not scored.")
        return caveats

    def _unmet_interpretation(self, code, gap, excess):
        """Plain-language reading of the unmet-demand numbers, decided in code so the model can't misread them."""
        k, lf = self.scores[code]["kpis"], self.scores[code]["kpis"]["load_factor"]
        target = scoring.TARGET_LOAD_FACTOR
        lines = [f"Load factor {lf:.1%} is below the {target:.0%} target: no seat-based gap; flights still have some "
                 "empty seats on average." if gap == 0 else
                 f"Load factor {lf:.1%} is above the {target:.0%} target: about {gap:,} more seats a year would be "
                 "needed to bring it down to the target (seat-based unmet demand)."]
        median = self.peer_median["delay_rate"]
        if excess is not None:
            above = k["delay_rate"] > median
            lines.append(f"Delay rate {k['delay_rate']:.1%} is {'above' if above else 'at or below'} the peer median "
                         f"{median:.1%}" + (f": about {excess:,} more delayed departures than the median rate would give "
                                            "(flight-based signal of traffic the airport cannot handle on time)."
                                            if above else ": no flight-based congestion signal."))
            causes = self._delay_causes_pct(code) or {}
            if causes:
                top = max(causes, key=causes.get)
                lines.append(f"Largest delay cause: {top} ({causes[top]}% of arrival delay minutes).")
        return lines

    # ---------- tools ----------

    def find_airports(self, region=None, state=None, query=None):
        allowed = self._states(region, [state] if state else None)
        found = [c for c, a in self.airports.items()
                 if (allowed is None or a["state"] in allowed)
                 and (not query or query.lower() in a["name"].lower() or query.lower() in (a["city"] or "").lower()
                      or query.upper() == c)]
        found.sort(key=lambda c: -self.airports[c]["pax"])
        return {"airports": [self._info(c) for c in found[:30]], "total_matches": len(found), "as_of": self.as_of}

    def get_airport_metrics(self, code):
        code = self.resolve(code)
        a, s = self.airports[code], self.scores[code]
        o = a["ontime"] or {}
        return {**self._info(code), "seats": a["seats"], "departures": a["departures"],
                "pax_base_period": a["pax_base"], "score": s["score"], "label": s["label"],
                "confidence": s["confidence"], "components": s["components"], "kpis": s["kpis"],
                "not_scored_reason": s["reason"], "peer_median": self.peer_median,
                "ontime_flights": o.get("departures"),
                "cancelled": o.get("cancelled"), "delay_causes_pct": self._delay_causes_pct(code),
                "as_of": self.as_of, "caveats": self._caveats(code) + [GROWTH_CAVEAT] + ([ONTIME_CAVEAT] if o else [])}

    def rank_airports(self, region=None, states=None, top_n=10, weights=None):
        allowed = self._states(region, states)
        w = scoring.normalize_weights(weights)
        scores = self.scores if weights is None else scoring.score_all(self.airports, w)
        codes = [c for c in self.airports if allowed is None or self.airports[c]["state"] in allowed]
        scored = sorted((c for c in codes if scores[c]["score"] is not None),
                        key=lambda c: (-scores[c]["score"], c))[:max(1, int(top_n))]
        ranking = [{"rank": i, "code": c, "name": self.airports[c]["name"], "state": self.airports[c]["state"],
                    "pax": self.airports[c]["pax"], "score": scores[c]["score"], "label": scores[c]["label"],
                    "confidence": scores[c]["confidence"], "components": scores[c]["components"],
                    "kpis": scores[c]["kpis"]} for i, c in enumerate(scored, 1)]
        not_scored = [c for c in codes if scores[c]["score"] is None]
        return {"weights": {k: round(v, 3) for k, v in w.items()}, "ranking": ranking,
                "not_scored": {"codes": sorted(not_scored, key=lambda c: -self.airports[c]["pax"])[:10],
                               "count": len(not_scored),
                               "reason": f"below peer threshold of {scoring.PEER_MIN_PAX:,} annual passengers"},
                "as_of": self.as_of,
                "caveats": ["Percentiles are relative to all US airports with at least "
                            f"{scoring.PEER_MIN_PAX:,} annual passengers.", GROWTH_CAVEAT, ONTIME_CAVEAT]}

    def compare_airports(self, codes):
        if not isinstance(codes, list) or not 2 <= len(codes) <= 6:
            raise ToolError("Give between 2 and 6 airport codes to compare.")
        airports, caveats = [], []
        for c in codes:
            m = self.get_airport_metrics(c)
            caveats += [f"{m['code']}: {x}" if "No " in x else x for x in m.pop("caveats") if x not in caveats]
            m.pop("as_of")
            airports.append(m)
        return {"airports": airports, "as_of": self.as_of, "caveats": list(dict.fromkeys(caveats))}

    def distance_mix(self, code, threshold_mi=scoring.LONG_HAUL_MI):
        code = self.resolve(code)
        o = self.airports[code]["ontime"]
        if not o:
            raise ToolError(f"No route data for {code} (no BTS On-Time flights).")
        return {"code": code, **scoring.distance_mix(o["routes"], float(threshold_mi)),
                "data_period": self.as_of["ontime"], "caveats": [ONTIME_CAVEAT]}

    def unmet_demand(self, code):
        code = self.resolve(code)
        a, k, o = self.airports[code], self.scores[code]["kpis"], self.airports[code]["ontime"]
        median = self.peer_median["delay_rate"]
        excess = (max(0, round(o["delayed"] - o["departures"] * median))
                  if o and o["departures"] and median is not None else None)
        gap = scoring.seat_gap(a["pax"], a["seats"])
        return {"code": code, "seat_gap": gap,
                "interpretation": self._unmet_interpretation(code, gap, excess),
                "excess_delayed_flights": excess, "delay_rate_peer_median": median,
                "taxi_out_peer_median": self.peer_median["taxi_out_min"],
                "pax": a["pax"], "seats": a["seats"], "load_factor": k["load_factor"],
                "target_load_factor": scoring.TARGET_LOAD_FACTOR, "growth_cagr": k["growth_cagr"],
                "delay_rate": k["delay_rate"], "taxi_out_min": k["taxi_out_min"],
                "delay_causes_pct": self._delay_causes_pct(code),
                "data_period": {"passengers_seats": self.as_of["t100"], "flights_delays": self.as_of["ontime"]},
                "caveats": ["Lower bound: at capacity-constrained airports suppressed demand never appears in the data.",
                            "seat_gap = extra annual seats needed for flights to run at an 85% load factor "
                            "(seat-based unmet demand).",
                            "excess_delayed_flights = delayed departures above what the national peer median delay "
                            "rate would give, over the On-Time period (flight-based signal: traffic the airport "
                            "cannot handle on time).",
                            GROWTH_CAVEAT,
                            "Delay causes are minutes of arrival delay into this airport, by BTS cause category; "
                            "'nas' = National Airspace System (airport capacity, air traffic control, non-extreme weather)."]
                + self._caveats(code)}

    def live_status(self, code):
        code = self.resolve(code)
        try:
            root = ET.fromstring(self.fetch_live())
        except Exception as e:  # network down or bad XML: the agent carries on without live data
            return {"available": False, "code": code, "reason": f"FAA live status unavailable ({type(e).__name__})."}
        events = []
        for delay_type in root.iter("Delay_type"):
            kind = delay_type.findtext("Name") or "Delay"
            for el in delay_type.iter():
                if el.findtext("ARPT") == code:
                    details = {c.tag.lower(): (c.text or "").strip() for c in el if c.tag != "ARPT" and len(c) == 0}
                    events.append({"type": kind, **details})
        return {"available": True, "code": code, "events": events,
                "updated": root.findtext("Update_Time"), "source": "FAA NAS Status (live)"}

    # ---------- dispatcher ----------

    def run(self, name, args):
        """Run a tool by name. Returns (output_json, is_error)."""
        method = getattr(self, name, None) if name in TOOL_NAMES else None
        if method is None:
            return json.dumps({"error": f"Unknown tool {name}"}), True
        try:
            return json.dumps(_compact(method(**args)), ensure_ascii=False, separators=(",", ":")), False
        except (ToolError, ValueError, TypeError) as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False), True


def _tool(name, description, properties=None, required=None):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties or {}, "required": required or []}}


CODE = {"code": {"type": "string", "description": "IATA code (e.g. SFO), ICAO (KSFO) or airport/city name"}}
REGION = {"region": {"type": "string", "enum": [r.title() for r in REGIONS],
                     "description": "US census division, e.g. New England"}}

TOOLS = [
    _tool("find_airports", "List US airports by region, state or name, sorted by passengers. "
          "Use to resolve an unclear airport name or to see which airports are in an area.",
          {**REGION, "state": {"type": "string", "description": "Two-letter US state, e.g. CA"},
           "query": {"type": "string", "description": "Part of an airport or city name"}}),
    _tool("get_airport_metrics", "All KPIs, the Expansion Opportunity Score with its breakdown, "
          "delay causes and data caveats for one airport.", CODE, ["code"]),
    _tool("rank_airports", "Rank airports by Expansion Opportunity Score (0-100) within a region or "
          "list of states (or nationally). Optionally with custom KPI weights.",
          {**REGION, "states": {"type": "array", "items": {"type": "string"}, "description": "Two-letter states"},
           "top_n": {"type": "integer", "description": "How many to return (default 10)"},
           "weights": {"type": "object", "description": "Custom weights; omitted keys keep their default",
                       "properties": {k: {"type": "number"} for k in scoring.DEFAULT_WEIGHTS}}}),
    _tool("compare_airports", "Side-by-side KPIs, congestion (delay rate, taxi-out time, delay causes) "
          "and scores for 2-6 airports.",
          {"codes": {"type": "array", "items": {"type": "string"}, "description": "2-6 airport codes"}}, ["codes"]),
    _tool("distance_mix", "Share of an airport's departures that are long-haul (distance >= threshold), "
          "with a distance breakdown and top long-haul routes.",
          {**CODE, "threshold_mi": {"type": "number", "description": "Long-haul threshold in miles (default 3000)"}},
          ["code"]),
    _tool("unmet_demand", "Estimate unmet demand for one airport (seat gap at an 85% load factor) with the "
          "signals that explain it: load factor, growth, delays and delay causes.", CODE, ["code"]),
    _tool("live_status", "Current delays or closures at an airport right now, from the FAA live feed. "
          "Use only when the user asks about today / right now.", CODE, ["code"]),
]
TOOL_NAMES = {t["name"] for t in TOOLS}
