"""Build the data snapshot the agent works from:  python -m scripts.fetch_data

Pulls three public sources and boils them down to one small file, data/airports.json:
- BTS T-100 (data.bts.gov API): passengers, seats and flights per airport, last 12 months
  plus the same 12 months three years earlier (for growth).
- BTS On-Time Reporting (monthly zip files): every domestic flight of the big US airlines,
  with distance, delays, taxi time and delay causes.
- OurAirports: airport names and US states.
The agent never calls these at chat time; it reads the snapshot (fast, and the demo can't break
because an API is down). The live FAA status is the one source queried at chat time (tools.py).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "airports.json"
RAW_DIR = ROOT / "data" / "raw"

T100_URL = "https://data.bts.gov/resource/r495-tyji.json"
ONTIME_URL = ("https://transtats.bts.gov/PREZIP/"
              "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

DELAY_COLUMNS = {"carrier": "CarrierDelay", "weather": "WeatherDelay", "nas": "NASDelay",
                 "security": "SecurityDelay", "late_aircraft": "LateAircraftDelay"}


# ---------- pure helpers (unit-tested) ----------

def _shift(year: int, month: int, months: int) -> tuple:
    """Move a (year, month) by a number of months (negative = back in time)."""
    index = year * 12 + (month - 1) + months
    return index // 12, index % 12 + 1


def _ym(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def month_windows(t100_last: str, ontime_last: str, months: int = 12) -> dict:
    """The periods we aggregate: last 12 T-100 months, same months 3 years earlier, last 12 On-Time months."""
    ty, tm = map(int, t100_last.split("-"))
    oy, om = map(int, ontime_last.split("-"))
    t_start = _shift(ty, tm, -(months - 1))
    return {
        "t100": (_ym(*t_start), _ym(ty, tm)),
        "t100_base": (_ym(*_shift(*t_start, -36)), _ym(*_shift(ty, tm, -36))),
        "ontime": [_shift(oy, om, -i) for i in range(months - 1, -1, -1)],
    }


def _num(value):
    """BTS writes '1.00' for numbers and '' for missing values."""
    value = (value or "").strip()
    return float(value) if value else None


def _new_ontime_entry() -> dict:
    return {"departures": 0, "delayed": 0, "cancelled": 0, "taxi_out_sum": 0.0,
            "arr_delay_minutes": {k: 0.0 for k in DELAY_COLUMNS}, "routes": {}}


def aggregate_ontime(rows, acc: dict) -> None:
    """Fold On-Time CSV rows into per-airport totals.

    Departure stats and routes belong to the Origin airport. Delay-cause minutes belong to the Dest
    airport: they describe delays experienced arriving there (e.g. SFO ground-delay programs).
    Cancelled flights count as cancelled only; they are left out of delays, taxi time and routes.
    """
    for r in rows:
        origin, dest = r["Origin"], r["Dest"]
        o = acc.setdefault(origin, _new_ontime_entry())
        if _num(r["Cancelled"]):
            o["cancelled"] += 1
            continue
        o["departures"] += 1
        if _num(r["DepDel15"]):
            o["delayed"] += 1
        o["taxi_out_sum"] += _num(r["TaxiOut"]) or 0.0
        route = o["routes"].setdefault(dest, {"distance": int(round(_num(r["Distance"]) or 0)), "flights": 0})
        route["flights"] += 1
        d = acc.setdefault(dest, _new_ontime_entry())
        for key, column in DELAY_COLUMNS.items():
            d["arr_delay_minutes"][key] += _num(r[column]) or 0.0


def build_snapshot(t100: dict, t100_base: dict, ontime: dict, ourairports: dict, windows: dict) -> dict:
    """Join the sources on the IATA code. Keep US airports with passengers in the last 12 months."""
    airports = {}
    for code, t in sorted(t100.items()):
        info = ourairports.get(code)
        if not info or not info.get("state") or not t.get("pax"):
            continue
        base = t100_base.get(code)
        o = ontime.get(code)
        airports[code] = {
            "name": info["name"], "city": info["city"], "state": info["state"],
            "pax": t["pax"], "pax_base": base["pax"] if base and base.get("pax") else None,
            "seats": t["seats"], "departures": t["departures"],
            "ontime": o if o and o["departures"] > 0 else None,
        }
    ontime_months = windows["ontime"]
    return {
        "as_of": {
            "t100": "..".join(windows["t100"]),
            "t100_base": "..".join(windows["t100_base"]),
            "ontime": f"{_ym(*ontime_months[0])}..{_ym(*ontime_months[-1])}",
        },
        "sources": ["BTS T-100 Segment by Origin (data.bts.gov r495-tyji)", "BTS On-Time Reporting",
                    "OurAirports", "FAA NAS Status (live)"],
        "airports": airports,
    }


# ---------- downloads ----------

def _get(url: str, timeout: int = 120, retries: int = 2, method: str = "GET") -> bytes:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": "airport-agent/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))


def _socrata(params: dict) -> list:
    return json.loads(_get(T100_URL + "?" + urllib.parse.urlencode(params)))


def latest_t100_month() -> str:
    value = _socrata({"$select": "max(reporting_month) as m"})[0]["m"]
    return value[:7]


def fetch_t100(start: str, end: str) -> dict:
    rows = _socrata({
        "$select": "origin_airport_code, sum(total_passengers) as pax, sum(total_seats) as seats, "
                   "sum(total_departures) as departures",
        "$where": f"reporting_month between '{start}-01T00:00:00' and '{end}-01T00:00:00'",
        "$group": "origin_airport_code",
        "$limit": "50000",
    })
    return {r["origin_airport_code"]: {k: float(r[k]) for k in ("pax", "seats", "departures")} for r in rows}


def latest_ontime_month() -> str:
    year, month = time.gmtime().tm_year, time.gmtime().tm_mon
    for i in range(1, 7):
        y, m = _shift(year, month, -i)
        try:
            _get(ONTIME_URL.format(year=y, month=m), timeout=30, retries=0, method="HEAD")
            return _ym(y, m)
        except Exception:
            continue
    raise RuntimeError("No On-Time file found for the last 6 months")


def fetch_ontime(months: list) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    acc = {}
    for i, (y, m) in enumerate(months, 1):
        path = RAW_DIR / f"ontime_{y}_{m}.zip"
        if not path.exists():
            print(f"  On-Time {y}-{m:02d}: downloading ({i}/{len(months)})...", flush=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(_get(ONTIME_URL.format(year=y, month=m), timeout=600))
            os.replace(tmp, path)
        else:
            print(f"  On-Time {y}-{m:02d}: cached ({i}/{len(months)})", flush=True)
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            with z.open(name) as f:
                aggregate_ontime(csv.DictReader(io.TextIOWrapper(f, encoding="latin-1")), acc)
    return acc


def fetch_ourairports() -> dict:
    text = _get(OURAIRPORTS_URL).decode("utf-8")
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r["iso_country"] == "US" and r["iata_code"]:
            out[r["iata_code"]] = {"name": r["name"], "city": r["municipality"], "state": r["iso_region"][3:]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--months", type=int, default=12)
    args = parser.parse_args()
    step = "starting"
    try:
        step = "BTS T-100 (latest month)"
        windows = month_windows(latest_t100_month(), "2000-01", args.months)
        step = "BTS On-Time (latest month)"
        windows["ontime"] = month_windows("2000-01", latest_ontime_month(), args.months)["ontime"]
        print(f"T-100 {windows['t100']} vs {windows['t100_base']}; On-Time {windows['ontime'][0]}..{windows['ontime'][-1]}")
        step = "BTS T-100"
        t100, base = fetch_t100(*windows["t100"]), fetch_t100(*windows["t100_base"])
        step = "OurAirports"
        ourairports = fetch_ourairports()
        step = "BTS On-Time"
        ontime = fetch_ontime(windows["ontime"])
    except Exception as e:
        print(f"ERROR while fetching {step}: {e}. Existing {OUT_PATH.name} was left unchanged.", file=sys.stderr)
        sys.exit(1)
    snapshot = build_snapshot(t100, base, ontime, ourairports, windows)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(snapshot['airports'])} airports.")


if __name__ == "__main__":
    main()
