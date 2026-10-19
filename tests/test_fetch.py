"""Offline tests for the data-fetch helpers (no network)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_data import aggregate_ontime, build_snapshot, month_windows  # noqa: E402


def row(origin, dest, distance, dep15="", taxi="", cancelled="0.00", **delays):
    r = {"Origin": origin, "Dest": dest, "Distance": distance, "DepDel15": dep15, "TaxiOut": taxi,
         "Cancelled": cancelled, "CarrierDelay": "", "WeatherDelay": "", "NASDelay": "",
         "SecurityDelay": "", "LateAircraftDelay": ""}
    r.update(delays)
    return r


class FetchTests(unittest.TestCase):
    def test_month_windows(self):
        w = month_windows("2026-04", "2026-07")
        self.assertEqual(w["t100"], ("2025-05", "2026-04"))
        self.assertEqual(w["t100_base"], ("2022-05", "2023-04"))
        self.assertEqual(w["ontime"][0], (2025, 8))
        self.assertEqual(w["ontime"][-1], (2026, 7))
        self.assertEqual(len(w["ontime"]), 12)

    def test_aggregate_ontime_skips_cancelled_and_splits_origin_dest(self):
        rows = [
            row("ANC", "ATL", "3417.00", dep15="1.00", taxi="20.00"),
            row("SEA", "ANC", "1448.00", dep15="0.00", taxi="10.00", CarrierDelay="5.00",
                WeatherDelay="0.00", NASDelay="30.00", SecurityDelay="0.00", LateAircraftDelay="0.00"),
            row("ANC", "SEA", "1448.00", cancelled="1.00"),
        ]
        acc = {}
        aggregate_ontime(rows, acc)
        anc = acc["ANC"]
        self.assertEqual((anc["departures"], anc["delayed"], anc["cancelled"]), (1, 1, 1))
        self.assertEqual(anc["taxi_out_sum"], 20.0)
        self.assertEqual(anc["routes"], {"ATL": {"distance": 3417, "flights": 1}})
        self.assertEqual(anc["arr_delay_minutes"]["nas"], 30.0)
        self.assertEqual(anc["arr_delay_minutes"]["carrier"], 5.0)

    def test_build_snapshot_joins_and_filters(self):
        t100 = {"ANC": {"pax": 2e6, "seats": 2.5e6, "departures": 2e4},
                "BOS": {"pax": 2e7, "seats": 2.4e7, "departures": 2e5},
                "XYZ": {"pax": 10, "seats": 20, "departures": 1}}
        base = {"BOS": {"pax": 1.8e7, "seats": 2.2e7, "departures": 1.9e5}}
        ontime = {"ANC": {"departures": 5, "delayed": 1, "cancelled": 0, "taxi_out_sum": 50.0,
                          "arr_delay_minutes": {"carrier": 0, "weather": 0, "nas": 0, "security": 0, "late_aircraft": 0},
                          "routes": {}}}
        ourairports = {"ANC": {"name": "Ted Stevens Anchorage International Airport", "city": "Anchorage", "state": "AK"},
                       "BOS": {"name": "Logan International Airport", "city": "Boston", "state": "MA"}}
        snap = build_snapshot(t100, base, ontime, ourairports, month_windows("2026-04", "2026-07"))
        self.assertEqual(set(snap["airports"]), {"ANC", "BOS"})
        self.assertIsNone(snap["airports"]["ANC"]["pax_base"])
        self.assertIsNone(snap["airports"]["BOS"]["ontime"])
        self.assertEqual(snap["airports"]["BOS"]["pax_base"], 1.8e7)
        self.assertEqual(snap["as_of"]["t100"], "2025-05..2026-04")
        self.assertEqual(snap["as_of"]["ontime"], "2025-08..2026-07")


if __name__ == "__main__":
    unittest.main()
