"""Offline tests for the agent's tools, on a small fixture snapshot."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import TOOLS, Backend  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "airports_fixture.json"
SAMPLE_XML = ("<AIRPORT_STATUS_INFORMATION><Delay_type><Name>Ground Delay Programs</Name><Ground_Delay_List>"
              "<Ground_Delay><ARPT>BOS</ARPT><Reason>low ceilings</Reason><Avg>45 minutes</Avg></Ground_Delay>"
              "</Ground_Delay_List></Delay_type></AIRPORT_STATUS_INFORMATION>")


def run(backend, name, args):
    out, err = backend.run(name, args)
    return json.loads(out), err


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.b = Backend(FIXTURE, fetch_live=lambda: SAMPLE_XML)

    def test_resolve_variants(self):
        for q in ("bos", "KBOS", "BOS", "Boston"):
            self.assertEqual(self.b.resolve(q), "BOS")
        self.assertEqual(self.b.resolve("Providence"), "PVD")

    def test_metrics_include_peer_medians(self):
        out, _ = run(self.b, "get_airport_metrics", {"code": "BOS"})
        self.assertEqual(set(out["peer_median"]), {"growth_cagr", "load_factor", "delay_rate", "taxi_out_min", "pax"})
        self.assertAlmostEqual(out["peer_median"]["delay_rate"], 0.19, places=3)

    def test_resolve_short_names_on_real_data(self):
        real = Backend()  # the committed snapshot
        cases = {"LA": "LAX", "SF": "SFO", "Jackson": "JAN", "Santa Ana": "SNA", "Los Angeles": "LAX", "KSFO": "SFO"}
        self.assertEqual({q: real.resolve(q) for q in cases}, cases)

    def test_unexpected_tool_exception_becomes_error_result(self):
        for name, args in (("rank_airports", {"weights": "abc"}), ("get_airport_metrics", {"code": 123}),
                           ("find_airports", {"query": 5}), ("rank_airports", {"region": 5})):
            out, err = self.b.run(name, args)
            self.assertTrue(err, name)
            self.assertIn("error", json.loads(out))

    def test_interpretation_follows_load_factor_not_rounded_gap(self):
        data = json.loads(FIXTURE.read_text())
        data["airports"]["NEW"].update(pax=6, seats=7)  # LF 85.7% but seat gap rounds to 0
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
        out, _ = run(Backend(f.name, fetch_live=lambda: ""), "unmet_demand", {"code": "NEW"})
        self.assertFalse(any("below" in line for line in out["interpretation"]))

    def test_unknown_airport_suggests(self):
        out, err = run(self.b, "get_airport_metrics", {"code": "ZZZ"})
        self.assertTrue(err)
        self.assertIn("Unknown airport", out["error"])

    def test_metrics_include_score_and_as_of(self):
        out, err = run(self.b, "get_airport_metrics", {"code": "bos"})
        self.assertFalse(err)
        self.assertEqual(out["code"], "BOS")
        self.assertEqual(out["confidence"], "high")
        self.assertIn("as_of", out)
        self.assertAlmostEqual(sum(out["delay_causes_pct"].values()), 100, delta=0.5)

    def test_find_by_region(self):
        out, _ = run(self.b, "find_airports", {"region": "New England"})
        self.assertEqual({a["code"] for a in out["airports"]}, {"BOS", "PVD", "BDL", "LEB", "NEW"})
        self.assertEqual(out["airports"][0]["code"], "BOS")  # sorted by passengers

    def test_rank_new_england(self):
        out, err = run(self.b, "rank_airports", {"region": "new england"})
        self.assertFalse(err)
        codes = [r["code"] for r in out["ranking"]]
        self.assertNotIn("ANC", codes)
        self.assertIn("LEB", out["not_scored"]["codes"])
        self.assertEqual([r["rank"] for r in out["ranking"]], list(range(1, len(codes) + 1)))
        scores = [r["score"] for r in out["ranking"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn("as_of", out)

    def test_rank_custom_weights(self):
        out, _ = run(self.b, "rank_airports", {"states": ["RI", "CT", "MA"], "weights": {"growth": 1, "capacity": 0, "congestion": 0, "scale": 0}})
        self.assertEqual(out["ranking"][0]["code"], "PVD")
        self.assertEqual(out["weights"]["growth"], 1.0)

    def test_rank_bad_weights_is_error(self):
        self.assertTrue(self.b.run("rank_airports", {"weights": {"growth": -1}})[1])
        self.assertTrue(self.b.run("rank_airports", {"weights": {"price": 1}})[1])

    def test_rank_unknown_region_is_error(self):
        self.assertTrue(self.b.run("rank_airports", {"region": "Atlantis"})[1])

    def test_ranking_robustness_tool(self):
        out, err = run(self.b, "ranking_robustness", {"region": "New England"})
        self.assertFalse(err)
        self.assertEqual(out["top_k"], 3)
        self.assertEqual(out["samples"], 500)
        shares = [a["top_k_share"] for a in out["airports"]]
        self.assertEqual(shares, sorted(shares, reverse=True))
        self.assertNotIn("LEB", [a["code"] for a in out["airports"]])
        self.assertTrue(out["caveats"])

    def test_compare_needs_two(self):
        self.assertTrue(self.b.run("compare_airports", {"codes": ["BOS"]})[1])
        out, err = run(self.b, "compare_airports", {"codes": ["BOS", "PVD"]})
        self.assertFalse(err)
        self.assertEqual([a["code"] for a in out["airports"]], ["BOS", "PVD"])

    def test_distance_mix_tool(self):
        out, _ = run(self.b, "distance_mix", {"code": "ANC"})
        self.assertEqual(out["long_haul_pct"], 25.0)
        self.assertTrue(out["caveats"])

    def test_distance_mix_without_ontime_is_error(self):
        self.assertTrue(self.b.run("distance_mix", {"code": "NEW"})[1])

    def test_unmet_demand_has_lower_bound_caveat(self):
        out, _ = run(self.b, "unmet_demand", {"code": "BOS"})
        self.assertEqual(out["seat_gap"], 529412)  # 20M / 0.85 - 23M
        self.assertTrue(any("Lower bound" in c for c in out["caveats"]))

    def test_unmet_demand_flight_based_excess(self):
        # Peers with On-Time data: BOS 0.25, PVD 0.18, BDL 0.18, ANC 0.20 -> median 0.19
        out, _ = run(self.b, "unmet_demand", {"code": "BOS"})
        self.assertAlmostEqual(out["delay_rate_peer_median"], 0.19, places=3)
        self.assertEqual(out["excess_delayed_flights"], 6000)  # 25,000 - 100,000 x 0.19
        out, _ = run(self.b, "unmet_demand", {"code": "PVD"})
        self.assertEqual(out["excess_delayed_flights"], 0)

    def test_unmet_demand_interpretation_when_no_seat_gap(self):
        out, _ = run(self.b, "unmet_demand", {"code": "PVD"})  # LF 76.9% < 85%
        self.assertEqual(out["seat_gap"], 0)
        self.assertTrue(any("no seat-based gap" in line for line in out["interpretation"]))
        out, _ = run(self.b, "unmet_demand", {"code": "BOS"})  # delay rate 25% vs median 19%
        self.assertTrue(any("above the peer median" in line for line in out["interpretation"]))

    def test_outputs_state_their_data_period(self):
        self.assertEqual(run(self.b, "distance_mix", {"code": "ANC"})[0]["data_period"], "2025-08..2026-07")
        out = run(self.b, "unmet_demand", {"code": "BOS"})[0]["data_period"]
        self.assertEqual(out, {"passengers_seats": "2025-05..2026-04", "flights_delays": "2025-08..2026-07"})

    def test_capacity_runway_in_unmet_demand_and_metrics(self):
        out, _ = run(self.b, "unmet_demand", {"code": "PVD"})  # LF 76.9%, CAGR 12.6%
        self.assertEqual(out["years_to_ceiling"], 0.8)
        self.assertTrue(any("scenario, not a forecast" in line for line in out["interpretation"]))
        out, _ = run(self.b, "unmet_demand", {"code": "BOS"})  # LF 87.0% >= 85%
        self.assertEqual(out["years_to_ceiling"], 0.0)
        self.assertTrue(any("already" in line for line in out["interpretation"]))
        self.assertEqual(run(self.b, "get_airport_metrics", {"code": "PVD"})[0]["years_to_ceiling"], 0.8)

    def test_growth_recovery_caveat(self):
        out, _ = run(self.b, "rank_airports", {"region": "New England"})
        self.assertTrue(any("COVID" in c for c in out["caveats"]))

    def test_outputs_are_compact(self):
        out, _ = self.b.run("rank_airports", {"region": "New England"})
        self.assertLess(len(out), 3500)
        floats = [x for x in json.loads(out, parse_float=str).__repr__().split("'") if "." in x and x.replace(".", "").replace("-", "").isdigit()]
        self.assertTrue(all(len(f.split(".")[1]) <= 4 for f in floats), floats[:5])
        out, _ = run(self.b, "compare_airports", {"codes": ["BOS", "PVD"]})
        self.assertNotIn("caveats", out["airports"][0])
        self.assertTrue(out["caveats"])

    def test_live_status_parses(self):
        out, _ = run(self.b, "live_status", {"code": "BOS"})
        self.assertTrue(out["available"])
        self.assertEqual(len(out["events"]), 1)
        self.assertEqual(out["events"][0]["type"], "Ground Delay Programs")

    def test_live_status_unavailable(self):
        def boom():
            raise OSError("down")
        for fetch in (boom, lambda: "<not xml"):
            out, err = run(Backend(FIXTURE, fetch_live=fetch), "live_status", {"code": "BOS"})
            self.assertFalse(err)
            self.assertFalse(out["available"])

    def test_unknown_tool_and_bad_args(self):
        self.assertTrue(self.b.run("delete_everything", {})[1])
        self.assertTrue(self.b.run("get_airport_metrics", {"wrong": "x"})[1])

    def test_tool_definitions_match_methods(self):
        self.assertEqual(len(TOOLS), 8)
        for t in TOOLS:
            self.assertTrue(callable(getattr(self.b, t["name"])))


if __name__ == "__main__":
    unittest.main()
