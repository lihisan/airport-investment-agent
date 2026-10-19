"""Offline tests for the deterministic scoring logic."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import (DEFAULT_WEIGHTS, distance_mix, kpis, label, normalize_weights,  # noqa: E402
                     percentile, rank_stability, score_all, seat_gap, years_to_ceiling)

FIXTURE = Path(__file__).parent / "fixtures" / "airports_fixture.json"
FIX = json.loads(FIXTURE.read_text())["airports"]


class ScoringTests(unittest.TestCase):
    def test_default_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(normalize_weights(None).values()), 1.0)

    def test_partial_weights_merge_and_normalize(self):
        w = normalize_weights({"growth": 0.5})
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertGreater(w["growth"], DEFAULT_WEIGHTS["growth"])

    def test_partial_weights_are_kept_exact(self):
        w = normalize_weights({"growth": 0.5})
        self.assertAlmostEqual(w["growth"], 0.5)
        self.assertAlmostEqual(w["capacity"] / w["scale"], DEFAULT_WEIGHTS["capacity"] / DEFAULT_WEIGHTS["scale"])
        with self.assertRaises(ValueError):
            normalize_weights({"growth": 50})  # partial weights are fractions, not percents

    def test_bad_weights_raise(self):
        for bad in ({"price": 1}, {"growth": -1}, {k: 0 for k in DEFAULT_WEIGHTS}):
            with self.assertRaises(ValueError):
                normalize_weights(bad)

    def test_percentile(self):
        self.assertEqual(percentile(3, [1, 2, 3]), 1.0)
        self.assertEqual(percentile(1, [1, 2, 3]), 0.0)
        self.assertEqual(percentile(5, [5]), 1.0)

    def test_seat_gap_known_example(self):
        self.assertEqual(seat_gap(90, 95), 11)  # 90 / 0.85 = 105.9 -> gap 10.9 -> 11
        self.assertEqual(seat_gap(50, 100), 0)

    def test_growth_none_when_base_missing(self):
        self.assertIsNone(kpis(FIX["NEW"])["growth_cagr"])
        self.assertIsNone(kpis(dict(FIX["BOS"], pax_base=0))["growth_cagr"])

    def test_non_peer_not_scored(self):
        r = score_all(FIX)["LEB"]
        self.assertIsNone(r["score"])
        self.assertIn("500,000", r["reason"])

    def test_missing_data_lowers_confidence_but_scores(self):
        scores = score_all(FIX)
        self.assertIsNotNone(scores["NEW"]["score"])
        self.assertEqual(scores["NEW"]["confidence"], "low")
        self.assertEqual(scores["BOS"]["confidence"], "high")

    def test_scores_deterministic_and_in_range(self):
        a, b = score_all(FIX), score_all(FIX)
        self.assertEqual(a, b)
        self.assertTrue(all(0 <= r["score"] <= 100 for r in a.values() if r["score"] is not None))

    def test_weights_change_scores(self):
        growth_only = {"growth": 1, "capacity": 0, "congestion": 0, "scale": 0}
        self.assertNotEqual(score_all(FIX)["PVD"]["score"], score_all(FIX, growth_only)["PVD"]["score"])
        self.assertEqual(score_all(FIX, growth_only)["PVD"]["score"], 100.0)  # fastest-growing peer

    def test_distance_mix(self):
        routes = FIX["ANC"]["ontime"]["routes"]
        m = distance_mix(routes)
        self.assertEqual((m["flights"], m["long_haul_flights"], m["long_haul_pct"]), (40, 10, 25.0))
        self.assertEqual(m["top_long_haul_routes"][0]["dest"], "ATL")
        self.assertEqual(m["buckets"], {"<800": 0, "800-1799": 30, "1800-2999": 0, "3000+": 10})
        self.assertEqual(distance_mix(routes, 1000)["long_haul_pct"], 100.0)
        self.assertIsNone(distance_mix({})["long_haul_pct"])

    def test_years_to_ceiling(self):
        self.assertEqual(years_to_ceiling(0.80, 0.05), 1.2)    # ln(0.85/0.80) / ln(1.05) = 1.24
        self.assertEqual(years_to_ceiling(0.826, 0.063), 0.5)
        self.assertEqual(years_to_ceiling(0.90, 0.05), 0.0)    # already above the ceiling
        self.assertIsNone(years_to_ceiling(0.80, 0.0))         # no growth: never reaches it
        self.assertIsNone(years_to_ceiling(0.80, -0.02))
        self.assertIsNone(years_to_ceiling(None, 0.05))

    def test_rank_stability_is_deterministic_and_consistent(self):
        codes = ["BOS", "PVD", "BDL", "NEW"]
        a = rank_stability(score_all(FIX), codes, top_k=3)
        self.assertEqual(a, rank_stability(score_all(FIX), codes, top_k=3))  # fixed seed
        self.assertEqual(set(a), set(codes))
        # Every scenario has exactly 3 airports in the top 3.
        self.assertAlmostEqual(sum(r["top_k_share"] for r in a.values()), 3.0, places=6)
        for r in a.values():
            self.assertTrue(0 <= r["top_k_share"] <= 1)
            self.assertTrue(1 <= r["rank_min"] <= r["rank_max"] <= 4)
            self.assertIn(r["verdict"], {"robust", "sensitive", "rarely in top"})

    def test_rank_stability_skips_unscored(self):
        out = rank_stability(score_all(FIX), ["BOS", "LEB"], top_k=1)  # LEB is below the peer threshold
        self.assertEqual(set(out), {"BOS"})
        self.assertEqual(out["BOS"]["top_k_share"], 1.0)

    def test_labels(self):
        self.assertEqual((label(70), label(69.9), label(49.9)), ("strong", "moderate", "weak"))


if __name__ == "__main__":
    unittest.main()
