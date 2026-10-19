"""Offline tests for the number-grounding check (hallucination detector)."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grounding import extract_numbers, normalize_text, ungrounded_numbers  # noqa: E402

TOOL = json.dumps({"long_haul_pct": 4.9, "flights": 19779, "load_factor": 0.8255, "pax": 26593024,
                   "delay_rate": 0.2414, "as_of": {"ontime": "2025-08..2026-07"}})


class GroundingTests(unittest.TestCase):
    def test_extract_numbers_handles_formats(self):
        found = {n.text: n.value for n in extract_numbers("4.9% of 19,779 flights; 26.6M passengers; 1.2k delays")}
        self.assertEqual(found, {"4.9%": 4.9, "19,779": 19779, "26.6M": 26.6e6, "1.2k": 1200})

    def test_numbers_from_tools_are_grounded_with_rounding_and_units(self):
        reply = "4.9% of 19,779 departures are long-haul. Load factor 82.6%, delays 24.1%, 26.6M passengers."
        self.assertEqual(ungrounded_numbers(reply, [TOOL]), [])

    def test_invented_number_is_flagged(self):
        reply = "Delays are 24.1%, well above the national average of about 15%."
        self.assertEqual(ungrounded_numbers(reply, [TOOL]), ["15%"])

    def test_numbers_from_prompt_question_and_dates_are_ignored(self):
        reply = "Weights: growth 35%, capacity 30%. Threshold 3,000 miles. Data 2025-08 to 2026-07. Top 3 airports."
        self.assertEqual(ungrounded_numbers(reply, [TOOL], allowed_text="35% 30% 3,000"), [])

    def test_narrow_space_thousands_separator(self):
        reply = "about 6\u202f219 delayed flights, delay rate 24.1\u202f%"
        self.assertEqual(normalize_text(reply), "about 6,219 delayed flights, delay rate 24.1 %")
        self.assertEqual(ungrounded_numbers(reply, [json.dumps({"excess": 6219, "delay_rate": 0.2414})]), [])

    def test_no_tools_means_every_real_number_is_unverified(self):
        self.assertEqual(ungrounded_numbers("SFO handles 57 million passengers.", []), ["57 million"])


if __name__ == "__main__":
    unittest.main()
