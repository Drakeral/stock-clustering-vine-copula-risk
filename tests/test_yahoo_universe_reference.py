import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_yahoo_universe_reference import (
    TARGET_DATE,
    chart_observation,
    retrieval_timestamp,
    select_quote,
)


class YahooUniverseReferenceTests(unittest.TestCase):
    def test_quote_selection_requires_exact_symbol(self):
        payload = {"quotes": [{"symbol": "AAPL"}, {"symbol": "AAPL.MX"}]}
        self.assertEqual(select_quote(payload, "aapl"), {"symbol": "AAPL"})
        self.assertIsNone(select_quote(payload, "AAP"))

    def test_chart_observation_selects_target_date(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "ABC"},
                        "timestamp": [1577975400, 1578061800],
                        "indicators": {
                            "quote": [{"close": [10.0, 11.0]}],
                            "adjclose": [{"adjclose": [9.5, 10.5]}],
                        },
                    }
                ]
            }
        }
        metadata, observation = chart_observation(payload)
        self.assertEqual(metadata["symbol"], "ABC")
        self.assertEqual(observation["date"], TARGET_DATE.isoformat())
        self.assertEqual(observation["close"], 10.0)
        self.assertEqual(observation["adjusted_close"], 9.5)

    def test_empty_chart_is_explicitly_missing(self):
        self.assertEqual(chart_observation({"chart": {"result": None}}), ({}, None))

    def test_cached_rebuild_preserves_original_retrieval_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "candidate_sha256": "candidate",
                        "output_sha256": "output",
                        "retrieved_at_utc": "2020-01-03T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            timestamp = retrieval_timestamp(
                manifest_path,
                candidate_sha256="candidate",
                output_sha256="output",
                fetched_any_payload=False,
                current_timestamp="2026-01-01T00:00:00+00:00",
            )
            self.assertEqual(timestamp, "2020-01-03T00:00:00+00:00")

    def test_new_payload_uses_current_retrieval_time(self):
        timestamp = retrieval_timestamp(
            Path("missing.json"),
            candidate_sha256="candidate",
            output_sha256="output",
            fetched_any_payload=True,
            current_timestamp="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(timestamp, "2026-01-01T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
