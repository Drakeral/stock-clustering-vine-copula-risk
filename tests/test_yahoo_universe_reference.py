import unittest

from scripts.build_yahoo_universe_reference import (
    TARGET_DATE,
    chart_observation,
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


if __name__ == "__main__":
    unittest.main()
