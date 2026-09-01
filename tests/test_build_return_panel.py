import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_return_panel.py"
SPEC = importlib.util.spec_from_file_location("build_return_panel", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReturnConstructionTests(unittest.TestCase):
    def base_prices(self):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
                "research_ticker": ["TEST"] * 3,
                "ticker": ["OLD", "NEW", "NEW"],
                "continuity_event": ["ticker_change"] * 3,
                "open": [100.0, 50.0, 49.0],
                "high": [101.0, 52.0, 50.0],
                "low": [99.0, 49.0, 48.0],
                "close": [100.0, 51.0, 49.0],
                "volume": [1, 1, 1],
                "transactions": [1, 1, 1],
                "window_start": [1, 2, 3],
            }
        )

    def test_split_and_dividend_total_returns(self):
        splits = pd.DataFrame(
            {"research_ticker": ["TEST"], "date": pd.to_datetime(["2020-01-03"]), "split_factor": [2.0]}
        )
        dividends = pd.DataFrame(
            {"research_ticker": ["TEST"], "date": pd.to_datetime(["2020-01-06"]), "cash_dividend": [2.0]}
        )
        result = MODULE.construct_returns(self.base_prices(), splits, dividends)
        self.assertAlmostEqual(result.loc[1, "total_return"], 0.02)
        self.assertAlmostEqual(result.loc[2, "total_return"], 0.0)
        self.assertTrue(result.loc[1, "ticker_segment_transition"])

    def test_coverage_uses_expected_market_dates(self):
        prices = self.base_prices()
        prices["log_total_return"] = [float("nan"), 0.01, 0.02]
        constituents = [{"ticker": "TEST", "company_name": "Test", "gics_sector": "Test"}]
        dates = pd.date_range("2020-01-02", periods=3, freq="B")
        result = MODULE.coverage_table(
            prices,
            constituents,
            dates,
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-06"),
            0.95,
        )
        self.assertEqual(result.loc[0, "price_coverage"], 1.0)
        self.assertEqual(result.loc[0, "return_coverage"], 1.0)
        self.assertTrue(result.loc[0, "included"])

    def test_manual_stock_distribution_is_in_total_consideration(self):
        prices = self.base_prices().iloc[:2].copy()
        prices.loc[1, "close"] = 70.0
        manual = pd.DataFrame(
            {
                "research_ticker": ["TEST"],
                "date": pd.to_datetime(["2020-01-03"]),
                "manual_action_id": ["TEST_SPIN"],
                "manual_primary_provider_ticker": ["NEW"],
                "manual_primary_share_multiplier": [1.0],
                "manual_cash_per_old_share": [5.0],
                "stock_distribution_value": [25.0],
                "stock_distribution_components": ["[]"],
                "manual_action_source_url": ["https://example.com"],
            }
        )
        result = MODULE.construct_returns(
            prices,
            pd.DataFrame(columns=["research_ticker", "date", "split_factor"]),
            pd.DataFrame(columns=["research_ticker", "date", "cash_dividend"]),
            manual,
        )
        self.assertAlmostEqual(result.loc[1, "total_return"], 0.0)
        self.assertEqual(result.loc[1, "manual_action_id"], "TEST_SPIN")

    def test_terminal_position_is_cash_until_rebalance(self):
        index = pd.to_datetime(["2020-05-08", "2020-05-11", "2020-12-31", "2021-01-04"])
        stock = pd.DataFrame({"TEST": [0.01, float("nan"), float("nan"), float("nan")]}, index=index)
        result = MODULE.build_portfolio_panel(
            stock,
            {
                "TEST": {
                    "cash_carry_start": "2020-05-11",
                    "remove_effective_date": "2021-01-01",
                }
            },
        )
        self.assertEqual(result.loc[pd.Timestamp("2020-05-11"), "TEST"], 0.0)
        self.assertEqual(result.loc[pd.Timestamp("2020-12-31"), "TEST"], 0.0)
        self.assertTrue(pd.isna(result.loc[pd.Timestamp("2021-01-04"), "TEST"]))

    def test_verified_halts_are_zero_and_resumption_keeps_bridge_return(self):
        dates = pd.to_datetime(
            ["2020-11-05", "2020-11-06", "2020-11-09", "2023-06-08", "2023-06-09", "2023-06-12"]
        )
        observed_dates = dates[[0, 2, 3, 5]]
        prices = pd.DataFrame(
            {
                "date": observed_dates,
                "research_ticker": ["BIIB"] * 4,
                "ticker": ["BIIB"] * 4,
                "continuity_event": ["unchanged_ticker"] * 4,
                "open": [328.9, 236.26, 308.88, 313.41],
                "high": [328.9, 236.26, 308.88, 313.41],
                "low": [328.9, 236.26, 308.88, 313.41],
                "close": [328.9, 236.26, 308.88, 313.41],
                "volume": [1] * 4,
                "transactions": [1] * 4,
                "window_start": [1, 2, 3, 4],
            }
        )
        segments = pd.DataFrame(
            {
                "research_ticker": ["BIIB"],
                "provider_ticker": ["BIIB"],
                "start": [pd.Timestamp("2017-01-03")],
                "end": [pd.Timestamp("2025-12-31")],
                "event": ["unchanged_ticker"],
            }
        )
        lifecycle = {
            "verified_halts": [
                {"research_ticker": "BIIB", "date": "2020-11-06", "reason": "halt one"},
                {"research_ticker": "BIIB", "date": "2023-06-09", "reason": "halt two"},
            ]
        }
        completed, audit = MODULE.complete_lifecycle_grid(
            prices, dates, segments, lifecycle, {}, set()
        )
        result = MODULE.construct_returns(
            completed,
            pd.DataFrame(columns=["research_ticker", "date", "split_factor"]),
            pd.DataFrame(columns=["research_ticker", "date", "cash_dividend"]),
        ).set_index("date")
        self.assertEqual(audit["verified_halts_applied"], 2)
        self.assertEqual(audit["issues"], [])
        self.assertEqual(result.loc[pd.Timestamp("2020-11-06"), "total_return"], 0.0)
        self.assertEqual(result.loc[pd.Timestamp("2023-06-09"), "total_return"], 0.0)
        self.assertAlmostEqual(
            result.loc[pd.Timestamp("2020-11-09"), "total_return"], 236.26 / 328.9 - 1
        )
        self.assertAlmostEqual(
            result.loc[pd.Timestamp("2023-06-12"), "total_return"], 313.41 / 308.88 - 1
        )
        self.assertIn("gap_bridge_return", result.loc[pd.Timestamp("2020-11-09"), "quality_flag"])

    def test_unexplained_active_gap_is_materialized_and_fails_lifecycle_audit(self):
        prices = self.base_prices().drop(index=1)
        segments = pd.DataFrame(
            {
                "research_ticker": ["TEST"],
                "provider_ticker": ["NEW"],
                "start": [pd.Timestamp("2020-01-02")],
                "end": [pd.Timestamp("2020-01-06")],
                "event": ["unchanged_ticker"],
            }
        )
        completed, audit = MODULE.complete_lifecycle_grid(
            prices,
            pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            segments,
            {"verified_halts": []},
            {},
            set(),
        )
        missing = completed.loc[completed["date"].eq(pd.Timestamp("2020-01-03"))].iloc[0]
        self.assertEqual(missing["observation_status"], "unexplained_missing")
        self.assertTrue(pd.isna(missing["close"]))
        self.assertEqual(len(audit["issues"]), 1)

    def test_dow_lineage_starts_in_2019_without_legacy_bridge(self):
        constituents = [{"ticker": "DOW"}]
        master = {
            "mappings": {
                "DOW": [
                    {
                        "ticker": "DOW",
                        "start": "2019-04-02",
                        "end": "2025-12-31",
                        "event": "new_dow_inc_inception",
                    }
                ]
            }
        }
        segments = MODULE.build_segments(constituents, master, "2017-01-03", "2025-12-31")
        raw = pd.DataFrame(
            {
                "ticker": ["DOW", "DOW"],
                "date": pd.to_datetime(["2017-08-31", "2019-04-02"]),
                "open": [66.65, 56.25],
                "high": [66.65, 56.25],
                "low": [66.65, 56.25],
                "close": [66.65, 56.25],
                "volume": [1, 1],
                "transactions": [1, 1],
                "window_start": [1, 2],
            }
        )
        mapped = MODULE.map_provider_rows(raw, segments)
        self.assertEqual(mapped["date"].tolist(), [pd.Timestamp("2019-04-02")])
        result = MODULE.construct_returns(
            mapped,
            pd.DataFrame(columns=["research_ticker", "date", "split_factor"]),
            pd.DataFrame(columns=["research_ticker", "date", "cash_dividend"]),
        )
        self.assertTrue(pd.isna(result.loc[0, "total_return"]))

    def test_terminal_consideration_formulas(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config" / "manual_corporate_actions.json").read_text())
        selected = {
            "actions": [
                action
                for action in config["actions"]
                if action["research_ticker"] in {"AGN", "WBA"}
            ]
        }
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-05-07", "2020-05-08", "2025-08-27"]),
                "research_ticker": ["AGN", "AGN", "WBA"],
                "ticker": ["AGN", "AGN", "WBA"],
                "continuity_event": ["unchanged_ticker"] * 3,
                "open": [192.99, 193.02, 11.98],
                "high": [192.99, 193.02, 11.98],
                "low": [192.99, 193.02, 11.98],
                "close": [192.99, 193.02, 11.98],
                "volume": [1, 1, 1],
                "transactions": [1, 1, 1],
                "window_start": [1, 2, 3],
            }
        )
        prices = MODULE.add_synthetic_primary_rows(prices, selected)
        ancillary = pd.DataFrame(
            {"date": [pd.Timestamp("2020-05-08")], "ticker": ["ABBV"], "close": [83.96]}
        )
        actions = MODULE.manual_action_table(selected, ancillary)
        result = MODULE.construct_returns(
            prices,
            pd.DataFrame(columns=["research_ticker", "date", "split_factor"]),
            pd.DataFrame(columns=["research_ticker", "date", "cash_dividend"]),
            actions,
        )
        agn = result.loc[
            result["research_ticker"].eq("AGN") & result["date"].eq(pd.Timestamp("2020-05-08"))
        ].iloc[0]
        wba = result.loc[
            result["research_ticker"].eq("WBA") & result["date"].eq(pd.Timestamp("2025-08-28"))
        ].iloc[0]
        self.assertAlmostEqual(agn["total_consideration_per_old_share"], 120.30 + 0.866 * 83.96)
        self.assertAlmostEqual(wba["total_consideration_per_old_share"], 11.45 + 0.53)
        self.assertEqual(wba["manual_contingent_value_low"], 0.0)
        self.assertEqual(wba["manual_contingent_value_high"], 3.0)

    def test_provider_spinoff_factor_is_replaced_not_double_applied(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config" / "manual_corporate_actions.json").read_text())
        segments = pd.DataFrame(
            {
                "research_ticker": ["IBM"],
                "provider_ticker": ["IBM"],
                "start": [pd.Timestamp("2017-01-03")],
                "end": [pd.Timestamp("2025-12-31")],
                "event": ["unchanged_ticker"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            reference_root = Path(temporary)
            (reference_root / "splits").mkdir()
            (reference_root / "dividends").mkdir()
            (reference_root / "splits" / "IBM.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "id": "P4947cf8a0de38d18c248870eba72b95c03ac83af9444bbd96a711f3d54d354c8",
                                "ticker": "IBM",
                                "execution_date": "2021-11-04",
                                "split_from": 1.0,
                                "split_to": 1.046,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            splits, _, records = MODULE.aggregate_actions_v2(
                reference_root, segments, config
            )
        self.assertFalse(
            ((splits["research_ticker"] == "IBM") & (splits["date"] == pd.Timestamp("2021-11-04"))).any()
        )
        ibm = next(record for record in records if record["event_id"].startswith("P4947"))
        self.assertEqual(ibm["typed_event"], "spin_off")
        self.assertEqual(ibm["application_method"], "manual_action_replacement")
        self.assertEqual(ibm["provider_continuity_factor"], 1.046)

    def test_extreme_review_registry_covers_both_prespecified_rows(self):
        root = Path(__file__).resolve().parents[1]
        reviews = json.loads((root / "config" / "observation_reviews.json").read_text())
        daily = pd.DataFrame(
            {
                "research_ticker": ["OXY", "BIIB"],
                "date": pd.to_datetime(["2020-03-09", "2020-11-04"]),
                "total_return": [-0.5048, 0.4397],
            }
        )
        audit = MODULE.audit_extreme_observations(daily, reviews)
        self.assertEqual(audit["observed_extreme_count"], 2)
        self.assertEqual(audit["issues"], [])
        self.assertEqual({row["disposition"] for row in audit["dispositions"]}, {"retain"})

    def test_public_construction_audit_contains_no_provider_event_rows(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "data/audit/data_quality_report.json").read_text())
        self.assertNotIn("reference_event_reconciliation", report)
        self.assertEqual(
            report["reference_event_record_count"],
            sum(report["reference_event_disposition_counts"].values()),
        )
        self.assertEqual(len(report["reference_event_detail_sha256"]), 64)
        self.assertEqual(report["reference_event_detail_scope"], "local_untracked_audit_only")


if __name__ == "__main__":
    unittest.main()
