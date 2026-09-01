import unittest

import numpy as np
import pandas as pd

from scripts.validate_portfolio_arithmetic import validate_annual_reconstruction


class AnnualPortfolioArithmeticValidationTests(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame(
            {
                "A": [np.nan, 0.10, 0.02, -0.03],
                "B": [np.nan, -0.05, np.nan, np.nan],
                "C": [np.nan, 0.01, 0.04, 0.05],
            },
            index=pd.to_datetime(["2020-01-02", "2020-01-03", "2021-01-04", "2021-01-05"]),
        )
        self.schedule = {
            "years": [
                {
                    "year": 2020,
                    "rebalance_date": "2020-01-02",
                    "security_count": 3,
                    "active_tickers": ["A", "B", "C"],
                },
                {
                    "year": 2021,
                    "rebalance_date": "2021-01-04",
                    "security_count": 2,
                    "active_tickers": ["A", "C"],
                },
            ]
        }
        self.labels = {"A": "one", "B": "one", "C": "two"}

    def test_removed_column_does_not_remain_in_later_group_weights(self):
        report = validate_annual_reconstruction(self.panel, self.schedule, self.labels)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checked_return_dates"], 3)
        self.assertEqual(report["undefined_initial_dates"], ["2020-01-02"])
        self.assertLessEqual(report["maximum_absolute_error"], 1e-12)
        results = {row["year"]: row for row in report["annual_results"]}
        self.assertEqual(results[2020]["group_sizes"], {"one": 2, "two": 1})
        self.assertEqual(results[2021]["group_sizes"], {"one": 1, "two": 1})

    def test_identity_is_grouping_invariant(self):
        alternative_labels = {"A": "cluster_1", "B": "cluster_2", "C": "cluster_2"}
        report = validate_annual_reconstruction(
            self.panel,
            self.schedule,
            alternative_labels,
            grouping_id="hierarchical_fixture",
        )
        self.assertEqual(report["status"], "pass")
        self.assertLessEqual(report["maximum_absolute_error"], 1e-12)

    def test_nonfinite_value_in_an_active_security_fails(self):
        panel = self.panel.copy()
        panel.loc[pd.Timestamp("2021-01-05"), "A"] = np.nan
        report = validate_annual_reconstruction(panel, self.schedule, self.labels)
        self.assertEqual(report["status"], "fail")
        self.assertIn("nonfinite_active_return", {issue["code"] for issue in report["issues"]})

    def test_declared_annual_count_must_match_active_tickers(self):
        schedule = {"years": [dict(row) for row in self.schedule["years"]]}
        schedule["years"][1]["security_count"] = 3
        report = validate_annual_reconstruction(self.panel, schedule, self.labels)
        self.assertEqual(report["status"], "fail")
        self.assertIn(
            "annual_security_count_mismatch", {issue["code"] for issue in report["issues"]}
        )


if __name__ == "__main__":
    unittest.main()
