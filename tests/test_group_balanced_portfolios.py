from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_group_balanced_portfolios import (
    GROUP_RETURN_COLUMNS,
    PORTFOLIO_RETURN_COLUMNS,
    _annual_sample,
    group_balanced_buy_and_hold,
    validate_group_balanced_protocol,
)
from scripts.pipeline_io import require_current_hash_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GroupBalancedPortfolioMethodTests(unittest.TestCase):
    def test_unequal_groups_start_equal_drift_and_reset_annually(self) -> None:
        dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2021-01-04"])
        returns = pd.DataFrame(
            {
                "A": [0.10, -0.05, 0.02],
                "B": [0.00, 0.02, 0.01],
                "C": [0.00, 0.04, -0.03],
            },
            index=dates,
        )

        result = group_balanced_buy_and_hold(
            returns,
            {"A": "large", "B": "large", "C": "singleton"},
        )

        self.assertAlmostEqual(result.portfolio_simple_returns.iloc[0], 0.025)
        np.testing.assert_allclose(result.group_pre_return_weights.iloc[0], [0.5, 0.5])
        expected_large = (0.25 * 1.10 + 0.25) / 1.025
        self.assertAlmostEqual(
            result.group_pre_return_weights.loc[dates[1], "large"], expected_large
        )
        np.testing.assert_allclose(result.group_pre_return_weights.iloc[2], [0.5, 0.5])
        self.assertEqual(
            [date.date().isoformat() for date in result.reset_dates],
            ["2020-01-02", "2021-01-04"],
        )
        reconstructed = (result.group_pre_return_weights * result.group_simple_returns).sum(axis=1)
        np.testing.assert_allclose(reconstructed, result.portfolio_simple_returns, atol=1e-12)
        self.assertLessEqual(result.maximum_identity_error, 1e-12)

    def test_protocol_is_frozen_before_results(self) -> None:
        import tomllib

        with (PROJECT_ROOT / "config/group_balanced_robustness.toml").open("rb") as handle:
            config = tomllib.load(handle)

        specs = validate_group_balanced_protocol(config)

        self.assertEqual(
            [spec.portfolio_id for spec in specs],
            [
                "gics_balanced",
                "hierarchical_balanced",
            ],
        )
        self.assertEqual(config["evaluation"]["dm_family_size"], 6)
        self.assertEqual(config["evaluation"]["calibration_family_size"], 24)

    def test_output_schemas_bind_all_produced_fields(self) -> None:
        cases = {
            "group_balanced_group_return_record.schema.json": GROUP_RETURN_COLUMNS,
            "group_balanced_portfolio_return_record.schema.json": PORTFOLIO_RETURN_COLUMNS,
        }
        for filename, columns in cases.items():
            with self.subTest(schema=filename):
                schema = json.loads(
                    (PROJECT_ROOT / "config/schemas" / filename).read_text(encoding="utf-8")
                )
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(set(schema["required"]), set(columns))
                self.assertEqual(set(schema["properties"]), set(columns))

    def test_incomplete_or_total_loss_inputs_fail_closed(self) -> None:
        frame = pd.DataFrame(
            {"A": [0.0, np.nan], "B": [0.0, 0.0]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            group_balanced_buy_and_hold(frame, {"A": "one", "B": "two"})
        frame.loc[frame.index[1], "A"] = -1.0
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            group_balanced_buy_and_hold(frame, {"A": "one", "B": "two"})

    def test_invalid_tolerance_columns_and_labels_fail_closed(self) -> None:
        frame = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=pd.to_datetime(["2020-01-02"]))
        with self.assertRaisesRegex(ValueError, "tolerance"):
            group_balanced_buy_and_hold(frame, {"A": "one", "B": "two"}, tolerance=0)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            group_balanced_buy_and_hold(frame, {"A": "one", "B": " "})
        frame.columns = [1, 2]
        with self.assertRaisesRegex(ValueError, "unique strings"):
            group_balanced_buy_and_hold(frame, {1: "one", 2: "two"})

    def test_annual_sample_requires_active_columns_and_exact_rebalance_date(self) -> None:
        panel = pd.DataFrame(
            {"A": [0.0, 0.0]},
            index=pd.to_datetime(["2019-12-31", "2020-01-03"]),
        )
        annual = {
            "year": 2020,
            "active_security_count": 1,
            "group_count": 1,
            "training_start_inclusive": "2019-12-01",
            "rebalance_date": "2020-01-02",
            "assignments": [{"ticker": "A"}],
        }
        with self.assertRaisesRegex(ValueError, "misaligned"):
            _annual_sample(panel, annual)
        annual["assignments"] = [{"ticker": "B"}]
        with self.assertRaisesRegex(ValueError, "missing active securities"):
            _annual_sample(panel, annual)


@unittest.skipUnless(
    (PROJECT_ROOT / "data/audit/group_balanced_portfolio_construction.json").is_file(),
    "requires locally generated group-balanced portfolio artifacts",
)
class ProductionGroupBalancedPortfolioTests(unittest.TestCase):
    def test_artifacts_are_complete_and_hash_bound(self) -> None:
        audit_path = PROJECT_ROOT / "data/audit/group_balanced_portfolio_construction.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

        require_current_hash_records(audit, source_name="group-balanced construction audit")
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(len(audit["annual_diagnostics"]), 12)
        self.assertLessEqual(audit["maximum_portfolio_identity_error"], 1e-12)
        group_path = PROJECT_ROOT / audit["outputs"]["group_returns"]["path"]
        portfolio_path = PROJECT_ROOT / audit["outputs"]["portfolio_returns"]["path"]
        groups = pd.read_parquet(group_path)
        portfolios = pd.read_parquet(portfolio_path)
        self.assertEqual(set(groups["grouping_id"]), {"gics_balanced", "hierarchical_balanced"})
        self.assertEqual(
            set(portfolios["portfolio_id"]), {"gics_balanced", "hierarchical_balanced"}
        )
        self.assertTrue(
            portfolios.loc[portfolios["sample_role"].eq("evaluation")]
            .groupby(["year", "portfolio_id"])["date"]
            .nunique()
            .gt(0)
            .all()
        )


if __name__ == "__main__":
    unittest.main()
