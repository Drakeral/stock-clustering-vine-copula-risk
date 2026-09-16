from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_risk_models import DAILY_SCORE_COLUMNS
from scripts.evaluate_wba_dap_sensitivity import (
    ALL_MODEL_IDS,
    Scenario,
    compare_with_primary,
    construct_scenario_inputs,
    validate_sensitivity_protocol,
)
from scripts.pipeline_io import require_current_hash_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WbaDapProtocolTests(unittest.TestCase):
    def test_repository_protocol_matches_manual_action(self) -> None:
        import tomllib

        with (PROJECT_ROOT / "config/wba_dap_sensitivity.toml").open("rb") as handle:
            protocol = tomllib.load(handle)
        actions = json.loads(
            (PROJECT_ROOT / "config/manual_corporate_actions.json").read_text(encoding="utf-8")
        )

        action, scenarios = validate_sensitivity_protocol(protocol, actions)

        self.assertEqual(action["action_id"], "WBA_2025_SYCAMORE_CASH_DAP_ACQUISITION")
        self.assertEqual(
            [(scenario.scenario_id, scenario.dap_value_per_old_share) for scenario in scenarios],
            [("dap_zero", 0.0), ("dap_cap", 3.0)],
        )


class WbaDapConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.date = pd.Timestamp("2025-08-28")
        self.stock = pd.DataFrame(
            {"WBA": [0.0], "OTHER": [0.10]},
            index=pd.DatetimeIndex([self.date], name="date"),
        )
        self.security_daily = pd.DataFrame(
            [
                {
                    "date": self.date,
                    "research_ticker": "WBA",
                    "manual_action_id": "WBA_2025_SYCAMORE_CASH_DAP_ACQUISITION",
                    "manual_action_type": "cash_acquisition",
                    "is_synthetic": True,
                    "previous_close": 11.98,
                }
            ]
        )
        self.event = {
            "action_id": "WBA_2025_SYCAMORE_CASH_DAP_ACQUISITION",
            "event_type": "cash_acquisition",
            "research_ticker": "WBA",
            "effective_date": "2025-08-28",
            "cash_per_old_share": 11.45,
            "contingent_value_per_old_share": 0.53,
        }
        common = {
            "date": self.date,
            "year": 2025,
            "universe_variant": "security_primary",
            "sample_role": "evaluation",
            "group_size": 2,
            "portfolio_weight": 1.0,
            "simple_return": 0.05,
            "log_return": np.log1p(0.05),
        }
        self.core_groups = pd.DataFrame(
            [
                {**common, "grouping_id": "gics_sector", "group_id": "Consumer Staples"},
                {
                    **common,
                    "grouping_id": "hierarchical_cluster",
                    "group_id": "cluster_01",
                },
            ]
        )
        self.ml_groups = pd.DataFrame(
            [
                {
                    **common,
                    "grouping_id": "spectral_cluster",
                    "group_id": "spectral_01",
                },
                {
                    **common,
                    "grouping_id": "pca_kmeans_cluster",
                    "group_id": "pca_kmeans_01",
                },
            ]
        )
        core_rows = [
            {
                "ticker": "WBA",
                "gics_sector": "Consumer Staples",
                "hierarchical_cluster": "cluster_01",
            },
            {
                "ticker": "OTHER",
                "gics_sector": "Consumer Staples",
                "hierarchical_cluster": "cluster_01",
            },
        ]
        ml_rows = [
            {
                **row,
                "spectral_cluster": "spectral_01",
                "pca_kmeans_cluster": "pca_kmeans_01",
            }
            for row in core_rows
        ]
        self.core_assignments = {"years": [{"year": 2025, "assignments": core_rows}]}
        self.ml_assignments = {"years": [{"year": 2025, "assignments": ml_rows}]}

    def test_one_stock_cell_propagates_through_all_fixed_groups(self) -> None:
        scenario = Scenario("dap_zero", 0.0)

        stock, groups, audit = construct_scenario_inputs(
            self.stock,
            self.security_daily,
            self.core_groups,
            self.ml_groups,
            self.core_assignments,
            self.ml_assignments,
            scenario,
            self.event,
            tolerance=1e-12,
        )

        expected_wba = 11.45 / 11.98 - 1.0
        expected_group = 0.05 + expected_wba / 2.0
        self.assertAlmostEqual(stock.at[self.date, "WBA"], expected_wba)
        np.testing.assert_allclose(groups["simple_return"], expected_group)
        np.testing.assert_allclose(groups["log_return"], np.log1p(expected_group))
        self.assertEqual(audit["changed_stock_return_cell_count"], 1)
        self.assertEqual(len(audit["affected_groups"]), 4)
        self.assertLessEqual(audit["maximum_group_portfolio_identity_error"], 1e-12)
        pd.testing.assert_frame_equal(
            self.stock,
            pd.DataFrame(
                {"WBA": [0.0], "OTHER": [0.10]},
                index=pd.DatetimeIndex([self.date], name="date"),
            ),
        )


class WbaDapComparisonTests(unittest.TestCase):
    @staticmethod
    def _scores() -> pd.DataFrame:
        dates = pd.to_datetime(["2025-08-27", "2025-08-28", "2025-08-29"])
        rows = []
        for model_id in ALL_MODEL_IDS:
            grouping = "none" if model_id == "M0" else "test"
            for date in dates:
                rows.append(
                    {
                        "date": date,
                        "year": 2025,
                        "model_id": model_id,
                        "grouping_id": grouping,
                        "realised_loss": 0.01,
                        "var_95": 0.02,
                        "var_975": 0.025,
                        "var_99": 0.03,
                        "es_975": 0.035,
                        "quantile_loss_95": 0.001,
                        "quantile_loss_99": 0.0002,
                        "fz0_975": -2.0,
                        "exception_95": False,
                        "exception_975": False,
                        "exception_99": False,
                        "copula_log_score": np.nan if model_id == "M0" else 0.0,
                        "forecast_status": "ok",
                        "margin_fallback_count": 0,
                        "whole_vine_fallback": False,
                    }
                )
        return pd.DataFrame(rows).loc[:, DAILY_SCORE_COLUMNS]

    def test_pre_event_forecasts_are_held_identical(self) -> None:
        primary = self._scores()
        scenario = primary.copy()
        event = pd.Timestamp("2025-08-28")
        scenario.loc[scenario["date"].eq(event), "realised_loss"] += 0.004
        scenario.loc[scenario["date"].eq(event), "quantile_loss_95"] += 0.0002
        scenario.loc[scenario["date"].gt(event), "var_95"] += 0.001

        comparison, identity = compare_with_primary(
            scenario, primary, event_date=event, tolerance=1e-12
        )

        self.assertEqual(len(comparison), 9)
        self.assertAlmostEqual(identity["common_event_realised_loss_delta"], 0.004)
        self.assertEqual(comparison[0]["changed_date_count_var_95"], 1)

    def test_pre_event_forecast_change_is_rejected(self) -> None:
        primary = self._scores()
        scenario = primary.copy()
        scenario.loc[scenario["date"].eq(pd.Timestamp("2025-08-27")), "var_95"] += 0.001

        with self.assertRaisesRegex(ValueError, "before information"):
            compare_with_primary(
                scenario,
                primary,
                event_date=pd.Timestamp("2025-08-28"),
                tolerance=1e-12,
            )


@unittest.skipUnless(
    (PROJECT_ROOT / "data/audit/wba_dap_sensitivity.json").is_file(),
    "requires locally generated WBA DAP sensitivity artifacts",
)
class ProductionWbaDapSensitivityTests(unittest.TestCase):
    def test_audit_binds_complete_passed_scenarios(self) -> None:
        audit_path = PROJECT_ROOT / "data/audit/wba_dap_sensitivity.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

        require_current_hash_records(audit, source_name="WBA DAP sensitivity audit")
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(
            {scenario["scenario_id"] for scenario in audit["scenarios"]},
            {"dap_zero", "dap_cap"},
        )
        for scenario in audit["scenarios"]:
            self.assertEqual(scenario["construction"]["changed_stock_return_cell_count"], 1)
            self.assertLessEqual(
                scenario["construction"]["maximum_group_portfolio_identity_error"], 1e-12
            )
            self.assertFalse(scenario["construction"]["annual_assignments_reestimated"])
            for stage in ("historical", "marginal", "gaussian", "vine"):
                self.assertEqual(scenario["quality"][stage]["status"], "pass")
            score_path = PROJECT_ROOT / scenario["outputs"]["daily_scores"]["path"]
            scores = pd.read_parquet(score_path)
            self.assertEqual(len(scores), 9 * 1508)
            self.assertEqual(set(scores["model_id"]), set(ALL_MODEL_IDS))
            self.assertTrue(scores.groupby("model_id")["date"].nunique().eq(1508).all())


if __name__ == "__main__":
    unittest.main()
