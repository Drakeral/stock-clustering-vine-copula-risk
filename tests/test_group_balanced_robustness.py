from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_group_balanced_portfolios import validate_group_balanced_protocol
from scripts.build_historical_simulation import FORECAST_COLUMNS
from scripts.evaluate_group_balanced_robustness import (
    combine_forecasts,
    evaluate_forecasts,
    validate_risk_protocol,
)
from scripts.evaluate_risk_models import DAILY_SCORE_COLUMNS
from scripts.pipeline_io import require_current_hash_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _forecast_frame(model_ids: list[tuple[str, str, str]], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for model_id, grouping_id, portfolio_id in model_ids:
        portfolio_shift = 0.001 if portfolio_id == "hierarchical_balanced" else 0.0
        realised = 0.006 * np.sin(np.arange(len(dates)) / 6) + portfolio_shift
        model_phase = sum(ord(character) for character in model_id) % 17
        for date_number, (date, realised_return) in enumerate(zip(dates, realised, strict=True)):
            variation = 0.0002 * np.sin((date_number + model_phase) / 5)
            rows.append(
                {
                    "date": date,
                    "model_id": model_id,
                    "grouping_id": grouping_id,
                    "refit_id": f"{model_id}:{date.date()}",
                    "var_95": 0.014 + variation,
                    "var_975": 0.018 + variation,
                    "var_99": 0.022 + variation,
                    "es_975": 0.026 + variation,
                    "realised_simple_return": realised_return,
                    "realised_loss": -realised_return,
                    "seed_components": None if model_id.endswith("HS") else [5110, 2020, 1],
                    "margin_fallback_count": 0,
                    "whole_vine_fallback": False,
                    "copula_log_score": None if model_id.endswith("HS") else -0.1,
                    "forecast_status": "ok",
                }
            )
    return pd.DataFrame(rows).loc[:, FORECAST_COLUMNS]


class GroupBalancedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tomllib

        with (PROJECT_ROOT / "config/group_balanced_robustness.toml").open("rb") as handle:
            cls.robustness_config = tomllib.load(handle)
        with (PROJECT_ROOT / "config/model_config.toml").open("rb") as handle:
            cls.model_config = tomllib.load(handle)
        cls.specs = validate_group_balanced_protocol(cls.robustness_config)
        cls.dates = pd.bdate_range("2020-01-02", periods=200)
        cls.historical = _forecast_frame(
            [
                ("B_GICS_HS", "gics_balanced", "gics_balanced"),
                ("B_HIER_HS", "hierarchical_balanced", "hierarchical_balanced"),
            ],
            cls.dates,
        )
        cls.gaussian = _forecast_frame(
            [
                ("B_GICS_GAUSSIAN", "gics_balanced", "gics_balanced"),
                (
                    "B_HIER_GAUSSIAN",
                    "hierarchical_balanced",
                    "hierarchical_balanced",
                ),
            ],
            cls.dates,
        )
        cls.vine = _forecast_frame(
            [
                ("B_GICS_VINE", "gics_balanced", "gics_balanced"),
                ("B_HIER_VINE", "hierarchical_balanced", "hierarchical_balanced"),
            ],
            cls.dates,
        )

    def test_protocol_is_bound_to_core_settings(self) -> None:
        specs = validate_risk_protocol(self.robustness_config, self.model_config)
        self.assertEqual(specs, self.specs)

    def test_shared_schemas_cover_all_group_balanced_models(self) -> None:
        forecast_schema = json.loads(
            (PROJECT_ROOT / "config/schemas/forecast_record.schema.json").read_text()
        )
        score_schema = json.loads(
            (PROJECT_ROOT / "config/schemas/risk_evaluation_daily_record.schema.json").read_text()
        )
        expected_models = {
            "B_GICS_HS",
            "B_GICS_GAUSSIAN",
            "B_GICS_VINE",
            "B_HIER_HS",
            "B_HIER_GAUSSIAN",
            "B_HIER_VINE",
        }
        for schema, columns in (
            (forecast_schema, FORECAST_COLUMNS),
            (score_schema, DAILY_SCORE_COLUMNS),
        ):
            self.assertEqual(set(schema["required"]), set(columns))
            self.assertEqual(set(schema["properties"]), set(columns))
            self.assertTrue(expected_models.issubset(schema["properties"]["model_id"]["enum"]))
        marginal_groupings = {
            "gics_balanced",
            "hierarchical_balanced",
        }
        for filename in (
            "marginal_refit_record.schema.json",
            "marginal_daily_record.schema.json",
            "marginal_training_pit_record.schema.json",
        ):
            schema = json.loads((PROJECT_ROOT / "config/schemas" / filename).read_text())
            self.assertTrue(
                marginal_groupings.issubset(schema["properties"]["grouping_id"]["enum"])
            )
        gaussian_schema = json.loads(
            (PROJECT_ROOT / "config/schemas/gaussian_copula_refit_record.schema.json").read_text()
        )
        vine_schema = json.loads(
            (PROJECT_ROOT / "config/schemas/vine_copula_refit_record.schema.json").read_text()
        )
        self.assertTrue(
            {"B_GICS_GAUSSIAN", "B_HIER_GAUSSIAN"}.issubset(
                gaussian_schema["properties"]["model_id"]["enum"]
            )
        )
        self.assertTrue(
            {"B_GICS_VINE", "B_HIER_VINE"}.issubset(vine_schema["properties"]["model_id"]["enum"])
        )

    def test_forecasts_are_matched_only_within_portfolio(self) -> None:
        combined, validation = combine_forecasts(
            self.historical,
            self.gaussian,
            self.vine,
            self.specs,
            tolerance=1e-12,
        )
        self.assertEqual(len(combined), 6 * len(self.dates))
        self.assertLessEqual(
            validation["maximum_within_portfolio_realised_loss_identity_error"], 1e-12
        )
        altered = self.vine.copy()
        target = altered["model_id"].eq("B_HIER_VINE")
        altered.loc[target, "realised_loss"] += 0.001
        altered.loc[target, "realised_simple_return"] -= 0.001
        with self.assertRaisesRegex(ValueError, "differs"):
            combine_forecasts(
                self.historical,
                self.gaussian,
                altered,
                self.specs,
                tolerance=1e-12,
            )

    def test_complete_evaluation_keeps_portfolio_rankings_separate(self) -> None:
        scores, audit = evaluate_forecasts(
            self.historical,
            self.gaussian,
            self.vine,
            self.specs,
            self.robustness_config,
            self.model_config,
            {
                "gics_balanced": {"eligible_to_be_declared_best": True},
                "hierarchical_balanced": {"eligible_to_be_declared_best": True},
            },
        )
        self.assertEqual(len(scores), 6 * len(self.dates))
        self.assertEqual(len(audit["full_period_calibration"]), 12)
        self.assertEqual(len(audit["diebold_mariano_comparisons"]), 6)
        self.assertEqual(len(audit["portfolio_conclusions"]), 2)
        self.assertFalse(audit["interpretation"]["cross_portfolio_ranking_performed"])
        json.dumps(audit, allow_nan=False)
        summaries = pd.DataFrame(audit["model_summaries"])
        self.assertTrue(
            summaries.loc[summaries["dependence"].eq("historical_simulation")][
                "mean_copula_log_score"
            ]
            .isna()
            .all()
        )
        self.assertTrue(
            summaries.groupby("portfolio_id")["within_portfolio_overall_rank"].min().eq(1.0).all()
        )


@unittest.skipUnless(
    (PROJECT_ROOT / "data/audit/group_balanced_robustness.json").is_file(),
    "requires locally generated group-balanced risk artifacts",
)
class ProductionGroupBalancedRobustnessTests(unittest.TestCase):
    def test_complete_risk_artifacts_are_passed_and_bound(self) -> None:
        audit_path = PROJECT_ROOT / "data/audit/group_balanced_robustness.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

        require_current_hash_records(audit, source_name="group-balanced robustness audit")
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(audit["quality"]["marginal"]["refit_count"], 1584)
        self.assertEqual(
            audit["quality"]["marginal"]["portfolio_weight_rule"],
            ("annual_equal_group_buy_and_hold"),
        )
        self.assertEqual(len(audit["evaluation"]["diebold_mariano_comparisons"]), 6)
        self.assertEqual(len(audit["evaluation"]["full_period_calibration"]), 12)
        self.assertFalse(audit["evaluation"]["interpretation"]["core_H1_H2_H3_revised"])
        scores_path = PROJECT_ROOT / audit["outputs"]["daily_scores"]["path"]
        scores = pd.read_parquet(scores_path)
        score_schema = json.loads(
            (PROJECT_ROOT / "config/schemas/risk_evaluation_daily_record.schema.json").read_text()
        )
        self.assertEqual(set(scores.columns), set(score_schema["properties"]))
        self.assertEqual(len(scores), 6 * 1508)
        self.assertTrue(scores.groupby("model_id")["date"].nunique().eq(1508).all())


if __name__ == "__main__":
    unittest.main()
