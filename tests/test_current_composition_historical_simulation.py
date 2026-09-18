import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from scripts.build_historical_simulation import (
    ACTIVE_UNIVERSE_RULE,
    FORECAST_COLUMNS,
    build_historical_simulation_outputs,
)
from scripts.evaluate_current_composition_historical_simulation import (
    ROBUSTNESS_MODEL,
    build_current_composition_forecasts,
    combine_historical_forecasts,
    evaluate_forecasts,
    validate_robustness_protocol,
)
from scripts.evaluate_risk_models import DAILY_SCORE_COLUMNS
from scripts.pipeline_io import sha256_file
from scripts.research_methods import rolling_historical_var_es

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    ROOT / "data/audit/current_composition_hs_robustness.json",
    ROOT / "data/processed/current_composition_hs_robustness/forecasts.parquet",
    ROOT / "data/processed/current_composition_hs_robustness/windows.parquet",
    ROOT / "data/processed/current_composition_hs_robustness/daily_scores.parquet",
)


def synthetic_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    dates = pd.bdate_range("2017-01-03", "2025-12-31", name="date")
    index = np.arange(len(dates), dtype=float)
    panel = pd.DataFrame(
        {
            "A": 0.012 * np.sin(index / 11.0),
            "B": 0.018 * np.cos(index / 7.0),
        },
        index=dates,
    )
    panel.iloc[0] = np.nan
    years = []
    for year in range(2017, 2026):
        active = ["A", "B"] if year < 2020 else ["A"]
        annual_dates = dates[dates.year == year]
        years.append(
            {
                "year": year,
                "rebalance_date": annual_dates.min().date().isoformat(),
                "security_count": len(active),
                "active_tickers": active,
            }
        )
    return panel, {"schema_version": 1, "rule": ACTIVE_UNIVERSE_RULE, "years": years}


class CurrentCompositionMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "config/model_config.toml").open("rb") as handle:
            cls.model_config = tomllib.load(handle)
        with (ROOT / "config/current_composition_hs_robustness.toml").open("rb") as handle:
            cls.robustness_config = tomllib.load(handle)

    def _outputs(self):
        panel, schedule = synthetic_inputs()
        primary, _, _ = build_historical_simulation_outputs(
            panel,
            schedule,
            self.model_config,
        )
        forecasts, windows, audit = build_current_composition_forecasts(
            panel,
            schedule,
            primary,
            self.model_config,
            self.robustness_config,
        )
        return panel, primary, forecasts, windows, audit

    def test_protocol_is_frozen_and_bound_to_core_estimators(self):
        sections = validate_robustness_protocol(self.robustness_config, self.model_config)
        self.assertEqual(sections["models"]["robustness_model_id"], ROBUSTNESS_MODEL)
        changed = copy.deepcopy(self.robustness_config)
        changed["dm"]["family_size"] = 6
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            validate_robustness_protocol(changed, self.model_config)

    def test_backcast_uses_evaluation_year_active_set_and_canonical_realised_return(self):
        panel, primary, forecasts, windows, audit = self._outputs()
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(set(forecasts["model_id"]), {ROBUSTNESS_MODEL})
        first = forecasts.iloc[0]
        date = pd.Timestamp(first["date"])
        expected = rolling_historical_var_es(-panel["A"].dropna(), date, 0.95)
        self.assertEqual(first["var_95"], expected.var)
        self.assertEqual(first["realised_simple_return"], panel.loc[date, "A"])
        np.testing.assert_allclose(
            forecasts["realised_loss"],
            primary["realised_loss"],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue((windows["training_end"] < windows["forecast_date"]).all())
        self.assertTrue(
            np.any(np.abs(forecasts["var_95"].to_numpy() - primary["var_95"].to_numpy()) > 0)
        )

    def test_missing_backcast_return_fails_instead_of_reweighting(self):
        panel, schedule = synthetic_inputs()
        primary, _, _ = build_historical_simulation_outputs(
            panel,
            schedule,
            self.model_config,
        )
        panel.loc[pd.Timestamp("2018-01-02"), "A"] = np.nan
        with self.assertRaisesRegex(ValueError, "active returns are incomplete"):
            build_current_composition_forecasts(
                panel,
                schedule,
                primary,
                self.model_config,
                self.robustness_config,
            )

    def test_evaluation_is_matched_and_uses_separate_test_families(self):
        _, primary, forecasts, _, _ = self._outputs()
        scores, audit = evaluate_forecasts(primary, forecasts, self.robustness_config)
        self.assertEqual(set(scores["model_id"]), {"M0", "M0_CC"})
        self.assertEqual(len(scores), 2 * len(primary))
        self.assertEqual(len(audit["diebold_mariano_comparisons"]), 3)
        self.assertEqual(len(audit["full_period_calibration"]), 4)
        self.assertFalse(audit["interpretation"]["core_H1_H2_H3_revised"])
        self.assertGreater(audit["forecast_validation"]["forecast_difference_date_count"], 0)

        corrupted = forecasts.copy()
        corrupted.loc[0, "realised_loss"] += 0.01
        corrupted.loc[0, "realised_simple_return"] -= 0.01
        with self.assertRaisesRegex(ValueError, "realised losses differ"):
            combine_historical_forecasts(primary, corrupted, tolerance=1e-12)

        corrupted = forecasts.copy()
        corrupted.loc[0, "var_99"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite risk values"):
            combine_historical_forecasts(primary, corrupted, tolerance=1e-12)

    def test_shared_schemas_cover_current_composition_outputs(self):
        forecast_schema = json.loads(
            (ROOT / "config/schemas/forecast_record.schema.json").read_text()
        )
        score_schema = json.loads(
            (ROOT / "config/schemas/risk_evaluation_daily_record.schema.json").read_text()
        )
        self.assertIn(ROBUSTNESS_MODEL, forecast_schema["properties"]["model_id"]["enum"])
        self.assertIn(ROBUSTNESS_MODEL, score_schema["properties"]["model_id"]["enum"])
        self.assertEqual(set(forecast_schema["properties"]), set(FORECAST_COLUMNS))
        self.assertEqual(set(score_schema["properties"]), set(DAILY_SCORE_COLUMNS))


@unittest.skipUnless(
    all(path.is_file() for path in ARTIFACTS), "requires current-composition artifacts"
)
class ProductionCurrentCompositionTests(unittest.TestCase):
    def test_artifacts_are_complete_hash_bound_and_exploratory(self):
        audit_path, forecast_path, window_path, score_path = ARTIFACTS
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        forecasts = pd.read_parquet(forecast_path)
        windows = pd.read_parquet(window_path)
        scores = pd.read_parquet(score_path)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(audit["analysis_role"], "exploratory_robustness")
        self.assertEqual((len(forecasts), len(windows), len(scores)), (1_508, 1_508, 3_016))
        self.assertEqual(set(forecasts["model_id"]), {ROBUSTNESS_MODEL})
        self.assertEqual(set(scores["model_id"]), {"M0", ROBUSTNESS_MODEL})
        self.assertTrue((windows["training_end"] < windows["forecast_date"]).all())
        self.assertFalse(audit["interpretation"]["core_H1_H2_H3_revised"])
        self.assertEqual(audit["interpretation"]["evidence"], "partial_realised_history_favoured")
        self.assertEqual(audit["interpretation"]["tests_favouring_realised_history_after_holm"], 1)
        self.assertEqual(
            [row["significant_after_holm"] for row in audit["diebold_mariano_comparisons"]],
            [False, True, False],
        )
        for key, path in (
            ("current_composition_forecasts", forecast_path),
            ("current_composition_windows", window_path),
            ("daily_scores", score_path),
        ):
            self.assertEqual(audit["outputs"][key]["sha256"], sha256_file(path))


if __name__ == "__main__":
    unittest.main()
