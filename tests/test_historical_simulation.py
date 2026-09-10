import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_historical_simulation import (
    ACTIVE_UNIVERSE_RULE,
    build_historical_simulation_outputs,
    daily_portfolio_simple_returns,
    validate_historical_protocol,
    validate_portfolio_audit_binding,
)
from scripts.pipeline_io import sha256_file
from scripts.research_methods import rolling_historical_var_es

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ARTIFACTS = (
    ROOT / "data/audit/historical_simulation_quality.json",
    ROOT / "data/processed/historical_simulation_risk_forecasts.parquet",
    ROOT / "data/processed/historical_simulation_windows.parquet",
)


def miniature_config() -> dict[str, object]:
    return {
        "forecast": {
            "var_confidence_levels": [0.95, 0.975, 0.99],
            "es_confidence_level": 0.975,
            "horizon_trading_days": 1,
            "training_window_calendar_years": 3,
            "minimum_training_observations": 700,
        },
        "historical_simulation": {
            "quantile_method": "inverted_empirical_cdf",
            "es_boundary_method": "fractional_order_statistic",
        },
        "portfolio": {
            "primary_return_scale": "simple",
            "primary_rebalancing": "daily",
            "membership_rebalancing": "annual",
            "realised_loss_definition": "negative_simple_portfolio_return",
        },
        "clustering": {
            "evaluation_start_year": 2020,
            "evaluation_end_year": 2025,
        },
    }


def miniature_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    dates = pd.bdate_range("2017-01-03", "2025-01-10", name="date")
    index = np.arange(len(dates), dtype=float)
    panel = pd.DataFrame(
        {
            "A": 0.01 * np.sin(index / 7.0),
            "B": 0.012 * np.cos(index / 11.0),
        },
        index=dates,
    )
    panel.iloc[0] = np.nan
    rows = []
    for year in range(2017, 2026):
        tickers = ["A", "B"] if year < 2020 else ["A"]
        annual_dates = dates[dates.year == year]
        rows.append(
            {
                "year": year,
                "rebalance_date": annual_dates.min().date().isoformat(),
                "security_count": len(tickers),
                "active_tickers": tickers,
            }
        )
    schedule = {"schema_version": 1, "rule": ACTIVE_UNIVERSE_RULE, "years": rows}
    return panel, schedule


class HistoricalSimulationMethodTests(unittest.TestCase):
    def test_protocol_rejects_nonfrozen_historical_quantiles(self):
        config = miniature_config()
        config["historical_simulation"]["quantile_method"] = "linear"
        with self.assertRaisesRegex(ValueError, "unsupported historical-simulation protocol"):
            validate_historical_protocol(config)

    def test_portfolio_uses_each_years_active_set(self):
        panel, schedule = miniature_inputs()
        returns, counts = daily_portfolio_simple_returns(panel, schedule)
        date_2019 = returns.index[returns.index.year == 2019][0]
        date_2020 = returns.index[returns.index.year == 2020][0]
        self.assertAlmostEqual(returns.loc[date_2019], panel.loc[date_2019].mean())
        self.assertAlmostEqual(returns.loc[date_2020], panel.loc[date_2020, "A"])
        self.assertEqual((counts.loc[date_2019], counts.loc[date_2020]), (2, 1))

    def test_missing_active_return_fails_instead_of_changing_weights(self):
        panel, schedule = miniature_inputs()
        panel.loc[pd.Timestamp("2021-01-04"), "A"] = np.nan
        with self.assertRaisesRegex(ValueError, "active security returns are incomplete"):
            daily_portfolio_simple_returns(panel, schedule)

    def test_active_schedule_cannot_reintroduce_a_removed_security(self):
        panel, schedule = miniature_inputs()
        row = schedule["years"][4]
        row["active_tickers"].append("B")
        row["security_count"] = 2
        with self.assertRaisesRegex(ValueError, "introduces later additions"):
            daily_portfolio_simple_returns(panel, schedule)


class HistoricalSimulationPipelineTests(unittest.TestCase):
    def test_outputs_use_exact_rolling_window_and_common_schema(self):
        panel, schedule = miniature_inputs()
        forecasts, windows, audit = build_historical_simulation_outputs(
            panel, schedule, miniature_config()
        )
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(set(forecasts["model_id"]), {"M0"})
        self.assertEqual(set(forecasts["grouping_id"]), {"none"})
        self.assertTrue(forecasts["seed_components"].isna().all())
        self.assertTrue(forecasts["copula_log_score"].isna().all())
        self.assertTrue((forecasts["forecast_status"] == "ok").all())

        forecast_schema = json.loads(
            (ROOT / "config/schemas/forecast_record.schema.json").read_text(encoding="utf-8")
        )
        window_schema = json.loads(
            (ROOT / "config/schemas/historical_simulation_window_record.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(forecasts.columns), set(forecast_schema["properties"]))
        self.assertEqual(set(windows.columns), set(window_schema["properties"]))
        self.assertFalse(forecast_schema["additionalProperties"])
        self.assertFalse(window_schema["additionalProperties"])
        self.assertEqual(set(window_schema["required"]), set(window_schema["properties"]))
        self.assertEqual(
            forecast_schema["allOf"][0]["if"]["properties"]["model_id"], {"const": "M0"}
        )
        self.assertEqual(
            forecast_schema["allOf"][0]["then"]["properties"]["seed_components"],
            {"type": "null"},
        )

        portfolio, _ = daily_portfolio_simple_returns(panel, schedule)
        first = forecasts.iloc[0]
        first_window = windows.iloc[0]
        date = pd.Timestamp(first["date"])
        expected = rolling_historical_var_es(portfolio.mul(-1), date, 0.95)
        self.assertEqual(first["var_95"], expected.var)
        self.assertEqual(first_window["requested_training_start"], date - pd.DateOffset(years=3))
        self.assertLess(first_window["training_end"], date)
        self.assertEqual(first_window["training_observations"], expected.observations)
        self.assertEqual(first["realised_loss"], -first["realised_simple_return"])

    def test_stale_portfolio_input_cannot_reuse_a_passed_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            returns = root / "returns.parquet"
            active = root / "active.json"
            returns.write_bytes(b"returns-v1")
            active.write_bytes(b"active-v1")
            audit = {
                "status": "pass",
                "portfolio_definition": "daily_rebalanced_equal_weight",
                "inputs": {
                    "simple_return_panel": {
                        "sha256": hashlib.sha256(returns.read_bytes()).hexdigest()
                    },
                    "active_universe": {"sha256": hashlib.sha256(active.read_bytes()).hexdigest()},
                },
            }
            validate_portfolio_audit_binding(
                audit,
                returns_path=returns,
                active_universe_path=active,
            )
            active.write_bytes(b"active-v2")
            with self.assertRaisesRegex(RuntimeError, "differs from the passed"):
                validate_portfolio_audit_binding(
                    audit,
                    returns_path=returns,
                    active_universe_path=active,
                )


@unittest.skipUnless(
    all(path.is_file() for path in HISTORICAL_ARTIFACTS),
    "requires locally generated historical-simulation artifacts",
)
class ProductionHistoricalSimulationArtifactTests(unittest.TestCase):
    def test_current_m0_gate_outputs_and_cross_model_identity_are_bound(self):
        audit_path, forecast_path, window_path = HISTORICAL_ARTIFACTS
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        forecasts = pd.read_parquet(forecast_path)
        windows = pd.read_parquet(window_path)
        gaussian = pd.read_parquet(ROOT / "data/processed/gaussian_risk_forecasts.parquet")
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(audit["issues"], [])
        self.assertEqual((len(forecasts), len(windows)), (1_508, 1_508))
        self.assertEqual(forecasts["date"].nunique(), 1_508)
        self.assertEqual(set(forecasts["model_id"]), {"M0"})
        self.assertEqual(
            audit["annual_forecast_counts"],
            {"2020": 253, "2021": 252, "2022": 251, "2023": 250, "2024": 252, "2025": 250},
        )
        self.assertTrue(forecasts["seed_components"].isna().all())
        self.assertTrue(forecasts["copula_log_score"].isna().all())
        self.assertGreaterEqual(windows["training_observations"].min(), 700)
        self.assertTrue((windows["training_end"] < windows["forecast_date"]).all())
        self.assertEqual(
            audit["outputs"]["historical_simulation_risk_forecasts"]["sha256"],
            sha256_file(forecast_path),
        )
        self.assertEqual(
            audit["outputs"]["historical_simulation_windows"]["sha256"],
            sha256_file(window_path),
        )
        reference = gaussian.loc[gaussian["model_id"] == "M1"].set_index("date")
        observed = forecasts.set_index("date")
        self.assertEqual(observed.index.tolist(), reference.index.tolist())
        np.testing.assert_allclose(
            observed["realised_simple_return"],
            reference["realised_simple_return"],
            rtol=0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
