import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_gaussian_copula import (
    _array_sha256,
    _ewma_empirical_innovations,
    build_gaussian_outputs,
    empirical_risk_levels,
    fit_gaussian_copula,
    gaussian_copula_log_density,
    gaussian_dependence_uniforms,
)
from scripts.research_methods import common_uniforms, empirical_var_es

ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_ARTIFACTS = (
    ROOT / "data/audit/gaussian_copula_quality.json",
    ROOT / "data/processed/gaussian_copula_refits.parquet",
    ROOT / "data/processed/gaussian_risk_forecasts.parquet",
    ROOT / "data/manifests/simulation_seed_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def miniature_config() -> dict[str, object]:
    return {
        "gaussian": {
            "estimator": "pearson_correlation_of_normal_scores",
            "repair_method": "symmetric_eigenvalue_floor_then_unit_diagonal_rescale",
            "factorization": "cholesky",
            "log_score": "gaussian_copula_log_density",
            "eigenvalue_floor": 1e-8,
            "repair_tolerance": 1e-12,
            "maximum_condition_number": 1e10,
        },
        "simulation": {
            "draws_per_month": 2_000,
            "dimension": 2,
            "bit_generator": "PCG64DXSM",
            "base_seed": 5110,
            "seed_components": ["base_seed", "year", "month"],
            "common_random_numbers": True,
            "student_t_standardization": "unit_variance_sqrt_df_minus_2_over_df",
            "ewma_inverse_cdf": "inverted_empirical_cdf",
        },
        "forecast": {
            "var_confidence_levels": [0.95, 0.975, 0.99],
            "es_confidence_level": 0.975,
            "horizon_trading_days": 1,
        },
        "marginal": {
            "pit_clip_lower": 1e-6,
            "pit_clip_upper": 0.999999,
            "estimation_return_scale": 100.0,
            "ewma_lambda": 0.94,
        },
        "portfolio": {
            "primary_return_scale": "simple",
            "primary_rebalancing": "daily",
            "group_weighting": "group_size",
            "realised_loss_definition": "negative_simple_portfolio_return",
            "reconstruction_tolerance": 1e-12,
        },
        "vine": {"dimension": 2},
    }


class GaussianMethodTests(unittest.TestCase):
    def test_fit_is_symmetric_positive_definite_and_repairs_singularity(self):
        pits = np.column_stack(
            [
                np.linspace(0.05, 0.95, 30),
                np.linspace(0.05, 0.95, 30),
                np.linspace(0.95, 0.05, 30),
            ]
        )
        fit = fit_gaussian_copula(pits)
        np.testing.assert_allclose(fit.correlation, fit.correlation.T, atol=1e-15)
        np.testing.assert_allclose(np.diag(fit.correlation), 1.0, atol=1e-15)
        self.assertGreater(np.linalg.eigvalsh(fit.correlation).min(), 0)
        self.assertTrue(fit.correlation_repaired)
        self.assertGreater(fit.maximum_absolute_adjustment, 0)

    def test_dependence_simulation_is_seeded_and_identity_log_density_is_zero(self):
        first = common_uniforms(2020, 1, draws=5_000, dimension=2)
        second = common_uniforms(2020, 1, draws=5_000, dimension=2)
        np.testing.assert_array_equal(first, second)
        correlation = np.asarray([[1.0, 0.7], [0.7, 1.0]])
        dependent = gaussian_dependence_uniforms(first, correlation)
        self.assertGreater(np.corrcoef(dependent, rowvar=False)[0, 1], 0.6)
        density = gaussian_copula_log_density(first[:10], np.eye(2))
        np.testing.assert_allclose(density, 0.0, atol=1e-12)

    def test_partitioned_risk_matches_frozen_reference_implementation(self):
        losses = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 8.0])
        risks = empirical_risk_levels(losses, [0.5, 0.75, 0.9])
        for confidence, (value_at_risk, expected_shortfall) in risks.items():
            reference = empirical_var_es(losses, confidence)
            self.assertEqual(value_at_risk, reference.var)
            self.assertAlmostEqual(expected_shortfall, reference.es)

    def test_ewma_reference_reconstruction_uses_the_complete_training_sample(self):
        dates = pd.bdate_range("2019-12-23", periods=5)
        log_returns = np.asarray([0.01, -0.02, 0.005, 0.0, 0.015])
        scaled = 100.0 * log_returns
        mean = float(scaled.mean())
        residuals = scaled - mean
        variance = float(np.var(residuals, ddof=1))
        expected = []
        for residual in residuals:
            expected.append(residual / np.sqrt(variance))
            variance = 0.94 * variance + 0.06 * residual**2
        frame = pd.DataFrame(
            {
                "date": dates,
                "year": 2020,
                "universe_variant": "security_primary",
                "grouping_id": "gics_sector",
                "group_id": "A",
                "log_return": log_returns,
            }
        )
        refit = pd.Series(
            {
                "refit_id": "fixture",
                "refit_date": pd.Timestamp("2020-01-02"),
                "requested_training_start": pd.Timestamp("2017-01-02"),
                "year": 2020,
                "universe_variant": "security_primary",
                "grouping_id": "gics_sector",
                "group_id": "A",
                "training_observations": 5,
                "mean_constant_scaled": mean,
                "first_forecast_variance_scaled": variance,
            }
        )
        observed = _ewma_empirical_innovations(frame, refit, scale=100.0, smoothing=0.94)
        np.testing.assert_allclose(observed, expected)


class GaussianPipelineTests(unittest.TestCase):
    def _frames(self):
        training_dates = pd.bdate_range("2019-12-16", periods=10)
        evaluation_dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
        training_rows = []
        refit_rows = []
        daily_rows = []
        return_rows = []
        specifications = {
            "gics_sector": ["A", "B"],
            "hierarchical_cluster": ["H1", "H2"],
        }
        realised = [np.log1p(0.01), np.log1p(-0.01)]
        for grouping_id, group_ids in specifications.items():
            copula_refit_id = f"2020-01:security_primary:{grouping_id}"
            for group_index, group_id in enumerate(group_ids):
                marginal_refit_id = f"{copula_refit_id}:{group_id}"
                for index, date in enumerate(training_dates):
                    pit = 0.05 + 0.9 * (index + group_index) / 11
                    training_rows.append(
                        {
                            "training_date": date,
                            "year": 2020,
                            "month": 1,
                            "refit_date": pd.Timestamp("2020-01-02"),
                            "universe_variant": "security_primary",
                            "grouping_id": grouping_id,
                            "group_id": group_id,
                            "copula_refit_id": copula_refit_id,
                            "pit": pit,
                        }
                    )
                refit_rows.append(
                    {
                        "refit_id": marginal_refit_id,
                        "refit_date": pd.Timestamp("2020-01-02"),
                        "year": 2020,
                        "month": 1,
                        "universe_variant": "security_primary",
                        "grouping_id": grouping_id,
                        "group_id": group_id,
                        "requested_training_start": pd.Timestamp("2017-01-02"),
                        "training_observations": 10,
                        "selected_method": "ar_garch_t",
                        "fallback_level": 0,
                        "mean_constant_scaled": 0.0,
                        "student_t_df": 8.0,
                        "first_forecast_variance_scaled": 1.0,
                    }
                )
                for date_index, date in enumerate(evaluation_dates):
                    daily_rows.append(
                        {
                            "date": date,
                            "year": 2020,
                            "month": 1,
                            "universe_variant": "security_primary",
                            "grouping_id": grouping_id,
                            "group_id": group_id,
                            "portfolio_weight": 0.5,
                            "conditional_mean_log_return": 0.0,
                            "conditional_volatility_log_return": 0.01,
                            "realised_log_return": realised[(group_index + date_index) % 2],
                            "pit": 0.4 + 0.1 * group_index,
                        }
                    )
                for date in training_dates:
                    return_rows.append(
                        {
                            "date": date,
                            "year": 2020,
                            "universe_variant": "security_primary",
                            "grouping_id": grouping_id,
                            "group_id": group_id,
                            "log_return": 0.0,
                        }
                    )
        return (
            pd.DataFrame(training_rows),
            pd.DataFrame(refit_rows),
            pd.DataFrame(daily_rows),
            pd.DataFrame(return_rows),
        )

    def test_pipeline_builds_matched_m1_m3_forecasts_with_common_random_numbers(self):
        refits, forecasts, seeds, audit = build_gaussian_outputs(
            *self._frames(), miniature_config(), progress_every=0
        )
        self.assertEqual(audit["status"], "pass")
        self.assertEqual((len(refits), len(forecasts)), (2, 4))
        self.assertEqual(set(refits["model_id"]), {"M1", "M3"})
        self.assertEqual(refits["base_uniform_sha256"].nunique(), 1)
        self.assertEqual(len(seeds["records"]), 1)
        self.assertLessEqual(audit["maximum_realised_portfolio_identity_error"], 1e-12)
        self.assertTrue((forecasts["var_95"] <= forecasts["var_975"]).all())
        self.assertTrue((forecasts["var_975"] <= forecasts["var_99"]).all())
        self.assertTrue((forecasts["es_975"] >= forecasts["var_975"]).all())

    def test_pipeline_rejects_training_lookahead(self):
        training, refits, daily, returns = self._frames()
        training.loc[0, "training_date"] = pd.Timestamp("2020-01-02")
        with self.assertRaisesRegex(ValueError, "look-ahead"):
            build_gaussian_outputs(
                training,
                refits,
                daily,
                returns,
                miniature_config(),
                progress_every=0,
            )

    def test_output_schemas_bind_all_produced_fields(self):
        refits, forecasts, _, _ = build_gaussian_outputs(
            *self._frames(), miniature_config(), progress_every=0
        )
        refit_schema = json.loads(
            (ROOT / "config/schemas/gaussian_copula_refit_record.schema.json").read_text()
        )
        forecast_schema = json.loads(
            (ROOT / "config/schemas/forecast_record.schema.json").read_text()
        )
        self.assertFalse(refit_schema["additionalProperties"])
        self.assertFalse(forecast_schema["additionalProperties"])
        self.assertEqual(set(refit_schema["required"]), set(refit_schema["properties"]))
        self.assertEqual(set(forecast_schema["required"]), set(forecast_schema["properties"]))
        self.assertEqual(set(refits.columns), set(refit_schema["properties"]))
        self.assertEqual(set(forecasts.columns), set(forecast_schema["properties"]))


@unittest.skipUnless(
    all(path.is_file() for path in GAUSSIAN_ARTIFACTS),
    "requires locally generated Gaussian-copula artifacts",
)
class ProductionGaussianArtifactTests(unittest.TestCase):
    def test_current_gaussian_gate_outputs_and_seeds_are_bound(self):
        audit_path, refit_path, forecast_path, seed_path = GAUSSIAN_ARTIFACTS
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        refits = pd.read_parquet(refit_path)
        forecasts = pd.read_parquet(forecast_path)
        seeds = json.loads(seed_path.read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(audit["issues"], [])
        self.assertEqual((len(refits), len(forecasts)), (144, 3_016))
        self.assertEqual(forecasts["date"].nunique(), 1_508)
        self.assertEqual(set(forecasts["model_id"]), {"M1", "M3"})
        self.assertEqual(len(seeds["records"]), 72)
        self.assertEqual(
            audit["outputs"]["gaussian_copula_refits"]["sha256"], sha256_file(refit_path)
        )
        self.assertEqual(
            audit["outputs"]["gaussian_risk_forecasts"]["sha256"], sha256_file(forecast_path)
        )
        self.assertEqual(
            audit["outputs"]["simulation_seed_manifest"]["sha256"], sha256_file(seed_path)
        )
        refit_schema = json.loads(
            (ROOT / "config/schemas/gaussian_copula_refit_record.schema.json").read_text()
        )
        forecast_schema = json.loads(
            (ROOT / "config/schemas/forecast_record.schema.json").read_text()
        )
        self.assertEqual(set(refits.columns), set(refit_schema["properties"]))
        self.assertEqual(set(forecasts.columns), set(forecast_schema["properties"]))
        realised = forecasts.pivot(
            index="date", columns="model_id", values="realised_simple_return"
        )
        self.assertLessEqual(float(np.max(np.abs(realised["M1"] - realised["M3"]))), 1e-12)
        for record in seeds["records"]:
            uniforms = common_uniforms(
                record["year"],
                record["month"],
                draws=record["draws"],
                dimension=record["dimension"],
                base_seed=seeds["base_seed"],
            )
            self.assertEqual(record["base_uniform_sha256"], _array_sha256(uniforms))


if __name__ == "__main__":
    unittest.main()
