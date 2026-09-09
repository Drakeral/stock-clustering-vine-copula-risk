import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from scripts.build_gaussian_copula import build_gaussian_outputs
from scripts.build_vine_copula import (
    VineFitError,
    build_vine_outputs,
    fit_vine_copula,
    validate_vine_protocol,
    vine_dependence_uniforms,
    vine_log_density,
)
from scripts.research_methods import common_uniforms

ROOT = Path(__file__).resolve().parents[1]
VINE_ARTIFACTS = (
    ROOT / "data/audit/vine_copula_quality.json",
    ROOT / "data/processed/vine_copula_refits.parquet",
    ROOT / "data/processed/vine_risk_forecasts.parquet",
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
        "vine": {
            "dimension": 2,
            "primary_truncation_tree": 1,
            "robustness_truncation_tree": 1,
            "structure_selection": "dissmann_abs_kendall_tau_mst",
            "family_selection": "aic_sequential_mle",
            "families": [
                "independence",
                "gaussian",
                "student_t",
                "frank",
                "clayton",
                "gumbel",
            ],
            "rotations_degrees": [0, 90, 180, 270],
            "pit_clip_lower": 1e-6,
            "pit_clip_upper": 0.999999,
            "student_t_df_minimum": 2.1,
            "student_t_df_maximum": 50.0,
            "pair_failure_action": "independence",
            "maximum_failed_pair_fraction": 0.10,
            "monthly_structure_failure_action": "matched_gaussian",
            "maximum_whole_vine_fallback_date_fraction": 0.01,
        },
    }


def miniature_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_dates = pd.bdate_range("2019-12-02", periods=20)
    evaluation_dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    training_rows: list[dict[str, object]] = []
    refit_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
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
                base_pit = 0.05 + 0.9 * index / (len(training_dates) - 1)
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
                        "margin_refit_id": marginal_refit_id,
                        "pit": base_pit if group_index == 0 else 0.98 * base_pit + 0.01,
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
                    "training_observations": len(training_dates),
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
                        "refit_id": marginal_refit_id,
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


class VineMethodTests(unittest.TestCase):
    def test_fit_simulation_density_and_serialization_are_valid(self):
        config = miniature_config()
        base = common_uniforms(2020, 1, draws=300, dimension=2)
        pits = np.column_stack([base[:, 0], 0.8 * base[:, 0] + 0.2 * base[:, 1]])
        fit = fit_vine_copula(
            pits,
            config["vine"],
            truncation_level=1,
            fit_seeds=[5110, 2020, 1],
        )
        self.assertEqual(fit.pair_count, 1)
        self.assertEqual(fit.failed_pair_count, 0)
        self.assertEqual(fit.model.trunc_lvl, 1)
        independent = common_uniforms(2021, 2, draws=100, dimension=2)
        first = vine_dependence_uniforms(independent, fit.model, lower=1e-6, upper=0.999999)
        second = vine_dependence_uniforms(independent, fit.model, lower=1e-6, upper=0.999999)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(vine_log_density(first[:10], fit.model)).all())
        restored = pv.Vinecop.from_json(fit.model.to_json())
        np.testing.assert_allclose(restored.pdf(first[:10]), fit.model.pdf(first[:10]))

    def test_protocol_rejects_an_unfrozen_family_search(self):
        config = miniature_config()
        config["vine"]["families"] = ["independence", "gaussian"]
        with self.assertRaisesRegex(ValueError, "unsupported vine protocol"):
            validate_vine_protocol(config)


class VinePipelineTests(unittest.TestCase):
    def _inputs(self):
        frames = miniature_frames()
        config = miniature_config()
        gaussian_refits, gaussian_forecasts, seeds, gaussian_audit = build_gaussian_outputs(
            *frames,
            config,
            progress_every=0,
        )
        self.assertEqual(gaussian_audit["status"], "pass")
        return frames, config, gaussian_refits, gaussian_forecasts, seeds

    def test_pipeline_builds_matched_m2_m4_forecasts_and_schema(self):
        frames, config, gaussian_refits, _, seeds = self._inputs()
        refits, forecasts, audit = build_vine_outputs(
            *frames,
            gaussian_refits,
            seeds,
            config,
            progress_every=0,
        )
        self.assertEqual(audit["status"], "pass")
        self.assertEqual((len(refits), len(forecasts)), (2, 4))
        self.assertEqual(set(refits["model_id"]), {"M2", "M4"})
        self.assertFalse(refits["whole_vine_fallback"].any())
        self.assertEqual(refits["pair_count"].tolist(), [1, 1])
        self.assertLessEqual(audit["maximum_realised_portfolio_identity_error"], 1e-12)
        schema = json.loads(
            (ROOT / "config/schemas/vine_copula_refit_record.schema.json").read_text()
        )
        forecast_schema = json.loads(
            (ROOT / "config/schemas/forecast_record.schema.json").read_text()
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(forecast_schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(set(forecast_schema["required"]), set(forecast_schema["properties"]))
        self.assertEqual(set(refits.columns), set(schema["properties"]))
        self.assertEqual(set(forecasts.columns), set(forecast_schema["properties"]))

    def test_structure_failure_uses_the_exact_matched_gaussian(self):
        frames, config, gaussian_refits, gaussian_forecasts, seeds = self._inputs()

        def failed_fit(*_args, **_kwargs):
            raise VineFitError("synthetic failure")

        refits, forecasts, audit = build_vine_outputs(
            *frames,
            gaussian_refits,
            seeds,
            config,
            progress_every=0,
            vine_fitter=failed_fit,
        )
        self.assertTrue(refits["whole_vine_fallback"].all())
        self.assertTrue(forecasts["whole_vine_fallback"].all())
        self.assertFalse(audit["eligible_to_be_declared_best"])
        self.assertEqual(audit["model_whole_vine_fallback_date_counts"], {"M2": 2, "M4": 2})
        self.assertEqual(audit["model_whole_vine_fallback_date_fractions"], {"M2": 1.0, "M4": 1.0})
        for vine_id, gaussian_id in (("M2", "M1"), ("M4", "M3")):
            vine_rows = forecasts.loc[forecasts["model_id"] == vine_id].set_index("date")
            gaussian_rows = gaussian_forecasts.loc[
                gaussian_forecasts["model_id"] == gaussian_id
            ].set_index("date")
            np.testing.assert_allclose(
                vine_rows[["var_95", "var_975", "var_99", "es_975"]],
                gaussian_rows[["var_95", "var_975", "var_99", "es_975"]],
                rtol=0.0,
                atol=0.0,
            )

    def test_programming_error_is_not_silently_converted_to_a_fallback(self):
        frames, config, gaussian_refits, _, seeds = self._inputs()

        def broken_fit(*_args, **_kwargs):
            raise KeyError("synthetic programming defect")

        with self.assertRaisesRegex(KeyError, "synthetic programming defect"):
            build_vine_outputs(
                *frames,
                gaussian_refits,
                seeds,
                config,
                progress_every=0,
                vine_fitter=broken_fit,
            )

    def test_seed_record_metadata_is_validated_before_modelling(self):
        frames, config, gaussian_refits, _, seeds = self._inputs()
        seeds["records"][0]["dimension"] = 99

        with self.assertRaisesRegex(ValueError, "seed manifest record differs"):
            build_vine_outputs(
                *frames,
                gaussian_refits,
                seeds,
                config,
                progress_every=0,
            )


@unittest.skipUnless(
    all(path.is_file() for path in VINE_ARTIFACTS),
    "requires locally generated vine-copula artifacts",
)
class ProductionVineArtifactTests(unittest.TestCase):
    def test_current_vine_gate_outputs_models_and_constraints_are_bound(self):
        audit_path, refit_path, forecast_path = VINE_ARTIFACTS
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        refits = pd.read_parquet(refit_path)
        forecasts = pd.read_parquet(forecast_path)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(audit["issues"], [])
        self.assertEqual((len(refits), len(forecasts)), (144, 3_016))
        self.assertEqual(forecasts["date"].nunique(), 1_508)
        self.assertEqual(set(forecasts["model_id"]), {"M2", "M4"})
        self.assertEqual(audit["outputs"]["vine_copula_refits"]["sha256"], sha256_file(refit_path))
        self.assertEqual(
            audit["outputs"]["vine_risk_forecasts"]["sha256"], sha256_file(forecast_path)
        )
        self.assertTrue((refits["pair_count"] == 27).all())
        self.assertTrue((refits["failed_pair_fraction"] <= 0.10).all())
        self.assertEqual(
            audit["whole_vine_fallback_refit_count"], int(refits["whole_vine_fallback"].sum())
        )
        self.assertEqual(
            audit["eligible_to_be_declared_best"],
            audit["maximum_model_whole_vine_fallback_date_fraction"] <= 0.01,
        )
        allowed_families = {
            pv.BicopFamily.indep,
            pv.BicopFamily.gaussian,
            pv.BicopFamily.student,
            pv.BicopFamily.frank,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
        }
        for row in refits.loc[~refits["whole_vine_fallback"]].itertuples():
            model = pv.Vinecop.from_json(row.vine_model_json)
            self.assertEqual((model.dim, model.trunc_lvl), (11, 3))
            for tree in model.pair_copulas:
                for pair in tree:
                    self.assertIn(pair.family, allowed_families)
                    self.assertIn(pair.rotation, {0, 90, 180, 270})
                    if pair.family == pv.BicopFamily.student:
                        degrees_of_freedom = float(pair.parameters[1, 0])
                        self.assertGreaterEqual(degrees_of_freedom, 2.1)
                        self.assertLessEqual(degrees_of_freedom, 50.0)
        realised = forecasts.pivot(
            index="date", columns="model_id", values="realised_simple_return"
        )
        self.assertLessEqual(float(np.max(np.abs(realised["M2"] - realised["M4"]))), 1e-12)


if __name__ == "__main__":
    unittest.main()
