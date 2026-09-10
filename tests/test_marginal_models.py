import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.build_marginal_models import (
    MarginState,
    MonthlyTask,
    PersistenceBoundedGARCH,
    _fit_arch_attempt,
    _task_protocol,
    _validate_protocol,
    align_training_pits,
    filter_month,
    prepare_monthly_tasks,
    select_margin_model,
    training_pit_frame,
)

ROOT = Path(__file__).resolve().parents[1]
MARGINAL_ARTIFACTS = (
    ROOT / "data/processed/marginal_refits.parquet",
    ROOT / "data/processed/marginal_daily_forecasts.parquet",
    ROOT / "data/processed/monthly_copula_training_pits.parquet",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def marginal_config() -> dict[str, object]:
    return {
        "primary": "AR(1)-GARCH(1,1)-Student-t",
        "fallback_order": [
            "retry_ar_garch_t",
            "constant_mean_garch_t",
            "ewma_empirical",
        ],
        "estimation_return_scale": 100.0,
        "initial_optimizer_max_iterations": 1000,
        "retry_optimizer_max_iterations": 3000,
        "optimizer_tolerance": 1e-8,
        "retry_start_phi_clip": 0.95,
        "retry_start_alpha": 0.05,
        "retry_start_beta": 0.90,
        "retry_start_student_t_df": 8.0,
        "ar_absolute_limit": 0.98,
        "garch_persistence_limit": 0.999,
        "student_t_df_minimum": 2.1,
        "ewma_lambda": 0.94,
        "ewma_mean": "constant_training_mean",
        "ewma_initial_variance": "sample_variance_ddof_1",
        "ewma_innovation_distribution": "frozen_training_empirical_midrank",
        "pit_clip_lower": 1e-6,
        "pit_clip_upper": 0.999999,
        "maximum_ewma_fit_fraction": 0.01,
        "drop_ar_for_insignificant_p_value": False,
    }


class MonthlyTaskTests(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        dates = pd.bdate_range("2017-01-03", "2020-12-31")
        return pd.DataFrame(
            {
                "date": dates,
                "year": 2020,
                "universe_variant": "security_primary",
                "sample_role": np.where(dates.year == 2020, "evaluation", "training"),
                "grouping_id": "gics_sector",
                "group_id": "Technology",
                "group_size": 10,
                "portfolio_weight": 1.0,
                "log_return": np.sin(np.arange(len(dates))) / 100,
            }
        )

    def test_tasks_are_monthly_and_exclude_the_refit_date(self):
        config = {
            "forecast": {"training_window_calendar_years": 3, "minimum_training_observations": 700},
            "clustering": {"evaluation_start_year": 2020, "evaluation_end_year": 2020},
        }
        tasks = prepare_monthly_tasks(self._panel(), config)
        self.assertEqual(len(tasks), 12)
        self.assertEqual([task.month for task in tasks], list(range(1, 13)))
        for task in tasks:
            self.assertLess(task.training.index.max(), task.refit_date)
            self.assertEqual(task.evaluation.index.min(), task.refit_date)
            self.assertGreaterEqual(len(task.training), 700)
        self.assertGreater(tasks[1].training.index.max(), tasks[0].training.index.max())

    def test_overlapping_sample_roles_fail_instead_of_being_deduplicated(self):
        panel = self._panel()
        duplicate = panel.loc[panel["date"] == pd.Timestamp("2020-01-02")].copy()
        duplicate["sample_role"] = "training"
        panel = pd.concat([panel, duplicate], ignore_index=True)
        config = {
            "forecast": {"training_window_calendar_years": 3, "minimum_training_observations": 700},
            "clustering": {"evaluation_start_year": 2020, "evaluation_end_year": 2020},
        }
        with self.assertRaisesRegex(ValueError, "overlapping sample roles"):
            prepare_monthly_tasks(panel, config)

    def test_invalid_group_weight_identity_is_rejected(self):
        panel = self._panel()
        panel["portfolio_weight"] = 0.25
        config = {
            "forecast": {"training_window_calendar_years": 3, "minimum_training_observations": 700},
            "clustering": {"evaluation_start_year": 2020, "evaluation_end_year": 2020},
        }
        with self.assertRaisesRegex(ValueError, "normalized group sizes"):
            prepare_monthly_tasks(panel, config)

    def test_task_protocol_rejects_invalid_windows(self):
        config = {
            "forecast": {"training_window_calendar_years": 3, "minimum_training_observations": 2},
            "clustering": {"evaluation_start_year": 2021, "evaluation_end_year": 2020},
        }
        with self.assertRaisesRegex(ValueError, "at least three"):
            _task_protocol(config)


class FallbackAndFilteringTests(unittest.TestCase):
    def test_persistence_constraint_rejects_invalid_limit(self):
        with self.assertRaisesRegex(ValueError, "must lie in"):
            PersistenceBoundedGARCH(1.0)

    def test_marginal_protocol_rejects_invalid_numeric_configuration(self):
        valid = {"marginal": marginal_config()}
        self.assertIs(_validate_protocol(valid), valid["marginal"])
        cases = [
            ("estimation_return_scale", 0.0, "must be positive"),
            ("optimizer_tolerance", False, "missing or invalid"),
            ("initial_optimizer_max_iterations", True, "positive integer"),
            ("retry_start_phi_clip", 0.99, "retry_start_phi_clip"),
            ("retry_start_beta", 0.95, "persistence limit"),
            ("retry_start_student_t_df", 2.1, "student_t_df_minimum"),
            ("ewma_lambda", 1.0, "ewma_lambda"),
            ("pit_clip_lower", 0.999999, "PIT clipping bounds"),
            ("maximum_ewma_fit_fraction", 1.1, "maximum_ewma_fit_fraction"),
        ]
        for key, value, message in cases:
            with self.subTest(key=key):
                invalid = copy.deepcopy(valid)
                invalid["marginal"][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    _validate_protocol(invalid)

    def test_fallback_order_reaches_ewma_deterministically(self):
        calls: list[tuple[str, str, int, bool]] = []

        def rejecting_fitter(sample, attempt_id, mean_model, starts, maxiter, _config):
            calls.append((attempt_id, mean_model, maxiter, starts is not None))
            return None, {
                "attempt_id": attempt_id,
                "status": "rejected",
                "rejection_reasons": ["fixture_rejection"],
            }

        index = pd.bdate_range("2017-01-03", periods=750)
        sample = pd.Series(np.sin(np.arange(750) / 7) / 100, index=index)
        first, attempts = select_margin_model(
            sample, marginal_config(), arch_fitter=rejecting_fitter
        )
        second, _ = select_margin_model(sample, marginal_config(), arch_fitter=rejecting_fitter)
        self.assertEqual(
            [call[0] for call in calls[:3]],
            ["initial_ar_garch_t", "retry_ar_garch_t", "constant_mean_garch_t"],
        )
        self.assertEqual(first.selected_method, "ewma_empirical")
        self.assertEqual(first.fallback_level, 3)
        self.assertEqual(len(attempts), 4)
        self.assertAlmostEqual(first.next_variance, second.next_variance)
        np.testing.assert_array_equal(first.empirical_innovations, second.empirical_innovations)

    def test_expected_solver_error_is_recorded_for_fallback(self):
        sample = pd.Series(
            np.sin(np.arange(750) / 7) / 100,
            index=pd.bdate_range("2017-01-03", periods=750),
        )
        with patch(
            "scripts.build_marginal_models.StudentsT",
            side_effect=RuntimeError("synthetic solver failure"),
        ):
            state, attempt = _fit_arch_attempt(
                sample,
                "initial_ar_garch_t",
                "AR",
                None,
                1000,
                marginal_config(),
            )

        self.assertIsNone(state)
        self.assertEqual(attempt["exception_type"], "RuntimeError")
        self.assertEqual(attempt["rejection_reasons"], ["fit_exception"])

    def test_programming_error_is_not_silently_treated_as_fit_failure(self):
        sample = pd.Series(
            np.sin(np.arange(750) / 7) / 100,
            index=pd.bdate_range("2017-01-03", periods=750),
        )
        with (
            patch(
                "scripts.build_marginal_models.StudentsT",
                side_effect=KeyError("synthetic programming defect"),
            ),
            self.assertRaisesRegex(KeyError, "synthetic programming defect"),
        ):
            _fit_arch_attempt(
                sample,
                "initial_ar_garch_t",
                "AR",
                None,
                1000,
                marginal_config(),
            )

    def test_daily_filter_uses_fixed_parameters_and_clips_pits(self):
        evaluation = pd.Series([0.0, 0.01], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
        task = MonthlyTask(
            year=2020,
            month=1,
            universe_variant="security_primary",
            grouping_id="gics_sector",
            group_id="Technology",
            group_size=10,
            portfolio_weight=0.1,
            refit_date=evaluation.index[0],
            requested_training_start=pd.Timestamp("2017-01-02"),
            training=pd.Series([0.0], index=pd.to_datetime(["2020-01-01"])),
            evaluation=evaluation,
        )
        state = MarginState(
            selected_method="ar_garch_t",
            fallback_level=0,
            mean_constant=0.0,
            phi=0.0,
            omega=0.1,
            alpha=0.1,
            beta=0.8,
            student_t_df=8.0,
            next_variance=1.0,
            previous_return=0.0,
            training_standardized_residuals=pd.Series([0.0], index=pd.to_datetime(["2020-01-01"])),
            empirical_innovations=None,
            convergence_flag=0,
            loglikelihood=-1.0,
            aic=1.0,
            bic=1.0,
        )
        rows = filter_month(task, state, marginal_config())
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["conditional_variance_log_return"], 0.0001)
        self.assertAlmostEqual(rows[1]["conditional_variance_log_return"], 0.00009)
        self.assertTrue(all(1e-6 <= row["pit"] <= 0.999999 for row in rows))

        training = training_pit_frame(task, state, marginal_config())
        self.assertEqual(len(training), 1)
        self.assertEqual(training.loc[0, "training_date"], pd.Timestamp("2020-01-01"))
        self.assertAlmostEqual(training.loc[0, "pit"], 0.5)

    def test_training_pit_alignment_keeps_only_complete_dimensions(self):
        dates = pd.to_datetime(["2019-12-27", "2019-12-30"])
        common = {
            "year": 2020,
            "month": 1,
            "refit_date": pd.Timestamp("2020-01-02"),
            "universe_variant": "security_primary",
            "grouping_id": "gics_sector",
        }
        first = pd.DataFrame(
            {
                **common,
                "training_date": dates,
                "group_id": "one",
                "pit": [0.2, 0.3],
            }
        )
        second = pd.DataFrame(
            {
                **common,
                "training_date": dates[1:],
                "group_id": "two",
                "pit": [0.4],
            }
        )
        training = pd.concat([first, second], ignore_index=True)
        refits = pd.DataFrame(
            {
                "year": [2020, 2020],
                "month": [1, 1],
                "universe_variant": ["security_primary", "security_primary"],
                "grouping_id": ["gics_sector", "gics_sector"],
                "group_id": ["one", "two"],
            }
        )
        aligned, issues = align_training_pits(training, refits, expected_dimension=2)
        self.assertEqual(issues, [])
        self.assertEqual(len(aligned), 2)
        self.assertEqual(set(aligned["training_date"]), {pd.Timestamp("2019-12-30")})

    def test_output_schemas_bind_all_produced_fields(self):
        for name in [
            "marginal_refit_record",
            "marginal_daily_record",
            "marginal_training_pit_record",
        ]:
            path = ROOT / f"config/schemas/{name}.schema.json"
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))


@unittest.skipUnless(
    all(path.is_file() for path in MARGINAL_ARTIFACTS),
    "requires locally generated marginal-model artifacts",
)
class ProductionArtifactTests(unittest.TestCase):
    def test_current_primary_marginal_gate_and_artifacts_pass(self):
        audit_path = ROOT / "data/audit/marginal_model_quality.json"
        refit_path = ROOT / "data/processed/marginal_refits.parquet"
        daily_path = ROOT / "data/processed/marginal_daily_forecasts.parquet"
        training_path = ROOT / "data/processed/monthly_copula_training_pits.parquet"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        refits = pd.read_parquet(refit_path)
        daily = pd.read_parquet(daily_path)
        training = pd.read_parquet(training_path)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["schema_version"], 2)
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(audit["issues"], [])
        self.assertEqual((len(refits), len(daily)), (1584, 33176))
        self.assertEqual(len(training), 1192730)
        self.assertEqual(audit["copula_training_block_count"], 144)
        self.assertEqual(audit["copula_dimension"], 11)
        self.assertGreaterEqual(audit["minimum_aligned_training_dates"], 700)
        training_schema = json.loads(
            (ROOT / "config/schemas/marginal_training_pit_record.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(training.columns), set(training_schema["properties"]))
        self.assertEqual(set(training["margin_refit_id"]), set(refits["refit_id"]))
        self.assertEqual(training["copula_refit_id"].nunique(), 144)
        self.assertLessEqual(audit["ewma_fit_fraction"], audit["maximum_ewma_fit_fraction"])
        self.assertEqual(audit["outputs"]["marginal_refits"]["sha256"], sha256_file(refit_path))
        self.assertEqual(
            audit["outputs"]["marginal_daily_forecasts"]["sha256"], sha256_file(daily_path)
        )
        self.assertEqual(
            audit["outputs"]["monthly_copula_training_pits"]["sha256"],
            sha256_file(training_path),
        )
        self.assertFalse(
            refits.duplicated(
                ["year", "month", "universe_variant", "grouping_id", "group_id"]
            ).any()
        )
        self.assertFalse(
            daily.duplicated(["date", "universe_variant", "grouping_id", "group_id"]).any()
        )
        self.assertTrue((pd.to_datetime(refits["last_training_date"]) < refits["refit_date"]).all())
        daily_counts = daily.groupby("date").size()
        self.assertEqual((int(daily_counts.min()), int(daily_counts.max())), (22, 22))
        matrix_key = ["year", "month", "universe_variant", "grouping_id", "training_date"]
        training_dimensions = training.groupby(matrix_key).size()
        self.assertEqual((int(training_dimensions.min()), int(training_dimensions.max())), (11, 11))
        self.assertFalse(training.duplicated(matrix_key + ["group_id"]).any())
        self.assertTrue((training["training_date"] < training["refit_date"]).all())


if __name__ == "__main__":
    unittest.main()
