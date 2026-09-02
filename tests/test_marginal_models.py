import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_marginal_models import (
    MarginState,
    MonthlyTask,
    filter_month,
    prepare_monthly_tasks,
    select_margin_model,
)

ROOT = Path(__file__).resolve().parents[1]


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
                "portfolio_weight": 0.1,
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


class FallbackAndFilteringTests(unittest.TestCase):
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

    def test_output_schemas_bind_all_produced_fields(self):
        for name in ["marginal_refit_record", "marginal_daily_record"]:
            path = ROOT / f"config/schemas/{name}.schema.json"
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))


class ProductionArtifactTests(unittest.TestCase):
    def test_current_primary_marginal_gate_and_artifacts_pass(self):
        audit_path = ROOT / "data/audit/marginal_model_quality.json"
        refit_path = ROOT / "data/processed/marginal_refits.parquet"
        daily_path = ROOT / "data/processed/marginal_daily_forecasts.parquet"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        refits = pd.read_parquet(refit_path)
        daily = pd.read_parquet(daily_path)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(audit["issues"], [])
        self.assertEqual((len(refits), len(daily)), (1584, 33176))
        self.assertLessEqual(audit["ewma_fit_fraction"], audit["maximum_ewma_fit_fraction"])
        self.assertEqual(audit["outputs"]["marginal_refits"]["sha256"], sha256_file(refit_path))
        self.assertEqual(
            audit["outputs"]["marginal_daily_forecasts"]["sha256"], sha256_file(daily_path)
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


if __name__ == "__main__":
    unittest.main()
