import copy
import json
import tomllib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_risk_models import (
    DAILY_SCORE_COLUMNS,
    MODEL_GROUPINGS,
    MODEL_IDS,
    _array_sha256,
    _gap,
    _spearman_correlation,
    accepted_h1_bootstrap_indices,
    build_evaluation_outputs,
    combine_forecasts,
    h1_bootstrap,
    prepare_h1_years,
    score_forecasts,
    validate_evaluation_protocol,
)
from scripts.pipeline_io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DAILY_OUTPUT = ROOT / "data/processed/risk_evaluation_daily.parquet"
AUDIT_OUTPUT = ROOT / "data/audit/model_evaluation.json"
SEED_MANIFEST = ROOT / "data/manifests/inference_seed_manifest.json"


def load_configs() -> tuple[dict, dict]:
    with (ROOT / "config/evaluation_config.toml").open("rb") as handle:
        evaluation = tomllib.load(handle)
    with (ROOT / "config/model_config.toml").open("rb") as handle:
        model = tomllib.load(handle)
    return evaluation, model


def forecast_fixture(
    periods: int = 30, dates: pd.DatetimeIndex | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=periods) if dates is None else dates
    periods = len(dates)
    realised_return = np.sin(np.arange(periods) / 3) / 100
    realised_loss = -realised_return
    frames = []
    for model_number, model_id in enumerate(MODEL_IDS):
        variation = np.cos(np.arange(periods) / (3.5 + model_number)) * 0.0002
        var_95 = 0.0105 + model_number * 0.0001 + variation
        var_975 = var_95 + 0.003
        var_99 = var_975 + 0.003
        es_975 = var_975 + 0.0035
        frame = pd.DataFrame(
            {
                "date": dates,
                "model_id": model_id,
                "grouping_id": MODEL_GROUPINGS[model_id],
                "refit_id": f"fixture:{model_id}",
                "var_95": var_95,
                "var_975": var_975,
                "var_99": var_99,
                "es_975": es_975,
                "realised_simple_return": realised_return,
                "realised_loss": realised_loss,
                "seed_components": None if model_id == "M0" else [[5110, 2020, 1]] * periods,
                "margin_fallback_count": 0,
                "whole_vine_fallback": False,
                "copula_log_score": (
                    None if model_id == "M0" else np.sin(np.arange(periods) / 5) + model_number
                ),
                "forecast_status": "ok",
            }
        )
        frames.append(frame)
    return frames[0], pd.concat([frames[1], frames[3]]), pd.concat([frames[2], frames[4]])


def h1_fixture(
    years: tuple[int, ...] = (2020, 2021), periods: int = 12
) -> tuple[pd.DataFrame, dict, dict]:
    dates = pd.DatetimeIndex(
        [date for year in years for date in pd.bdate_range(f"{year}-01-04", periods=periods)]
    )
    position = np.arange(len(dates), dtype=float)
    returns = pd.DataFrame(
        {
            "A": np.sin(position / 2),
            "B": np.sin(position / 2) + np.cos(position) / 10,
            "C": np.cos(position / 3),
            "D": -np.cos(position / 3) + np.sin(position) / 10,
        },
        index=dates,
    )
    labels = [
        {"ticker": "A", "gics_sector": "g1", "hierarchical_cluster": "c1"},
        {"ticker": "B", "gics_sector": "g1", "hierarchical_cluster": "c2"},
        {"ticker": "C", "gics_sector": "g2", "hierarchical_cluster": "c1"},
        {"ticker": "D", "gics_sector": "g2", "hierarchical_cluster": "c2"},
    ]
    assignments = {"years": [{"year": year, "assignments": labels} for year in years]}
    diagnostic_years = []
    for year in years:
        annual = returns.loc[returns.index.year == year].to_numpy()
        correlation = _spearman_correlation(annual)
        rows, columns = np.triu_indices(4, k=1)
        gics = np.asarray([row["gics_sector"] for row in labels])
        clusters = np.asarray([row["hierarchical_cluster"] for row in labels])
        diagnostic_years.append(
            {
                "year": year,
                "gics_dependence_gap": {
                    "gap": _gap(correlation, rows, columns, gics[rows] == gics[columns])
                },
                "hierarchical_dependence_gap": {
                    "gap": _gap(
                        correlation,
                        rows,
                        columns,
                        clusters[rows] == clusters[columns],
                    )
                },
            }
        )
    return returns, assignments, {"years": diagnostic_years}


class EvaluationMethodTests(unittest.TestCase):
    def test_protocol_is_exact_and_separate_from_model_fit_configuration(self):
        evaluation, model = load_configs()
        sections = validate_evaluation_protocol(evaluation, model)
        self.assertEqual(sections["evaluation"]["model_ids"], list(MODEL_IDS))
        invalid = copy.deepcopy(evaluation)
        invalid["h2"]["hac_lag"] = 8
        with self.assertRaisesRegex(ValueError, "settings disagree"):
            validate_evaluation_protocol(invalid, model)

    def test_combination_rejects_mismatched_dates_and_realised_losses(self):
        historical, gaussian, vine = forecast_fixture()
        evaluation = {
            "evaluation_start_year": 2020,
            "evaluation_end_year": 2020,
            "required_forecast_statuses": ["ok", "fallback"],
            "realised_loss_identity_tolerance": 1e-12,
        }
        combined, validation = combine_forecasts(historical, gaussian, vine, evaluation)
        self.assertEqual(len(combined), 30 * 5)
        self.assertEqual(validation["maximum_realised_loss_identity_error"], 0.0)

        bad_date = vine.copy()
        bad_date.loc[bad_date.index[0], "date"] = pd.Timestamp("2020-12-31")
        with self.assertRaisesRegex(ValueError, "dates do not match"):
            combine_forecasts(historical, gaussian, bad_date, evaluation)

        bad_loss = gaussian.copy()
        bad_loss.loc[bad_loss.index[0], "realised_loss"] += 0.01
        with self.assertRaisesRegex(ValueError, "negative realised simple return"):
            combine_forecasts(historical, bad_loss, vine, evaluation)

    def test_daily_scores_follow_schema_and_strict_exception_rule(self):
        historical, gaussian, vine = forecast_fixture()
        evaluation = {
            "evaluation_start_year": 2020,
            "evaluation_end_year": 2020,
            "required_forecast_statuses": ["ok", "fallback"],
            "realised_loss_identity_tolerance": 1e-12,
        }
        combined, _ = combine_forecasts(historical, gaussian, vine, evaluation)
        scores = score_forecasts(combined)
        self.assertEqual(tuple(scores.columns), DAILY_SCORE_COLUMNS)
        self.assertTrue((scores[["quantile_loss_95", "quantile_loss_99"]] >= 0).all().all())
        with (ROOT / "config/schemas/risk_evaluation_daily_record.schema.json").open() as handle:
            schema = json.load(handle)
        self.assertEqual(set(schema["required"]), set(DAILY_SCORE_COLUMNS))
        self.assertFalse(
            scores.loc[scores["realised_loss"] == scores["var_95"], "exception_95"].any()
        )

    def test_h1_bootstrap_is_paired_reproducible_and_within_year(self):
        returns, assignments, diagnostics = h1_fixture()
        years = prepare_h1_years(returns, assignments, diagnostics)
        first, first_manifest = h1_bootstrap(
            years, replications=100, block_length=3, confidence_level=0.95
        )
        second, second_manifest = h1_bootstrap(
            years, replications=100, block_length=3, confidence_level=0.95
        )
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["indices_shape"], [100, 24])
        self.assertEqual(len(first_manifest["indices_sha256"]), 64)

        accepted, candidate_count, rejected = accepted_h1_bootstrap_indices(
            years, replications=100, block_length=3
        )
        self.assertEqual(_array_sha256(accepted), first_manifest["indices_sha256"])
        self.assertEqual(candidate_count, first_manifest["candidate_draws"])
        self.assertEqual(rejected, first_manifest["degenerate_candidate_rejections"])

    def test_degenerate_h1_candidates_are_discarded_as_whole_replications(self):
        returns, assignments, diagnostics = h1_fixture()
        years = prepare_h1_years(returns, assignments, diagnostics)
        first = years[0]
        values = first.values.copy()
        values[3:, 0] = 0.0
        years[0] = type(first)(
            first.year,
            first.dates,
            values,
            first.upper_rows,
            first.upper_columns,
            first.gics_within,
            first.cluster_within,
        )
        accepted, candidate_count, rejected = accepted_h1_bootstrap_indices(
            years, replications=100, block_length=3
        )
        self.assertGreater(candidate_count, 100)
        self.assertGreater(rejected, 0)
        for candidate in accepted:
            local = candidate[: len(first.dates)]
            self.assertFalse((np.ptp(values[local], axis=0) == 0).any())

    def test_complete_evaluator_returns_six_dm_and_twenty_calibration_tests(self):
        years = tuple(range(2020, 2026))
        dates = pd.DatetimeIndex(
            [date for year in years for date in pd.bdate_range(f"{year}-01-04", periods=25)]
        )
        historical, gaussian, vine = forecast_fixture(dates=dates)
        returns, assignments, diagnostics = h1_fixture(years=years, periods=25)
        evaluation, model = load_configs()
        scores, audit, manifest = build_evaluation_outputs(
            historical,
            gaussian,
            vine,
            returns,
            assignments,
            diagnostics,
            evaluation,
            model,
            vine_eligible=True,
        )
        self.assertEqual(len(scores), 750)
        self.assertEqual(len(audit["full_period_calibration"]), 10)
        self.assertEqual(len(audit["diebold_mariano_comparisons"]), 6)
        self.assertEqual(manifest["replications"], 10_000)
        self.assertEqual(
            set(audit["hypotheses"]),
            {
                "H1_cluster_information",
                "H2_copula_specification",
                "H3_integrated_model",
            },
        )
        json.dumps(audit, allow_nan=False)


@unittest.skipUnless(
    DAILY_OUTPUT.exists() and AUDIT_OUTPUT.exists() and SEED_MANIFEST.exists(),
    "production evaluation artifacts are not available in this checkout",
)
class ProductionEvaluationArtifactTests(unittest.TestCase):
    def test_current_evaluation_outputs_and_bootstrap_seed_are_bound(self):
        audit = json.loads(AUDIT_OUTPUT.read_text(encoding="utf-8"))
        manifest = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
        scores = pd.read_parquet(DAILY_OUTPUT)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual(len(scores), 5 * 1_508)
        self.assertEqual(tuple(scores.columns), DAILY_SCORE_COLUMNS)
        self.assertEqual(
            audit["outputs"]["risk_evaluation_daily"]["sha256"], sha256_file(DAILY_OUTPUT)
        )
        self.assertEqual(
            audit["outputs"]["inference_seed_manifest"]["sha256"],
            sha256_file(SEED_MANIFEST),
        )
        self.assertEqual(len(audit["full_period_calibration"]), 10)
        self.assertEqual(len(audit["diebold_mariano_comparisons"]), 6)
        self.assertEqual(manifest["indices_shape"], [10_000, 1_508])
        stock_returns = pd.read_parquet(
            ROOT / "data/processed/portfolio_constituent_simple_returns.parquet"
        )
        assignments = json.loads(
            (ROOT / "data/processed/annual_group_assignments.json").read_text(encoding="utf-8")
        )
        clustering = json.loads(
            (ROOT / "data/audit/clustering_diagnostics.json").read_text(encoding="utf-8")
        )
        years = prepare_h1_years(stock_returns, assignments, clustering)
        indices, candidate_count, rejected = accepted_h1_bootstrap_indices(
            years,
            replications=manifest["replications"],
            block_length=manifest["block_length_trading_days"],
            base_seed=manifest["seed_components"][0],
        )
        self.assertEqual(_array_sha256(indices), manifest["indices_sha256"])
        self.assertEqual(candidate_count, manifest["candidate_draws"])
        self.assertEqual(rejected, manifest["degenerate_candidate_rejections"])


if __name__ == "__main__":
    unittest.main()
