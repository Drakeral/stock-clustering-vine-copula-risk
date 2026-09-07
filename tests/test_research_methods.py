import json
import math
import tomllib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research_methods import (
    alphabet_issuer_composite,
    annual_buy_and_hold_returns,
    benjamini_hochberg_adjust,
    christoffersen_conditional_coverage_p_value,
    christoffersen_independence,
    circular_block_bootstrap_indices,
    common_uniforms,
    dependence_gap,
    diebold_mariano,
    empirical_var_es,
    fallback_fraction_passes,
    fz0_loss,
    group_log_returns,
    group_simple_returns,
    holm_adjust,
    kupiec_unconditional_coverage,
    margin_fit_is_acceptable,
    margin_fit_rejection_reasons,
    pairwise_spearman,
    quantile_loss,
    reconstruct_primary_portfolio,
    rolling_historical_var_es,
    simple_returns_from_log,
    var_exceptions,
)

ROOT = Path(__file__).resolve().parents[1]


class PortfolioArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.simple = pd.DataFrame(
            {
                "A": [0.10, -0.05],
                "B": [0.00, 0.02],
                "C": [-0.10, 0.04],
                "D": [0.20, 0.00],
            },
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )
        self.labels = {"A": "one", "B": "one", "C": "one", "D": "two"}

    def test_log_simple_round_trip_and_group_identity(self):
        log_returns = np.log1p(self.simple)
        converted = simple_returns_from_log(log_returns)
        pd.testing.assert_frame_equal(converted, self.simple)
        grouped = group_simple_returns(converted, self.labels)
        result = reconstruct_primary_portfolio(converted, self.labels)
        np.testing.assert_allclose(
            result.direct_simple_return, result.grouped_simple_return, atol=1e-12
        )
        pd.testing.assert_frame_equal(group_log_returns(grouped), np.log1p(grouped.simple_returns))

    def test_buy_and_hold_weights_drift(self):
        result = annual_buy_and_hold_returns(self.simple[["A", "B"]], {"A": 0.5, "B": 0.5})
        self.assertAlmostEqual(result.iloc[0], 0.05)
        expected_second_weight_a = 0.55 / 1.05
        self.assertAlmostEqual(
            result.iloc[1], expected_second_weight_a * -0.05 + (1 - expected_second_weight_a) * 0.02
        )

    def test_alphabet_composite_averages_simple_not_log_returns(self):
        frame = pd.DataFrame({"GOOG": [0.10], "GOOGL": [-0.10], "X": [0.03]})
        combined, labels = alphabet_issuer_composite(
            frame, {"GOOG": "Communication", "GOOGL": "Communication", "X": "Other"}
        )
        self.assertEqual(list(combined.columns), ["X", "ALPHABET"])
        self.assertAlmostEqual(combined.loc[0, "ALPHABET"], 0.0)
        self.assertEqual(labels["ALPHABET"], "Communication")


class DependenceTests(unittest.TestCase):
    def test_pair_weighted_and_group_weighted_gap_with_singleton(self):
        correlation = pd.DataFrame(
            [
                [1.0, 0.8, 0.2, 0.1],
                [0.8, 1.0, 0.3, 0.0],
                [0.2, 0.3, 1.0, 0.6],
                [0.1, 0.0, 0.6, 1.0],
            ],
            index=list("ABCD"),
            columns=list("ABCD"),
        )
        labels = {"A": "g1", "B": "g1", "C": "g2", "D": "singleton"}
        result = dependence_gap(correlation, labels)
        self.assertEqual(result.within_pair_count, 1)
        self.assertEqual(result.between_pair_count, 5)
        self.assertEqual(result.contributing_within_groups, 1)
        self.assertAlmostEqual(result.within_pair_mean, 0.8)
        self.assertAlmostEqual(result.between_pair_mean, 0.24)

    def test_pairwise_spearman_enforces_completeness(self):
        returns = pd.DataFrame({"A": range(10), "B": range(10), "C": [1.0] + [np.nan] * 9})
        with self.assertRaises(ValueError):
            pairwise_spearman(returns, minimum_paired_fraction=0.8)
        correlation = pairwise_spearman(returns[["A", "B"]], minimum_paired_fraction=0.8)
        self.assertAlmostEqual(correlation.loc["A", "B"], 1.0)

    def test_frozen_gics_pair_counts_and_singletons(self):
        with (ROOT / "data/raw/universe_sp100_2020-01-02.json").open() as handle:
            universe = json.load(handle)
        # DOW is the sole initial-coverage exclusion after its lineage is corrected.
        sectors = [row["gics_sector"] for row in universe["constituents"] if row["ticker"] != "DOW"]
        counts = pd.Series(sectors).value_counts()
        within = int(sum(count * (count - 1) // 2 for count in counts))
        between = math.comb(len(sectors), 2) - within
        self.assertEqual(len(sectors), 100)
        self.assertEqual(within, 551)
        self.assertEqual(between, 4_399)
        self.assertEqual(counts["Materials"], 1)
        self.assertEqual(counts["Real Estate"], 1)


class RiskAndInferenceTests(unittest.TestCase):
    def test_scores_follow_frozen_loss_convention(self):
        qloss = quantile_loss(np.array([1.0, 3.0]), np.array([2.0, 2.0]), 0.95)
        np.testing.assert_allclose(qloss, [0.05, 0.95])
        scores = fz0_loss(np.array([1.0, 4.0]), 2.0, 3.0, 0.975)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[1], scores[0])
        with self.assertRaises(ValueError):
            fz0_loss(1.0, 2.0, 1.5)

    def test_empirical_var_and_fractional_es(self):
        result = empirical_var_es(np.arange(1.0, 11.0), 0.75)
        self.assertEqual(result.var, 8.0)
        self.assertAlmostEqual(result.es, 9.2)

    def test_rolling_window_is_left_closed_right_open(self):
        dates = pd.bdate_range("2017-01-02", periods=800)
        losses = pd.Series(np.arange(800, dtype=float), index=dates)
        forecast = dates[-1]
        result = rolling_historical_var_es(losses, forecast, 0.95, minimum_observations=700)
        expected = int(((dates >= forecast - pd.DateOffset(years=3)) & (dates < forecast)).sum())
        self.assertEqual(result.observations, expected)

    def test_common_random_numbers_are_month_specific_and_reproducible(self):
        first = common_uniforms(2020, 1, draws=5, dimension=3)
        second = common_uniforms(2020, 1, draws=5, dimension=3)
        other = common_uniforms(2020, 2, draws=5, dimension=3)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(((first > 0) & (first < 1)).all())

    def test_dm_adjustments_and_gates(self):
        differential = np.sin(np.arange(100) / 5) + 0.1
        result = diebold_mariano(differential, hac_lag=7)
        self.assertEqual(result.hac_lag, 7)
        self.assertTrue(0 <= result.p_value_two_sided <= 1)
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(
            benjamini_hochberg_adjust([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04]
        )
        self.assertTrue(
            margin_fit_is_acceptable(
                phi=0.1,
                omega=0.01,
                alpha=0.05,
                beta=0.9,
                student_t_df=6,
                forecast_variance=0.02,
            )
        )
        self.assertFalse(
            margin_fit_is_acceptable(
                phi=0.1,
                omega=0.01,
                alpha=0.1,
                beta=0.9,
                student_t_df=6,
                forecast_variance=0.02,
            )
        )
        self.assertEqual(
            margin_fit_rejection_reasons(
                phi=0.99,
                omega=-0.01,
                alpha=-0.1,
                beta=1.1,
                student_t_df=2.0,
                forecast_variance=0.0,
            ),
            [
                "ar_absolute_limit",
                "nonpositive_omega",
                "negative_alpha",
                "garch_persistence_limit",
                "student_t_df_minimum",
                "nonpositive_forecast_variance",
            ],
        )
        self.assertTrue(fallback_fraction_passes(1, 100))
        self.assertFalse(fallback_fraction_passes(2, 100))

    def test_var_calibration_uses_strict_exceptions(self):
        exceptions = var_exceptions([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 4.0])
        np.testing.assert_array_equal(exceptions, [False, False, True, False])
        coverage = kupiec_unconditional_coverage(exceptions, confidence=0.75)
        self.assertEqual(coverage.exceptions, 1)
        self.assertAlmostEqual(coverage.likelihood_ratio, 0.0)
        independence = christoffersen_independence(exceptions)
        self.assertEqual(
            (
                independence.transitions_00,
                independence.transitions_01,
                independence.transitions_10,
                independence.transitions_11,
            ),
            (1, 1, 1, 0),
        )
        conditional_p = christoffersen_conditional_coverage_p_value(coverage, independence)
        self.assertTrue(0 <= conditional_p <= 1)

    def test_bootstrap_never_crosses_year_boundaries(self):
        dates = pd.to_datetime(["2020-12-28", "2020-12-29", "2021-01-04", "2021-01-05"])
        indices = circular_block_bootstrap_indices(dates, replications=20, block_length=3)
        repeated = circular_block_bootstrap_indices(dates, replications=20, block_length=3)
        self.assertEqual(indices.shape, (20, 4))
        np.testing.assert_array_equal(indices, repeated)
        self.assertTrue(np.isin(indices[:, :2], [0, 1]).all())
        self.assertTrue(np.isin(indices[:, 2:], [2, 3]).all())

    def test_model_config_is_frozen_and_complete(self):
        with (ROOT / "config/model_config.toml").open("rb") as handle:
            config = tomllib.load(handle)
        self.assertEqual(config["schema_version"], 2)
        self.assertEqual(config["simulation"]["draws_per_month"], 100_000)
        self.assertEqual(config["vine"]["primary_truncation_tree"], 3)
        self.assertEqual(config["inference"]["dm_hac_lag"], 7)
        self.assertEqual(config["forecast"]["var_confidence_levels"], [0.95, 0.975, 0.99])
        self.assertEqual(config["clustering"]["training_window_calendar_years"], 3)
        self.assertEqual(config["clustering"]["nmi_normalization"], "arithmetic_mean_entropy")
        with (ROOT / "config/schemas/forecast_record.schema.json").open() as handle:
            schema = json.load(handle)
        required = set(schema["required"])
        self.assertTrue({"var_95", "var_975", "var_99", "es_975"}.issubset(required))
        self.assertTrue({"realised_loss", "seed_components", "forecast_status"}.issubset(required))


if __name__ == "__main__":
    unittest.main()
