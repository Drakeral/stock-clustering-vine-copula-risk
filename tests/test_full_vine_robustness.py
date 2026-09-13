import copy
import json
import tomllib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_full_vine_robustness import (
    DAILY_COLUMNS,
    VARIANT_ORDER,
    build_robustness_outputs,
    combine_robustness_forecasts,
    validate_refit_identity,
    validate_robustness_protocol,
)
from scripts.pipeline_io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ARTIFACTS = (
    ROOT / "data/processed/vine_copula_refits.parquet",
    ROOT / "data/processed/vine_risk_forecasts.parquet",
    ROOT / "data/audit/vine_copula_quality.json",
    ROOT / "data/processed/full_vine_copula_refits.parquet",
    ROOT / "data/processed/full_vine_risk_forecasts.parquet",
    ROOT / "data/audit/full_vine_robustness_quality.json",
    ROOT / "data/processed/full_vine_robustness_daily.parquet",
    ROOT / "data/audit/full_vine_robustness_evaluation.json",
)


def load_configs() -> tuple[dict, dict]:
    with (ROOT / "config/full_vine_robustness.toml").open("rb") as handle:
        robustness = tomllib.load(handle)
    with (ROOT / "config/model_config.toml").open("rb") as handle:
        model = tomllib.load(handle)
    return robustness, model


def forecast_fixtures(periods: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    position = np.arange(periods, dtype=float)
    realised_return = -(0.004 + 0.014 * np.sin(position / 4.0))
    frames: dict[int, list[pd.DataFrame]] = {3: [], 10: []}
    for model_number, (model_id, grouping_id) in enumerate(
        (("M2", "gics"), ("M4", "hierarchical"))
    ):
        primary_variation = 0.0004 * np.cos(position / (3.0 + model_number))
        full_adjustment = 0.0003 * np.sin(position / (4.5 + model_number))
        for level in (3, 10):
            adjustment = np.zeros(periods) if level == 3 else full_adjustment
            var_95 = 0.012 + model_number * 0.0005 + primary_variation + adjustment
            var_975 = var_95 + 0.004
            var_99 = var_975 + 0.003
            es_975 = var_975 + 0.004
            frames[level].append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "model_id": model_id,
                        "grouping_id": grouping_id,
                        "refit_id": f"fixture:{model_id}:t{level}",
                        "var_95": var_95,
                        "var_975": var_975,
                        "var_99": var_99,
                        "es_975": es_975,
                        "realised_simple_return": realised_return,
                        "realised_loss": -realised_return,
                        "seed_components": [[5110, date.year, date.month] for date in dates],
                        "margin_fallback_count": 0,
                        "whole_vine_fallback": False,
                        "copula_log_score": (
                            np.sin(position / 5.0) + model_number + (0.02 if level == 10 else 0.0)
                        ),
                        "forecast_status": "ok",
                    }
                )
            )
    return pd.concat(frames[3], ignore_index=True), pd.concat(frames[10], ignore_index=True)


def refit_fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: dict[int, list[dict]] = {3: [], 10: []}
    for month in range(1, 5):
        for model_id, grouping_id, input_grouping in (
            ("M2", "gics", "gics_sector"),
            ("M4", "hierarchical", "hierarchical_cluster"),
        ):
            source_id = f"2020-{month:02d}:security_primary:{input_grouping}"
            shared = {
                "source_copula_refit_id": source_id,
                "matched_gaussian_refit_id": f"{source_id}:gaussian",
                "refit_date": pd.Timestamp(2020, month, 1),
                "year": 2020,
                "month": month,
                "universe_variant": "security_primary",
                "model_id": model_id,
                "grouping_id": grouping_id,
                "training_start": pd.Timestamp(2017, month, 1),
                "training_end": pd.Timestamp(2019, 12, 31),
                "training_observations": 750,
                "dimension": 11,
                "group_order_json": json.dumps([f"G{i}" for i in range(11)]),
                "fit_seed_components": [5110, 2020, month],
                "base_uniform_sha256": str(month) * 64,
            }
            for level, pair_count in ((3, 27), (10, 55)):
                rows[level].append(
                    {
                        **shared,
                        "truncation_level": level,
                        "pair_count": pair_count,
                        "number_parameters": 40.0 if level == 3 else 66.0,
                        "aic": -4_200.0 if level == 3 else -4_450.0,
                        "bic": -4_000.0 if level == 3 else -4_150.0,
                        "whole_vine_fallback": False,
                    }
                )
    return pd.DataFrame(rows[3]), pd.DataFrame(rows[10])


class FullVineRobustnessMethodTests(unittest.TestCase):
    def test_protocol_is_exact_and_bound_to_model_truncation_levels(self):
        robustness, model = load_configs()
        protocol = validate_robustness_protocol(robustness, model)
        self.assertEqual(protocol["evaluation"]["full_truncation_level"], 10)
        invalid = copy.deepcopy(robustness)
        invalid["dm"]["loss_difference"] = "truncated_vine_minus_full_vine"
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            validate_robustness_protocol(invalid, model)

    def test_forecast_pairing_rejects_seed_and_realised_return_mismatches(self):
        robustness, model = load_configs()
        protocol = validate_robustness_protocol(robustness, model)
        primary, full = forecast_fixtures()
        scores, validation = combine_robustness_forecasts(primary, full, protocol["evaluation"])
        self.assertEqual(tuple(scores.columns), DAILY_COLUMNS)
        self.assertEqual(set(scores["variant_id"]), set(VARIANT_ORDER))
        self.assertTrue(validation["common_random_numbers_verified"])

        bad_seed = full.copy()
        bad_seed.at[0, "seed_components"] = [5110, 2020, 2]
        with self.assertRaisesRegex(ValueError, "seed that does not match"):
            combine_robustness_forecasts(primary, bad_seed, protocol["evaluation"])

        bad_loss = full.copy()
        bad_loss.loc[0, ["realised_loss", "realised_simple_return"]] += [0.01, -0.01]
        with self.assertRaisesRegex(ValueError, "identical realised returns"):
            combine_robustness_forecasts(primary, bad_loss, protocol["evaluation"])

    def test_refit_pairing_requires_same_information_set_and_pair_counts(self):
        primary, full = refit_fixtures()
        primary["fit_seed_components"] = primary["fit_seed_components"].map(
            lambda values: np.asarray(values, dtype=np.int64)
        )
        full["fit_seed_components"] = full["fit_seed_components"].map(
            lambda values: np.asarray(values, dtype=np.int64)
        )
        validation = validate_refit_identity(primary, full, primary_level=3, full_level=10)
        self.assertEqual(validation["matched_refit_count"], 8)
        self.assertEqual(validation["primary_pair_count_total"], 8 * 27)
        self.assertEqual(validation["full_pair_count_total"], 8 * 55)

        bad_hash = full.copy()
        bad_hash.loc[0, "base_uniform_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "base_uniform_sha256"):
            validate_refit_identity(primary, bad_hash, primary_level=3, full_level=10)

    def test_complete_evaluator_returns_frozen_test_families_and_schema(self):
        robustness, model = load_configs()
        primary_forecasts, full_forecasts = forecast_fixtures()
        primary_refits, full_refits = refit_fixtures()
        scores, audit = build_robustness_outputs(
            primary_forecasts,
            full_forecasts,
            primary_refits,
            full_refits,
            robustness,
            model,
        )
        self.assertEqual(len(scores), 80 * 4)
        self.assertEqual(len(audit["full_period_calibration"]), 8)
        self.assertEqual(len(audit["diebold_mariano_comparisons"]), 6)
        self.assertEqual(audit["refit_complexity"]["mean_parameter_increase"], 26.0)
        self.assertFalse(audit["interpretation"]["primary_hypotheses_changed"])
        with (ROOT / "config/schemas/full_vine_robustness_daily_record.schema.json").open(
            encoding="utf-8"
        ) as handle:
            schema = json.load(handle)
        self.assertEqual(set(schema["required"]), set(DAILY_COLUMNS))
        self.assertEqual(set(schema["properties"]), set(DAILY_COLUMNS))
        json.dumps(audit, allow_nan=False)


@unittest.skipUnless(
    all(path.exists() for path in PRODUCTION_ARTIFACTS),
    "production full-vine robustness artifacts are not available in this checkout",
)
class ProductionFullVineRobustnessArtifactTests(unittest.TestCase):
    def test_current_full_vine_outputs_are_complete_and_bound(self):
        primary_refits = pd.read_parquet(PRODUCTION_ARTIFACTS[0])
        full_refits = pd.read_parquet(PRODUCTION_ARTIFACTS[3])
        full_forecasts = pd.read_parquet(PRODUCTION_ARTIFACTS[4])
        full_quality = json.loads(PRODUCTION_ARTIFACTS[5].read_text(encoding="utf-8"))
        scores = pd.read_parquet(PRODUCTION_ARTIFACTS[6])
        evaluation = json.loads(PRODUCTION_ARTIFACTS[7].read_text(encoding="utf-8"))
        self.assertEqual((len(primary_refits), len(full_refits)), (144, 144))
        self.assertEqual(set(primary_refits["pair_count"]), {27})
        self.assertEqual(set(full_refits["pair_count"]), {55})
        self.assertEqual(len(full_forecasts), 2 * 1_508)
        self.assertEqual(len(scores), 4 * 1_508)
        self.assertEqual(tuple(scores.columns), DAILY_COLUMNS)
        self.assertEqual(full_quality["status"], "pass")
        self.assertEqual(full_quality["analysis_role"], "robustness")
        self.assertEqual(evaluation["status"], "pass")
        self.assertEqual(len(evaluation["diebold_mariano_comparisons"]), 6)
        self.assertEqual(len(evaluation["full_period_calibration"]), 8)
        self.assertEqual(
            evaluation["outputs"]["full_vine_robustness_daily"]["sha256"],
            sha256_file(PRODUCTION_ARTIFACTS[6]),
        )


if __name__ == "__main__":
    unittest.main()
