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

from scripts.evaluate_ml_risk_models import (
    build_ml_evaluation_outputs,
    combine_ml_forecasts,
    validate_ml_risk_protocol,
)
from scripts.evaluate_risk_models import FORECAST_COLUMNS, score_forecasts
from scripts.pipeline_io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
ML_RISK_ARTIFACTS = (
    ROOT / "data/audit/ml_marginal_model_quality.json",
    ROOT / "data/audit/ml_gaussian_copula_quality.json",
    ROOT / "data/audit/ml_vine_copula_quality.json",
    ROOT / "data/audit/ml_model_evaluation.json",
    ROOT / "data/processed/ml_marginal_refits.parquet",
    ROOT / "data/processed/ml_marginal_daily_forecasts.parquet",
    ROOT / "data/processed/ml_monthly_copula_training_pits.parquet",
    ROOT / "data/processed/ml_gaussian_copula_refits.parquet",
    ROOT / "data/processed/ml_gaussian_risk_forecasts.parquet",
    ROOT / "data/processed/ml_vine_copula_refits.parquet",
    ROOT / "data/processed/ml_vine_risk_forecasts.parquet",
    ROOT / "data/processed/ml_risk_evaluation_daily.parquet",
)


def production_configs():
    with (ROOT / "config/ml_extension_config.toml").open("rb") as handle:
        ml_config = tomllib.load(handle)
    with (ROOT / "config/model_config.toml").open("rb") as handle:
        model_config = tomllib.load(handle)
    return ml_config, model_config


def forecast_frame(model_ids, dates, realised_loss):
    groupings = {
        "M0": "none",
        "M1": "gics",
        "M2": "gics",
        "M3": "hierarchical",
        "M4": "hierarchical",
        "M5": "spectral",
        "M6": "spectral",
        "M7": "pca_kmeans",
        "M8": "pca_kmeans",
    }
    rows = []
    for model_id in model_ids:
        model_number = int(model_id.removeprefix("M"))
        for date_number, (date, loss) in enumerate(zip(dates, realised_loss, strict=True)):
            shift = 0.00015 * model_number
            variation = 0.0002 * np.sin((date_number + 1) * (model_number + 1))
            var_95 = 0.012 + shift + variation
            var_975 = 0.017 + shift + variation
            var_99 = 0.022 + shift + variation
            rows.append(
                {
                    "date": date,
                    "model_id": model_id,
                    "grouping_id": groupings[model_id],
                    "refit_id": f"2020-01:{model_id}",
                    "var_95": var_95,
                    "var_975": var_975,
                    "var_99": var_99,
                    "es_975": 0.025 + shift + variation,
                    "realised_simple_return": -loss,
                    "realised_loss": loss,
                    "seed_components": None if model_id == "M0" else [5110, 2020, 1],
                    "margin_fallback_count": 0,
                    "whole_vine_fallback": False,
                    "copula_log_score": None if model_id == "M0" else 0.1 + shift,
                    "forecast_status": "ok",
                }
            )
    return pd.DataFrame(rows).loc[:, FORECAST_COLUMNS]


class MLRiskEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.ml_config, self.model_config = production_configs()
        self.dates = pd.bdate_range("2020-01-02", periods=80)
        index = np.arange(len(self.dates))
        self.realised_loss = 0.004 + 0.021 * ((index % 17) == 0) - 0.003 * np.sin(index)
        self.core_scores = score_forecasts(
            forecast_frame(("M0", "M1", "M2", "M3", "M4"), self.dates, self.realised_loss)
        )
        self.gaussian = forecast_frame(("M5", "M7"), self.dates, self.realised_loss)
        self.vine = forecast_frame(("M6", "M8"), self.dates, self.realised_loss)

    def test_protocol_and_complete_twenty_four_test_family(self):
        protocol = validate_ml_risk_protocol(self.ml_config, self.model_config)
        self.assertEqual(len(protocol["comparisons"]), 8)
        scores, audit = build_ml_evaluation_outputs(
            self.gaussian,
            self.vine,
            self.core_scores,
            self.ml_config,
            self.model_config,
        )
        self.assertEqual(len(scores), 4 * len(self.dates))
        self.assertEqual(set(scores["model_id"]), {"M5", "M6", "M7", "M8"})
        self.assertEqual(len(audit["diebold_mariano_comparisons"]), 24)
        self.assertEqual(len(audit["full_period_calibration_descriptive"]), 12)
        self.assertEqual(len(audit["annual_calibration_descriptive"]), 12)
        self.assertFalse(audit["interpretation"]["core_hypotheses_and_ranking_revised"])
        for row in audit["diebold_mariano_comparisons"]:
            self.assertGreaterEqual(row["bh_fdr_p_value"], row["p_value_two_sided"])

    def test_realised_loss_mismatch_fails_before_scoring(self):
        corrupted = self.vine.copy()
        corrupted.loc[0, "realised_loss"] += 0.01
        with self.assertRaisesRegex(ValueError, "negative simple return"):
            combine_ml_forecasts(
                self.gaussian,
                corrupted,
                self.core_scores,
                identity_tolerance=1e-12,
            )

    def test_missing_model_date_fails_instead_of_producing_nan_identity(self):
        incomplete = self.vine.iloc[1:].copy()
        with self.assertRaisesRegex(ValueError, "dates or model coverage"):
            combine_ml_forecasts(
                self.gaussian,
                incomplete,
                self.core_scores,
                identity_tolerance=1e-12,
            )

    def test_nonfinite_core_score_is_rejected_before_dm_inference(self):
        corrupted = self.core_scores.copy()
        corrupted.loc[corrupted["model_id"] == "M1", "quantile_loss_95"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite risk or loss"):
            combine_ml_forecasts(
                self.gaussian,
                self.vine,
                corrupted,
                identity_tolerance=1e-12,
            )

    def test_core_row_order_does_not_change_matched_inference(self):
        shuffled = self.core_scores.sample(frac=1.0, random_state=5110).reset_index(drop=True)
        _, validation = combine_ml_forecasts(
            self.gaussian,
            self.vine,
            shuffled,
            identity_tolerance=1e-12,
        )
        self.assertEqual(validation["evaluation_date_count"], len(self.dates))

    def test_protocol_change_is_rejected(self):
        changed = copy.deepcopy(self.ml_config)
        changed["exploratory_inference"]["family_size"] = 12
        with self.assertRaisesRegex(ValueError, "unsupported exploratory ML inference protocol"):
            validate_ml_risk_protocol(changed, self.model_config)

    def test_shared_schemas_cover_ml_models(self):
        forecast_schema = json.loads(
            (ROOT / "config/schemas/forecast_record.schema.json").read_text()
        )
        score_schema = json.loads(
            (ROOT / "config/schemas/risk_evaluation_daily_record.schema.json").read_text()
        )
        self.assertTrue(
            set(range(5, 9)).issubset(
                {
                    int(model_id[1:])
                    for model_id in forecast_schema["properties"]["model_id"]["enum"]
                    if model_id.startswith("M") and model_id[1:].isdigit()
                }
            )
        )
        self.assertTrue(
            set(range(5, 9)).issubset(
                {
                    int(model_id[1:])
                    for model_id in score_schema["properties"]["model_id"]["enum"]
                    if model_id.startswith("M") and model_id[1:].isdigit()
                }
            )
        )
        for schema in (forecast_schema, score_schema):
            observed = {}
            for rule in schema["allOf"]:
                model_rule = rule["if"]["properties"]["model_id"]
                models = [model_rule["const"]] if "const" in model_rule else model_rule["enum"]
                grouping = rule["then"]["properties"].get("grouping_id", {}).get("const")
                if grouping is not None:
                    observed.update(dict.fromkeys(models, grouping))
            self.assertEqual(
                observed,
                {
                    "M0": "none",
                    "M0_CC": "none",
                    "M1": "gics",
                    "M2": "gics",
                    "M3": "hierarchical",
                    "M4": "hierarchical",
                    "M5": "spectral",
                    "M6": "spectral",
                    "M7": "pca_kmeans",
                    "M8": "pca_kmeans",
                    "B_GICS_HS": "gics_balanced",
                    "B_GICS_GAUSSIAN": "gics_balanced",
                    "B_GICS_VINE": "gics_balanced",
                    "B_HIER_HS": "hierarchical_balanced",
                    "B_HIER_GAUSSIAN": "hierarchical_balanced",
                    "B_HIER_VINE": "hierarchical_balanced",
                },
            )


@unittest.skipUnless(
    all(path.is_file() for path in ML_RISK_ARTIFACTS),
    "requires locally generated M5--M8 artifacts",
)
class ProductionMLRiskArtifactTests(unittest.TestCase):
    def test_current_ml_risk_artifacts_are_complete_and_bound(self):
        audits = {
            name: json.loads((ROOT / f"data/audit/{name}.json").read_text())
            for name in (
                "ml_marginal_model_quality",
                "ml_gaussian_copula_quality",
                "ml_vine_copula_quality",
                "ml_model_evaluation",
            )
        }
        for audit in audits.values():
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["reporting_scope"], "provisional_research_results")
            self.assertEqual(audit["issues"], [])
            for record in [*audit["inputs"].values(), *audit["outputs"].values()]:
                self.assertEqual(record["sha256"], sha256_file(ROOT / record["path"]))

        marginal = audits["ml_marginal_model_quality"]
        gaussian = audits["ml_gaussian_copula_quality"]
        vine = audits["ml_vine_copula_quality"]
        evaluation = audits["ml_model_evaluation"]
        self.assertEqual(marginal["refit_count"], 1_584)
        self.assertEqual(marginal["ewma_fit_count"], 0)
        self.assertEqual(
            set(marginal["selected_method_counts"]), {"ar_garch_t", "ar_garch_t_retry"}
        )
        self.assertEqual((gaussian["refit_count"], gaussian["forecast_count"]), (144, 3_016))
        self.assertEqual((vine["refit_count"], vine["forecast_count"]), (144, 3_016))
        self.assertEqual(gaussian["common_random_number_month_count"], 72)
        self.assertEqual(vine["pair_count"], 144 * 27)
        self.assertEqual(vine["failed_pair_count"], 0)
        self.assertEqual(vine["whole_vine_fallback_refit_count"], 0)
        self.assertTrue(vine["eligible_for_exploratory_comparison"])
        self.assertFalse(vine["eligible_to_be_declared_best"])
        self.assertEqual(len(evaluation["diebold_mariano_comparisons"]), 24)
        self.assertFalse(
            any(
                row["significant_after_bh_fdr"] for row in evaluation["diebold_mariano_comparisons"]
            )
        )
        self.assertFalse(evaluation["interpretation"]["core_hypotheses_and_ranking_revised"])
        scores = pd.read_parquet(ROOT / "data/processed/ml_risk_evaluation_daily.parquet")
        self.assertEqual((len(scores), scores["date"].nunique()), (6_032, 1_508))
        self.assertEqual(set(scores["model_id"]), {"M5", "M6", "M7", "M8"})

        seed_manifest = json.loads(
            (ROOT / "data/manifests/simulation_seed_manifest.json").read_text()
        )
        seed_hashes = {
            (int(row["year"]), int(row["month"])): row["base_uniform_sha256"]
            for row in seed_manifest["records"]
        }
        self.assertEqual(len(seed_hashes), 72)
        for path, seed_column in (
            ("ml_gaussian_copula_refits.parquet", "seed_components"),
            ("ml_vine_copula_refits.parquet", "fit_seed_components"),
        ):
            refits = pd.read_parquet(ROOT / "data/processed" / path)
            for row in refits.itertuples(index=False):
                self.assertEqual(list(getattr(row, seed_column)), [5110, row.year, row.month])
                self.assertEqual(row.base_uniform_sha256, seed_hashes[(row.year, row.month)])

        core = pd.read_parquet(ROOT / "data/processed/risk_evaluation_daily.parquet")
        reference = (
            core.loc[core["model_id"] == "M0", ["date", "realised_loss"]]
            .set_index("date")["realised_loss"]
            .sort_index()
        )
        realised = scores.pivot(index="date", columns="model_id", values="realised_loss")
        identity_error = realised.sub(reference, axis="index").abs().to_numpy(dtype=float).max()
        self.assertLessEqual(float(identity_error), 1e-12)


if __name__ == "__main__":
    unittest.main()
