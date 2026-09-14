import copy
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from scripts.build_ml_groupings import (
    GROUPING_IDS,
    build_ml_groupings,
    validate_ml_protocol,
)
from scripts.ml_clustering_methods import (
    canonical_cluster_labels,
    deterministic_kmeans,
    labels_sha256,
    pca_correlation_profile_embedding,
    spectral_embedding,
)

ROOT = Path(__file__).resolve().parents[1]
ML_ARTIFACTS = (
    ROOT / "data/audit/ml_clustering_diagnostics.json",
    ROOT / "data/processed/ml_annual_group_assignments.json",
    ROOT / "data/processed/ml_annual_group_returns.parquet",
    ROOT / "data/manifests/ml_clustering_seed_manifest.json",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_config():
    with (ROOT / "config/ml_extension_config.toml").open("rb") as handle:
        return tomllib.load(handle)


class MLMethodTests(unittest.TestCase):
    def setUp(self):
        self.names = ["A", "B", "C", "D", "E", "F"]
        self.correlation = np.array(
            [
                [1.00, 0.95, 0.10, 0.08, -0.10, -0.12],
                [0.95, 1.00, 0.09, 0.07, -0.11, -0.13],
                [0.10, 0.09, 1.00, 0.94, 0.05, 0.04],
                [0.08, 0.07, 0.94, 1.00, 0.03, 0.02],
                [-0.10, -0.11, 0.05, 0.03, 1.00, 0.93],
                [-0.12, -0.13, 0.04, 0.02, 0.93, 1.00],
            ]
        )

    def _labels(self, embedding, method_code):
        result = deterministic_kmeans(
            embedding,
            3,
            seed_components=[5110, 2020, method_code],
            n_init=20,
        )
        return canonical_cluster_labels(self.names, result.labels, prefix="test")

    def assert_pair_partition(self, labels):
        self.assertEqual(labels["A"], labels["B"])
        self.assertEqual(labels["C"], labels["D"])
        self.assertEqual(labels["E"], labels["F"])
        self.assertEqual(len(set(labels.values())), 3)

    def test_spectral_and_pca_embeddings_recover_block_structure(self):
        spectral = spectral_embedding(self.correlation, 3)
        pca = pca_correlation_profile_embedding(self.correlation)
        self.assert_pair_partition(self._labels(spectral.values, 1))
        self.assert_pair_partition(self._labels(pca.values, 2))
        self.assertGreater(spectral.bandwidth, 0)
        self.assertGreaterEqual(pca.component_count, 2)
        self.assertGreaterEqual(pca.cumulative_explained_variance, 0.8)

    def test_kmeans_seed_is_reproducible(self):
        values = spectral_embedding(self.correlation, 3).values
        first = deterministic_kmeans(values, 3, seed_components=[5110, 2020, 1], n_init=20)
        second = deterministic_kmeans(values, 3, seed_components=[5110, 2020, 1], n_init=20)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_allclose(first.centers, second.centers, rtol=0, atol=0)
        self.assertEqual(first.inertia, second.inertia)
        self.assertEqual(first.selected_initialization, second.selected_initialization)

    def test_invalid_correlation_is_rejected(self):
        invalid = self.correlation.copy()
        invalid[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite and symmetric"):
            spectral_embedding(invalid, 3)


class MLProtocolTests(unittest.TestCase):
    def test_production_protocol_is_exact_and_frozen(self):
        config = production_config()
        self.assertIs(validate_ml_protocol(config), config)
        changed = copy.deepcopy(config)
        changed["kmeans"]["n_init"] = 10
        with self.assertRaisesRegex(ValueError, "unsupported ML k-means protocol"):
            validate_ml_protocol(changed)

    def test_output_schemas_bind_all_produced_fields(self):
        return_schema = json.loads(
            (ROOT / "config/schemas/ml_group_return_record.schema.json").read_text()
        )
        seed_schema = json.loads(
            (ROOT / "config/schemas/ml_clustering_seed_record.schema.json").read_text()
        )
        self.assertFalse(return_schema["additionalProperties"])
        self.assertFalse(seed_schema["additionalProperties"])
        self.assertEqual(set(return_schema["required"]), set(return_schema["properties"]))
        self.assertEqual(set(seed_schema["required"]), set(seed_schema["properties"]))


class MLGroupingPipelineTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.Generator(np.random.PCG64DXSM(12345))
        training_dates = pd.bdate_range("2019-01-02", periods=60)
        evaluation_dates = pd.bdate_range("2020-01-02", periods=8)
        factors = generator.normal(0, 0.01, size=(len(training_dates), 3))
        training = np.column_stack(
            [
                factors[:, 0] + generator.normal(0, 0.001, len(training_dates)),
                factors[:, 0] + generator.normal(0, 0.001, len(training_dates)),
                factors[:, 1] + generator.normal(0, 0.001, len(training_dates)),
                factors[:, 1] + generator.normal(0, 0.001, len(training_dates)),
                factors[:, 2] + generator.normal(0, 0.001, len(training_dates)),
                factors[:, 2] + generator.normal(0, 0.001, len(training_dates)),
            ]
        )
        evaluation = generator.normal(0, 0.01, size=(len(evaluation_dates), 6))
        self.names = ["A", "B", "C", "D", "E", "F"]
        self.panel = pd.DataFrame(
            np.vstack([training, evaluation]),
            index=training_dates.append(evaluation_dates),
            columns=self.names,
        )
        self.evaluation_dates = evaluation_dates
        self.schedule = {
            "years": [
                {
                    "year": 2020,
                    "rebalance_date": "2020-01-02",
                    "security_count": 6,
                    "active_tickers": self.names,
                }
            ]
        }
        self.gics = dict(zip(self.names, ["s1", "s1", "s2", "s2", "s3", "s3"], strict=True))
        self.hierarchical = {
            2020: dict(zip(self.names, ["h1", "h1", "h2", "h2", "h3", "h3"], strict=True))
        }
        self.config = production_config()
        self.config["grouping"]["evaluation_end_year"] = 2020
        self.config["grouping"]["minimum_training_observations"] = 40

    def build(self, panel):
        return build_ml_groupings(
            panel,
            self.schedule,
            self.gics,
            self.hierarchical,
            self.config,
            reporting_scope_value="provisional_research_results",
        )

    def test_evaluation_returns_cannot_change_assignments(self):
        assignments, returns, diagnostics, seeds = self.build(self.panel)
        altered = self.panel.copy()
        altered.loc[self.evaluation_dates] *= -20
        altered_assignments, _, _, altered_seeds = self.build(altered)
        self.assertEqual(
            assignments["years"][0]["assignments"],
            altered_assignments["years"][0]["assignments"],
        )
        self.assertEqual(seeds["records"], altered_seeds["records"])
        self.assertEqual(set(returns["grouping_id"]), set(GROUPING_IDS))
        self.assertEqual(set(returns["sample_role"]), {"training", "evaluation"})
        self.assertEqual(diagnostics["reporting_scope"], "provisional_research_results")

    def test_each_ml_grouping_reconstructs_primary_portfolio(self):
        _, returns, diagnostics, _ = self.build(self.panel)
        year = diagnostics["years"][0]
        for grouping_id in GROUPING_IDS:
            method = year["methods"][grouping_id]
            self.assertEqual(len(method["cluster_sizes"]), 3)
            self.assertLessEqual(method["maximum_portfolio_identity_error"], 1e-12)
        keys = ["date", "year", "sample_role", "grouping_id", "group_id"]
        self.assertFalse(returns.duplicated(keys).any())
        self.assertTrue(np.isfinite(returns[["simple_return", "log_return"]]).all().all())


@unittest.skipUnless(
    all(path.is_file() for path in ML_ARTIFACTS),
    "requires locally generated ML-grouping artifacts",
)
class CurrentMLGroupingArtifactTests(unittest.TestCase):
    def test_current_artifacts_are_complete_and_bound(self):
        audit = json.loads((ROOT / "data/audit/ml_clustering_diagnostics.json").read_text())
        assignments = json.loads(
            (ROOT / "data/processed/ml_annual_group_assignments.json").read_text()
        )
        seeds = json.loads((ROOT / "data/manifests/ml_clustering_seed_manifest.json").read_text())
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual([row["year"] for row in audit["years"]], list(range(2020, 2026)))
        self.assertEqual(assignments["grouping_ids"], list(GROUPING_IDS))
        self.assertEqual(len(seeds["records"]), 12)
        assignment_years = {int(row["year"]): row for row in assignments["years"]}
        method_codes = {"spectral_cluster": 1, "pca_kmeans_cluster": 2}
        seed_schema = json.loads(
            (ROOT / "config/schemas/ml_clustering_seed_record.schema.json").read_text()
        )
        for record in seeds["records"]:
            self.assertEqual(set(record), set(seed_schema["properties"]))
            self.assertEqual(
                record["seed_components"],
                [5110, record["year"], method_codes[record["grouping_id"]]],
            )
            labels = {
                row["ticker"]: row[record["grouping_id"]]
                for row in assignment_years[record["year"]]["assignments"]
            }
            self.assertEqual(record["labels_sha256"], labels_sha256(labels))
        for row in audit["years"]:
            self.assertEqual(row["group_count"], 11)
            for grouping_id in GROUPING_IDS:
                method = row["methods"][grouping_id]
                self.assertEqual(len(method["cluster_sizes"]), 11)
                self.assertLessEqual(method["maximum_portfolio_identity_error"], 1e-12)
        for record in [*audit["inputs"].values(), *audit["outputs"].values()]:
            path = ROOT / record["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(record["sha256"], sha256(path))
        returns = pd.read_parquet(ROOT / audit["outputs"]["ml_annual_group_returns"]["path"])
        schema = json.loads(
            (ROOT / "config/schemas/ml_group_return_record.schema.json").read_text()
        )
        self.assertEqual(set(returns.columns), set(schema["properties"]))
        keys = ["date", "year", "sample_role", "grouping_id", "group_id"]
        self.assertFalse(returns.duplicated(keys).any())

        stock_returns = pd.read_parquet(
            ROOT / "data/processed/portfolio_constituent_simple_returns.parquet"
        )
        schedule = json.loads((ROOT / "data/processed/active_universe_by_year.json").read_text())
        active_by_year = {
            int(row["year"]): list(row["active_tickers"]) for row in schedule["years"]
        }
        for year in range(2020, 2026):
            direct = stock_returns.loc[stock_returns.index.year == year, active_by_year[year]].mean(
                axis=1
            )
            for grouping_id in GROUPING_IDS:
                rows = returns.loc[
                    (returns["year"] == year)
                    & (returns["sample_role"] == "evaluation")
                    & (returns["grouping_id"] == grouping_id)
                ]
                grouped = rows.pivot(index="date", columns="group_id", values="simple_return")
                weights = (
                    rows[["group_id", "portfolio_weight"]]
                    .drop_duplicates()
                    .set_index("group_id")["portfolio_weight"]
                )
                reconstructed = grouped.mul(weights, axis="columns").sum(axis=1)
                error = float((direct - reconstructed).abs().max())
                self.assertLessEqual(error, 1e-12)


if __name__ == "__main__":
    unittest.main()
