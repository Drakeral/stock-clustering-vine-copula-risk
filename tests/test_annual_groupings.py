import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_annual_groupings import build_annual_groupings
from scripts.research_methods import (
    adjusted_rand_index,
    average_linkage_clusters,
    correlation_distance,
    normalized_mutual_information,
)

ROOT = Path(__file__).resolve().parents[1]
CLUSTERING_ARTIFACTS = (
    ROOT / "data/processed/annual_group_assignments.json",
    ROOT / "data/processed/annual_group_returns.parquet",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ClusteringMethodTests(unittest.TestCase):
    def test_correlation_distance_and_average_linkage_are_deterministic(self):
        names = ["A", "B", "C", "D"]
        correlation = pd.DataFrame(
            [
                [1.0, 0.9, -0.2, -0.1],
                [0.9, 1.0, -0.1, -0.2],
                [-0.2, -0.1, 1.0, 0.8],
                [-0.1, -0.2, 0.8, 1.0],
            ],
            index=names,
            columns=names,
        )
        distance = correlation_distance(correlation)
        self.assertAlmostEqual(distance.loc["A", "B"], np.sqrt(0.05))
        labels, merges = average_linkage_clusters(distance, 2)
        self.assertEqual(labels["A"], labels["B"])
        self.assertEqual(labels["C"], labels["D"])
        self.assertNotEqual(labels["A"], labels["C"])
        permuted = distance.loc[list(reversed(names)), list(reversed(names))]
        repeated, _ = average_linkage_clusters(permuted, 2)
        self.assertEqual(labels, repeated)
        self.assertEqual(len(merges), 2)

    def test_ari_and_arithmetic_entropy_nmi_are_label_invariant(self):
        left = {"A": "one", "B": "one", "C": "two", "D": "two"}
        relabelled = {"A": "x", "B": "x", "C": "y", "D": "y"}
        crossed = {"A": "x", "B": "y", "C": "x", "D": "y"}
        self.assertAlmostEqual(adjusted_rand_index(left, relabelled), 1.0)
        self.assertAlmostEqual(normalized_mutual_information(left, relabelled), 1.0)
        self.assertLess(adjusted_rand_index(left, crossed), 1.0)
        self.assertLess(normalized_mutual_information(left, crossed), 1.0)


class AnnualGroupingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.training_dates = pd.to_datetime(
            [
                "2017-01-03",
                "2017-07-03",
                "2018-01-03",
                "2018-07-03",
                "2019-01-03",
                "2019-07-03",
            ]
        )
        self.evaluation_dates = pd.to_datetime(
            ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
        )
        index = self.training_dates.append(self.evaluation_dates)
        self.panel = pd.DataFrame(
            {
                "GOOG": [1, 2, 3, 4, 5, 6, 0.01, -0.02, 0.03, 0.00],
                "GOOGL": [1, 2, 3, 4, 5, 6, 0.02, -0.01, 0.04, 0.01],
                "C": [6, 5, 4, 3, 2, 1, -0.01, 0.03, -0.02, 0.02],
                "D": [6, 5, 4, 3, 2, 1, 0.00, 0.04, -0.01, 0.03],
            },
            index=index,
        )
        self.schedule = {
            "years": [
                {
                    "year": 2020,
                    "rebalance_date": "2020-01-02",
                    "security_count": 4,
                    "active_tickers": ["GOOG", "GOOGL", "C", "D"],
                }
            ]
        }
        self.gics = {"GOOG": "s1", "GOOGL": "s1", "C": "s2", "D": "s2"}
        self.config = {
            "evaluation_start_year": 2020,
            "evaluation_end_year": 2020,
            "training_window_calendar_years": 3,
            "minimum_paired_fraction": 0.8,
        }

    def build(self, panel):
        return build_annual_groupings(
            panel,
            self.schedule,
            self.gics,
            self.config,
            reporting_scope="provisional_research_results",
        )

    def test_evaluation_returns_cannot_change_training_only_assignments(self):
        assignments, group_returns, diagnostics = self.build(self.panel)
        altered = self.panel.copy()
        altered.loc[self.evaluation_dates, "GOOG"] = [0.10, -0.20, 0.30, -0.40]
        altered_assignments, _, _ = self.build(altered)
        first = assignments["years"][0]
        second = altered_assignments["years"][0]
        self.assertEqual(first["assignments"], second["assignments"])
        self.assertEqual(first["training_end_exclusive"], "2020-01-02")
        self.assertEqual(set(group_returns["grouping_id"]), {"gics_sector", "hierarchical_cluster"})
        self.assertEqual(
            set(group_returns["universe_variant"]),
            {"security_primary", "issuer_deduplicated_robustness"},
        )
        self.assertEqual(set(group_returns["sample_role"]), {"training", "evaluation"})
        self.assertEqual(diagnostics["reporting_scope"], "provisional_research_results")

    def test_both_groupings_reconstruct_the_same_portfolio(self):
        _, _, diagnostics = self.build(self.panel)
        errors = diagnostics["years"][0]["maximum_portfolio_identity_error"]
        self.assertLessEqual(errors["gics_sector"], 1e-12)
        self.assertLessEqual(errors["hierarchical_cluster"], 1e-12)
        issuer = diagnostics["years"][0]["issuer_deduplicated_robustness"]
        self.assertEqual(issuer["position_count"], 3)
        self.assertLessEqual(max(issuer["maximum_portfolio_identity_error"].values()), 1e-12)
        self.assertEqual(diagnostics["years"][0]["gics_dependence_gap"]["within_pair_count"], 2)
        self.assertEqual(diagnostics["years"][0]["gics_dependence_gap"]["between_pair_count"], 4)


@unittest.skipUnless(
    all(path.is_file() for path in CLUSTERING_ARTIFACTS),
    "requires locally generated clustering artifacts",
)
class CurrentClusteringArtifactTests(unittest.TestCase):
    def test_current_clustering_audit_and_outputs_are_bound(self):
        audit = json.loads((ROOT / "data/audit/clustering_diagnostics.json").read_text())
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["reporting_scope"], "provisional_research_results")
        self.assertEqual([row["year"] for row in audit["years"]], list(range(2020, 2026)))
        first = audit["years"][0]
        self.assertEqual(first["gics_dependence_gap"]["within_pair_count"], 551)
        self.assertEqual(first["gics_dependence_gap"]["between_pair_count"], 4_399)
        for row in audit["years"]:
            self.assertEqual(row["group_count"], 11)
            self.assertLessEqual(max(row["maximum_portfolio_identity_error"].values()), 1e-12)
            expected_positions = 99 if row["year"] == 2020 else 97
            issuer = row["issuer_deduplicated_robustness"]
            self.assertEqual(issuer["position_count"], expected_positions)
            self.assertLessEqual(max(issuer["maximum_portfolio_identity_error"].values()), 1e-12)
        for record in audit["outputs"].values():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(record["sha256"], sha256(path))
        group_returns = pd.read_parquet(ROOT / audit["outputs"]["annual_group_returns"]["path"])
        keys = [
            "date",
            "year",
            "universe_variant",
            "sample_role",
            "grouping_id",
            "group_id",
        ]
        self.assertFalse(group_returns.duplicated(keys).any())
        self.assertEqual(set(group_returns["sample_role"]), {"training", "evaluation"})
        self.assertTrue(np.isfinite(group_returns[["simple_return", "log_return"]]).all().all())
        first_year = audit["years"][0]
        self.assertEqual(first_year["training_observations"], 754)
        self.assertEqual(first_year["training_group_return_observations"], 753)


if __name__ == "__main__":
    unittest.main()
