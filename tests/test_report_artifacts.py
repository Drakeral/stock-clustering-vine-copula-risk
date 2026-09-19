import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_report_artifacts import (
    build_calibration_rows,
    build_clustering_rows,
    build_dm_rows,
    build_hypothesis_rows,
    build_model_performance_rows,
    build_report_artifacts,
    load_reporting_config,
)
from scripts.pipeline_io import artifact_hash_issues, require_current_hash_records

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/reporting_config.toml"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class ReportArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_reporting_config(CONFIG_PATH)
        cls.core = load_json("data/audit/model_evaluation.json")
        cls.clustering = load_json("data/audit/clustering_diagnostics.json")
        cls.ml_clustering = load_json("data/audit/ml_clustering_diagnostics.json")
        cls.ml_evaluation = load_json("data/audit/ml_model_evaluation.json")

    def test_frozen_config_and_core_tables_match_audited_decisions(self):
        labels = self.config["model_labels"]
        core_rows = build_model_performance_rows(
            self.core,
            self.config["core_model_order"],
            labels,
            ml=False,
        )
        ml_rows = build_model_performance_rows(
            self.ml_evaluation,
            self.config["ml_model_order"],
            labels,
            ml=True,
        )
        self.assertEqual([row["model_id"] for row in core_rows], [f"M{i}" for i in range(5)])
        self.assertEqual([row["model_id"] for row in ml_rows], [f"M{i}" for i in range(5, 9)])
        m4 = next(row for row in core_rows if row["model_id"] == "M4")
        self.assertEqual(m4["overall_rank"], 1.0)
        self.assertAlmostEqual(m4["average_primary_score_rank"], 4.0 / 3.0)
        self.assertFalse(m4["systematic_calibration_failure"])
        self.assertEqual(len(build_dm_rows(self.core)), 6)
        self.assertEqual(len(build_calibration_rows(self.core)), 10)
        self.assertEqual(
            [row["decision"] for row in build_hypothesis_rows(self.core)],
            ["not_supported", "no_support", "supported"],
        )

    def test_clustering_table_preserves_pair_counts_and_all_methods(self):
        rows = build_clustering_rows(
            self.clustering,
            self.ml_clustering,
            self.config["method_labels"],
        )
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["year"] for row in rows}, set(range(2020, 2026)))
        self.assertEqual(
            {row["method_id"] for row in rows},
            {"gics", "hierarchical", "spectral", "pca_kmeans"},
        )
        gics_2020 = next(row for row in rows if row["year"] == 2020 and row["method_id"] == "gics")
        self.assertEqual(gics_2020["within_pair_count"], 551)
        self.assertEqual(gics_2020["between_pair_count"], 4399)

    def test_report_config_rejects_scope_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reporting_config.toml"
            text = CONFIG_PATH.read_text(encoding="utf-8").replace(
                'reporting_scope = "provisional_research_results"',
                'reporting_scope = "confirmatory"',
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reporting_scope"):
                load_reporting_config(path)

    def test_checked_in_report_manifest_binds_current_outputs(self):
        manifest = load_json("reports/report_manifest.json")
        self.assertEqual(manifest["status"], "pass")
        self.assertEqual(manifest["reporting_scope"], "provisional_research_results")
        self.assertFalse(manifest["quality"]["new_inference_performed"])
        self.assertFalse(manifest["quality"]["model_refits_performed"])
        self.assertEqual(manifest["generator"]["path"], "scripts/build_report_artifacts.py")
        self.assertEqual(sum(len(rows) for rows in manifest["outputs"].values()), 11)
        require_current_hash_records(manifest, project_root=ROOT, source_name="report_manifest")

    def test_full_report_generation_is_byte_reproducible_when_inputs_exist(self):
        source_payloads = [load_json(path) for path in self.config["sources"].values()]
        issues = [
            issue
            for index, payload in enumerate(source_payloads)
            for issue in artifact_hash_issues(
                payload,
                project_root=ROOT,
                source_name=f"report_source_{index}",
            )
        ]
        if issues:
            self.skipTest("ignored upstream modelling artifacts are unavailable in this clone")
        with (
            tempfile.TemporaryDirectory(dir=ROOT) as first_directory,
            tempfile.TemporaryDirectory(dir=ROOT) as second_directory,
        ):
            first = build_report_artifacts(ROOT, output_root=Path(first_directory))
            second = build_report_artifacts(ROOT, output_root=Path(second_directory))
            first_hashes = {
                Path(record["path"]).name: record["sha256"]
                for records in first["outputs"].values()
                for record in records
            }
            second_hashes = {
                Path(record["path"]).name: record["sha256"]
                for records in second["outputs"].values()
                for record in records
            }
            self.assertEqual(first_hashes, second_hashes)

    def test_model_table_fails_on_missing_frozen_model(self):
        incomplete = copy.deepcopy(self.core)
        incomplete["model_summaries"] = incomplete["model_summaries"][:-1]
        with self.assertRaisesRegex(ValueError, "frozen order"):
            build_model_performance_rows(
                incomplete,
                self.config["core_model_order"],
                self.config["model_labels"],
                ml=False,
            )


if __name__ == "__main__":
    unittest.main()
