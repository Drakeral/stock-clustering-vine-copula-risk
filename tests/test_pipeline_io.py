import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import scripts.build_annual_groupings as annual
import scripts.build_gaussian_copula as gaussian
import scripts.build_marginal_models as marginal
import scripts.build_vine_copula as vine
from scripts.pipeline_io import (
    PROJECT_ROOT,
    project_path,
    reporting_scope,
    sha256_file,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)


class PipelineIoTests(unittest.TestCase):
    def test_all_modelling_stages_share_the_same_infrastructure_functions(self):
        for module in (annual, marginal, gaussian, vine):
            with self.subTest(module=module.__name__):
                self.assertIs(module._sha256, sha256_file)
                self.assertIs(module._project_path, project_path)
                self.assertIs(module._reporting_scope, reporting_scope)
                self.assertIs(module._write_json_atomic, write_json_atomic)
                self.assertIs(module._write_parquet_atomic, write_parquet_atomic)
                self.assertEqual(module.PROJECT_ROOT, PROJECT_ROOT)

    def test_atomic_writers_replace_outputs_without_partial_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "audit.json"
            parquet_path = root / "rows.parquet"
            write_json_atomic(json_path, {"version": 1})
            write_json_atomic(json_path, {"version": 2})
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), {"version": 2})
            write_parquet_atomic(parquet_path, pd.DataFrame({"value": [1, 2]}))
            pd.testing.assert_frame_equal(
                pd.read_parquet(parquet_path), pd.DataFrame({"value": [1, 2]})
            )
            self.assertEqual(json_path.stat().st_mode & 0o777, 0o644)
            self.assertEqual(parquet_path.stat().st_mode & 0o777, 0o644)
            self.assertEqual(list(root.glob("*.part")), [])
            self.assertEqual(list(root.glob(".*.part")), [])

    def test_atomic_text_and_indexed_parquet_preserve_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text_path = root / "report.md"
            parquet_path = root / "panel.parquet"
            frame = pd.DataFrame(
                {"A": [0.1, 0.2]},
                index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
            )
            frame.index.name = "date"

            write_text_atomic(text_path, "complete\n")
            write_parquet_atomic(
                parquet_path,
                frame,
                index=True,
                compression="zstd",
            )

            self.assertEqual(text_path.read_text(encoding="utf-8"), "complete\n")
            pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), frame)
            self.assertEqual(list(root.glob(".*.part")), [])

    def test_atomic_writer_cleans_up_after_serialization_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(TypeError):
                write_json_atomic(root / "audit.json", {"invalid": object()})
            with self.assertRaises(ValueError):
                write_json_atomic(root / "audit.json", {"invalid": float("nan")})
            self.assertEqual(list(root.glob(".*.part")), [])

    def test_reporting_scope_requires_consistent_gate_states(self):
        provisional = {
            "foundation_v2": {
                "status": "provisional_pass",
                "reporting_scope": "provisional_research_results",
            },
            "gates": {"modelling_readiness": {"status": "pass_provisional"}},
        }
        strict = {
            "foundation_v2": {"status": "pass", "reporting_scope": "confirmatory"},
            "gates": {"modelling_readiness": {"status": "pass"}},
        }
        self.assertEqual(reporting_scope(provisional), "provisional_research_results")
        self.assertEqual(reporting_scope(strict), "confirmatory")

        inconsistent = dict(provisional)
        inconsistent["gates"] = {"modelling_readiness": {"status": "pass"}}
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            reporting_scope(inconsistent)


if __name__ == "__main__":
    unittest.main()
