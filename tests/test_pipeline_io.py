import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import scripts.build_annual_groupings as annual
import scripts.build_gaussian_copula as gaussian
import scripts.build_group_balanced_portfolios as balanced
import scripts.build_marginal_models as marginal
import scripts.build_vine_copula as vine
import scripts.evaluate_current_composition_historical_simulation as current_composition
import scripts.evaluate_group_balanced_robustness as balanced_risk
from scripts.pipeline_io import (
    PROJECT_ROOT,
    artifact_hash_issues,
    has_only_missing_artifact_hash_issues,
    project_path,
    reporting_scope,
    require_current_hash_records,
    sha256_file,
    temporary_sibling,
    write_csv_atomic,
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
                self.assertIs(module._require_current_hash_records, require_current_hash_records)
                self.assertIs(module._write_json_atomic, write_json_atomic)
                self.assertIs(module._write_parquet_atomic, write_parquet_atomic)
                self.assertEqual(module.PROJECT_ROOT, PROJECT_ROOT)
        for module in (balanced, balanced_risk, current_composition):
            with self.subTest(module=module.__name__):
                self.assertIs(module.project_path, project_path)
                self.assertIs(module.reporting_scope, reporting_scope)
                self.assertIs(module.require_current_hash_records, require_current_hash_records)
                self.assertIs(module.write_json_atomic, write_json_atomic)
                self.assertIs(module.write_parquet_atomic, write_parquet_atomic)
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

    def test_atomic_csv_preserves_column_order_and_replaces_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "licensed.csv"
            write_csv_atomic(
                path,
                [{"ticker": "OLD", "sector": "Old"}],
                fieldnames=("ticker", "sector"),
            )
            write_csv_atomic(
                path,
                [{"ticker": "ABC", "sector": "Industrials"}],
                fieldnames=("ticker", "sector"),
            )

            self.assertEqual(path.read_text(encoding="utf-8"), "ticker,sector\nABC,Industrials\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            self.assertEqual(list(root.glob(".*.part")), [])

    def test_atomic_writer_cleans_up_after_serialization_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(TypeError):
                write_json_atomic(root / "audit.json", {"invalid": object()})
            with self.assertRaises(ValueError):
                write_json_atomic(root / "audit.json", {"invalid": float("nan")})
            with self.assertRaises(ValueError):
                write_csv_atomic(
                    root / "rows.csv",
                    [{"expected": "value", "unexpected": "value"}],
                    fieldnames=("expected",),
                )
            self.assertEqual(list(root.glob(".*.part")), [])

    def test_temporary_sibling_cleans_up_failed_download_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "daily.csv.gz"
            with (
                self.assertRaisesRegex(RuntimeError, "simulated download failure"),
                temporary_sibling(destination) as staging,
            ):
                staging.write_bytes(b"partial")
                raise RuntimeError("simulated download failure")
            self.assertFalse(destination.exists())
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

    def test_hash_record_validation_detects_stale_missing_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "project"
            root.mkdir()
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"version-one")
            record = {
                "inputs": {
                    "artifact": {
                        "path": "artifact.bin",
                        "sha256": sha256_file(artifact),
                    }
                }
            }
            self.assertEqual(artifact_hash_issues(record, project_root=root), [])
            self.assertFalse(
                has_only_missing_artifact_hash_issues(
                    record,
                    allowed_missing_roots=(root,),
                    project_root=root,
                )
            )
            require_current_hash_records(record, project_root=root)

            artifact.write_bytes(b"version-two")
            issues = artifact_hash_issues(record, project_root=root, source_name="audit.json")
            self.assertEqual(len(issues), 1)
            self.assertIn("stale:artifact.bin", issues[0])
            self.assertFalse(
                has_only_missing_artifact_hash_issues(
                    record,
                    allowed_missing_roots=(root,),
                    project_root=root,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "artifact lineage check failed"):
                require_current_hash_records(record, project_root=root)

            invalid = {"path": "artifact.bin", "sha256": "not-a-digest"}
            missing = {"path": "missing.bin", "sha256": "0" * 64}
            self.assertIn(
                "payload:$:invalid_sha256",
                artifact_hash_issues(invalid, project_root=root),
            )
            self.assertIn(
                "payload:$:missing:missing.bin",
                artifact_hash_issues(missing, project_root=root),
            )
            self.assertTrue(
                has_only_missing_artifact_hash_issues(
                    missing,
                    allowed_missing_roots=(root,),
                    project_root=root,
                )
            )
            self.assertFalse(
                has_only_missing_artifact_hash_issues(
                    missing,
                    allowed_missing_roots=(root / "generated",),
                    project_root=root,
                )
            )
            self.assertFalse(
                has_only_missing_artifact_hash_issues(
                    invalid,
                    allowed_missing_roots=(root,),
                    project_root=root,
                )
            )
            self.assertFalse(
                has_only_missing_artifact_hash_issues(
                    [missing, invalid],
                    allowed_missing_roots=(root,),
                    project_root=root,
                )
            )

            outside = base / "outside.bin"
            outside.write_bytes(b"outside")
            escaped = {"path": "../outside.bin", "sha256": sha256_file(outside)}
            absolute = {"path": str(outside), "sha256": sha256_file(outside)}
            self.assertIn(
                "payload:$:external_path:../outside.bin",
                artifact_hash_issues(escaped, project_root=root),
            )
            self.assertIn(
                f"payload:$:external_path:{outside}",
                artifact_hash_issues(absolute, project_root=root),
            )
            self.assertFalse(
                has_only_missing_artifact_hash_issues(
                    escaped,
                    allowed_missing_roots=(root,),
                    project_root=root,
                )
            )
            self.assertFalse(
                has_only_missing_artifact_hash_issues(
                    absolute,
                    allowed_missing_roots=(root,),
                    project_root=root,
                )
            )
            with self.assertRaisesRegex(ValueError, "at least one"):
                has_only_missing_artifact_hash_issues(
                    missing,
                    allowed_missing_roots=(),
                    project_root=root,
                )
            with self.assertRaisesRegex(ValueError, "inside the project"):
                has_only_missing_artifact_hash_issues(
                    missing,
                    allowed_missing_roots=(outside.parent,),
                    project_root=root,
                )


if __name__ == "__main__":
    unittest.main()
