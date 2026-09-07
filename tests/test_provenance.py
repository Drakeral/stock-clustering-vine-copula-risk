import copy
import csv
import gzip
import hashlib
import json
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from scripts.download_market_data import authenticated_json
from scripts.generate_run_manifest import (
    build_run_manifest,
    default_config_paths,
    default_output_paths,
    git_state,
)
from scripts.prepare_universe import (
    EXPECTED_SP100_REVISION,
    EXPECTED_SP500_REVISION,
    LICENSED_COLUMNS,
    read_licensed_universe,
    reconcile_licensed_universe,
    revision_record,
)
from scripts.verify_foundation_inputs import (
    SPECIAL_CLOSURES,
    FrozenInputSpec,
    canonical_json_bytes,
    sha256_file,
    verify_foundation_inputs,
    xnys_sessions,
)

ROOT = Path(__file__).resolve().parents[1]


class UniverseProvenanceTests(unittest.TestCase):
    def test_cached_mediawiki_payloads_match_frozen_revisions(self):
        sources = [
            ("sp100_oldid_929329274.json", EXPECTED_SP100_REVISION, "S&P 100"),
            ("sp500_oldid_933762580.json", EXPECTED_SP500_REVISION, "List of S&P 500 companies"),
        ]
        for filename, expected_revision, expected_title in sources:
            metadata, content = revision_record(ROOT / "data/raw/universe_sources" / filename)
            self.assertEqual(metadata["revision_id"], expected_revision)
            self.assertEqual(metadata["page_title"], expected_title)
            self.assertEqual(len(metadata["payload_sha256"]), 64)
            self.assertGreater(len(content), 1000)

    def test_licensed_schema_and_reconciliation_are_strict(self):
        secondary = [
            {
                "ticker": "ABC",
                "gics_sector": "Industrials",
                "gics_sub_industry": "Machinery",
            }
        ]
        licensed_row = {
            "security_id": "SEC:ABC",
            "issuer_id": "ISSUER:ABC",
            "ticker": "ABC",
            "share_class": "common",
            "company_name": "ABC Corp",
            "gics_sector": "Industrials",
            "gics_sub_industry": "Machinery",
            "membership_date": "2019-01-01",
            "as_of_date": "2020-01-02",
            "source_record_id": "vendor:1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "licensed.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=LICENSED_COLUMNS)
                writer.writeheader()
                writer.writerow(licensed_row)
            licensed = read_licensed_universe(path)
        self.assertEqual(
            reconcile_licensed_universe(secondary, licensed, "2020-01-02")["status"], "pass"
        )
        licensed[0]["gics_sector"] = "Materials"
        self.assertEqual(
            reconcile_licensed_universe(secondary, licensed, "2020-01-02")["status"],
            "blocked_unresolved_differences",
        )
        self.assertEqual(
            reconcile_licensed_universe(secondary, None, "2020-01-02")["status"],
            "blocked_authorized_extract_absent",
        )

    def test_licensed_rows_reject_invalid_dates_and_duplicate_identifiers(self):
        row = {
            "security_id": "SEC:ABC",
            "issuer_id": "ISSUER:ABC",
            "ticker": "ABC",
            "share_class": "common",
            "company_name": "ABC Corp",
            "gics_sector": "Industrials",
            "gics_sub_industry": "Machinery",
            "membership_date": "2020-01-03",
            "as_of_date": "2020-01-02",
            "source_record_id": "vendor:1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "licensed.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=LICENSED_COLUMNS)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "membership_date after as_of_date"):
                read_licensed_universe(path)

            row["membership_date"] = "2019-01-01"
            duplicate = {**row, "ticker": "XYZ", "source_record_id": "vendor:2"}
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=LICENSED_COLUMNS)
                writer.writeheader()
                writer.writerows([row, duplicate])
            with self.assertRaisesRegex(ValueError, "duplicate security_id"):
                read_licensed_universe(path)

    def test_current_universe_audit_and_manifest_are_consistent(self):
        audit = json.loads((ROOT / "data/audit/universe_provenance.json").read_text())
        manifest = json.loads((ROOT / "data/manifests/universe_source_manifest.json").read_text())
        licensed_status = manifest["licensed_corroboration"]["status"]
        if audit["status"] == "pass":
            self.assertEqual(licensed_status, "pass")
            self.assertEqual(audit["blockers"], [])
        else:
            self.assertEqual(audit["status"], "blocked")
            self.assertTrue(licensed_status.startswith("blocked_"))
            self.assertTrue(audit["blockers"])
        self.assertNotIn("missing_from_licensed", audit["reconciliation"])
        self.assertNotIn("field_differences", audit["reconciliation"])
        self.assertTrue(all(source["payload_sha256"] for source in manifest["sources"]))


class MarketInputIntegrityTests(unittest.TestCase):
    def test_independent_calendar_matches_frozen_session_count(self):
        import datetime as dt

        sessions = xnys_sessions(dt.date(2017, 1, 3), dt.date(2025, 12, 31))
        self.assertEqual(len(sessions), 2262)
        self.assertNotIn(dt.date(2018, 12, 5), sessions)
        self.assertNotIn(dt.date(2025, 1, 9), sessions)
        self.assertIn(dt.date(2021, 12, 31), sessions)
        self.assertEqual(len(SPECIAL_CLOSURES), 2)

    def test_tiny_manifest_verifies_hashes_rows_references_and_calendar(self):
        expected_columns = [
            "ticker",
            "volume",
            "open",
            "close",
            "high",
            "low",
            "window_start",
            "transactions",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daily = root / "data/raw/market/day_aggs/2020/01/2020-01-02.csv.gz"
            daily.parent.mkdir(parents=True)
            with gzip.open(daily, "wt", encoding="utf-8", newline="") as handle:
                handle.write(",".join(expected_columns) + "\n")
                handle.write("ABC,1,10,10,10,10,1577941200000000000,1\n")
            universe = root / "universe.json"
            universe.write_text(json.dumps({"constituents": [{"ticker": "ABC"}]}))
            master = root / "master.json"
            master.write_text(json.dumps({"mappings": {}}))
            reference_root = root / "data/raw/market/reference"
            reference_records = []
            for event_type in ("splits", "dividends"):
                path = reference_root / event_type / "ABC.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "schema_version": 1,
                    "provider": "fixture",
                    "ticker": "ABC",
                    "event_type": event_type,
                    "sample_start": "2020-01-02",
                    "sample_end": "2020-01-02",
                    "results": [],
                }
                path.write_text(json.dumps(payload))
                reference_records.append(
                    {
                        "ticker": "ABC",
                        "event_type": event_type,
                        "local_path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                        "rows": 0,
                        "payload_bytes": path.stat().st_size,
                        "schema_version": 1,
                        "sample_start": "2020-01-02",
                        "sample_end": "2020-01-02",
                        "status": "verified",
                    }
                )
            manifest = {
                "schema_version": 2,
                "provider": "fixture",
                "sample_start": "2020-01-02",
                "sample_end": "2020-01-02",
                "daily_file_count": 1,
                "daily_compressed_bytes": daily.stat().st_size,
                "daily_files": [
                    {
                        "date": "2020-01-02",
                        "key": "fixture/day_aggs_v1/2020/01/2020-01-02.csv.gz",
                        "local_path": daily.relative_to(root).as_posix(),
                        "sha256": sha256_file(daily),
                        "row_count": 1,
                        "size": daily.stat().st_size,
                        "columns": sorted(expected_columns),
                        "status": "verified",
                    }
                ],
                "reference_status": "complete",
                "reference_file_count": 2,
                "reference_downloads": reference_records,
            }
            frozen = FrozenInputSpec(
                provider="fixture",
                manifest_schema_version=2,
                sample_start="2020-01-02",
                sample_end="2020-01-02",
                daily_aggregate_prefix="fixture/day_aggs_v1",
                licensed_market_data="data/raw/market",
            )
            public, audit = verify_foundation_inputs(
                root, manifest, universe, master, reference_root, frozen
            )
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(public["daily_file_count"], 1)
            self.assertEqual(
                audit["public_manifest_sha256"],
                hashlib.sha256(canonical_json_bytes(public)).hexdigest(),
            )
            bad_hash = copy.deepcopy(manifest)
            bad_hash["daily_files"][0]["sha256"] = "0" * 64
            _, bad_audit = verify_foundation_inputs(
                root, bad_hash, universe, master, reference_root, frozen
            )
            self.assertEqual(bad_audit["status"], "blocked")
            self.assertTrue(any("sha256 differs" in row["reason"] for row in bad_audit["errors"]))

            missing_attestation = copy.deepcopy(manifest)
            del missing_attestation["daily_files"][0]["row_count"]
            del missing_attestation["reference_downloads"][0]["sha256"]
            _, missing_audit = verify_foundation_inputs(
                root, missing_attestation, universe, master, reference_root, frozen
            )
            self.assertTrue(
                any(
                    "missing required fields: row_count" in row["reason"]
                    for row in missing_audit["errors"]
                )
            )
            self.assertTrue(
                any(
                    "missing required fields: sha256" in row["reason"]
                    for row in missing_audit["errors"]
                )
            )

            duplicate_reference = copy.deepcopy(manifest)
            duplicate_reference["reference_downloads"].append(
                copy.deepcopy(duplicate_reference["reference_downloads"][0])
            )
            duplicate_reference["reference_file_count"] = 3
            _, duplicate_audit = verify_foundation_inputs(
                root, duplicate_reference, universe, master, reference_root, frozen
            )
            self.assertTrue(
                any(
                    "duplicate_identity:ABC:splits" in row["reason"]
                    for row in duplicate_audit["errors"]
                )
            )
            self.assertTrue(
                any(
                    row["reason"] == "reference_count_mismatch" for row in duplicate_audit["errors"]
                )
            )

            wrong_identity = copy.deepcopy(manifest)
            wrong_identity["provider"] = "unfrozen-provider"
            wrong_identity["sample_end"] = "2020-01-03"
            wrong_identity["daily_files"][0]["key"] = "fixture/wrong-key.csv.gz"
            _, identity_audit = verify_foundation_inputs(
                root, wrong_identity, universe, master, reference_root, frozen
            )
            reasons = [row["reason"] for row in identity_audit["errors"]]
            self.assertTrue(any(reason.startswith("provider_mismatch") for reason in reasons))
            self.assertTrue(any(reason.startswith("sample_end_mismatch") for reason in reasons))
            self.assertIn("key differs from canonical provider key", reasons)

            wrong_content_date = copy.deepcopy(manifest)
            with gzip.open(daily, "wt", encoding="utf-8", newline="") as handle:
                handle.write(",".join(expected_columns) + "\n")
                handle.write("ABC,1,10,10,10,10,1578027600000000000,1\n")
            wrong_content_date["daily_files"][0]["sha256"] = sha256_file(daily)
            wrong_content_date["daily_files"][0]["size"] = daily.stat().st_size
            wrong_content_date["daily_compressed_bytes"] = daily.stat().st_size
            _, content_audit = verify_foundation_inputs(
                root, wrong_content_date, universe, master, reference_root, frozen
            )
            self.assertTrue(
                any(
                    "content date 2020-01-03 differs" in row["reason"]
                    for row in content_audit["errors"]
                )
            )

    def test_current_full_input_audit_passes(self):
        audit = json.loads((ROOT / "data/audit/market_input_integrity.json").read_text())
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["checks"]["verified_daily_files"], 2262)
        self.assertEqual(audit["checks"]["verified_reference_files"], 210)
        manifest = json.loads((ROOT / "data/manifests/market_input_manifest.json").read_text())
        self.assertEqual(manifest["verification_status"], "pass")


class RunManifestTests(unittest.TestCase):
    def test_default_configuration_capture_includes_nested_schemas(self):
        paths = {path.as_posix() for path in default_config_paths(ROOT)}
        self.assertIn("config/model_config.toml", paths)
        self.assertIn("config/schemas/forecast_record.schema.json", paths)

    def test_default_outputs_cover_panels_and_separate_gate_audits(self):
        paths = {path.as_posix() for path in default_output_paths()}
        self.assertIn("data/processed/daily_simple_total_returns.parquet", paths)
        self.assertIn("data/processed/daily_total_returns.parquet", paths)
        self.assertIn("data/audit/current_gate_status.json", paths)
        self.assertIn("data/audit/portfolio_arithmetic.json", paths)
        self.assertIn("data/audit/clustering_diagnostics.json", paths)
        self.assertIn("data/audit/marginal_model_quality.json", paths)
        self.assertIn("data/processed/annual_group_assignments.json", paths)
        self.assertIn("data/processed/annual_group_returns.parquet", paths)
        self.assertIn("data/processed/marginal_refits.parquet", paths)
        self.assertIn("data/processed/marginal_daily_forecasts.parquet", paths)
        self.assertIn("data/processed/monthly_copula_training_pits.parquet", paths)
        self.assertIn("data/processed/gaussian_copula_refits.parquet", paths)
        self.assertIn("data/processed/gaussian_risk_forecasts.parquet", paths)
        self.assertIn("data/manifests/simulation_seed_manifest.json", paths)
        self.assertIn("data/audit/gaussian_copula_quality.json", paths)

    def test_git_state_counts_untracked_files_as_dirty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            self.assertTrue(git_state(root)["worktree_clean"])
            (root / "untracked.txt").write_text("x", encoding="utf-8")
            state = git_state(root)
            self.assertFalse(state["commit_present"])
            self.assertFalse(state["worktree_clean"])

    def test_missing_output_marks_run_manifest_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text(
                '[project]\nname="fixture"\nversion="0"\ndependencies=[]\n',
                encoding="utf-8",
            )
            (root / ".python-version").write_text("3.12\n", encoding="utf-8")
            (root / "uv.lock").write_text("fixture\n", encoding="utf-8")
            manifest = build_run_manifest(root, [], [], [Path("missing.parquet")])
            self.assertFalse(manifest["completeness"]["outputs"])
            self.assertFalse(manifest["completeness"]["complete"])


class CredentialSafetyTests(unittest.TestCase):
    def test_rest_failure_never_exposes_api_key(self):
        secret = "do-not-print-this-key"
        error = urllib.error.HTTPError(
            f"https://api.massive.com/v3/reference/splits?apiKey={secret}",
            401,
            "unauthorized",
            hdrs=None,
            fp=None,
        )
        with (
            mock.patch("scripts.download_market_data.urllib.request.urlopen", side_effect=error),
            self.assertRaises(RuntimeError) as raised,
        ):
            authenticated_json(
                f"https://api.massive.com/v3/reference/splits?apiKey={secret}",
                secret,
                attempts=1,
            )
        self.assertNotIn(secret, str(raised.exception))
        self.assertIn("HTTP 401", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
