#!/usr/bin/env python3
"""Aggregate the independent pre-modelling gates into one foundation_v2 status."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.pipeline_io import write_json_atomic as _write_json_atomic
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import write_json_atomic as _write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provisional_amendment(
    universe: dict[str, Any],
    model_config: dict[str, Any],
    amendment: dict[str, Any] | None,
    yahoo_manifest: dict[str, Any] | None,
    current_file_hashes: dict[str, str] | None,
) -> tuple[bool, list[str]]:
    """Validate the narrowly scoped user-authorized Yahoo amendment."""

    issues: list[str] = []
    policy = model_config.get("provenance", {})
    quality = model_config.get("quality_gates", {})
    if amendment is None:
        return False, ["provisional_amendment_absent"]
    if yahoo_manifest is None:
        return False, ["yahoo_reference_manifest_absent"]
    if amendment.get("status") != "active":
        issues.append("provisional_amendment_not_active")
    if amendment.get("amendment_id") != policy.get("amendment_id"):
        issues.append("provisional_amendment_id_mismatch")
    if amendment.get("amended_universe_provenance_status") != "provisional_authorized_yahoo":
        issues.append("provisional_amendment_status_invalid")
    if not amendment.get("modelling_allowed"):
        issues.append("provisional_amendment_does_not_allow_modelling")
    if amendment.get("licensed_provenance_claim_allowed") is not False:
        issues.append("provisional_amendment_must_forbid_licensed_claim")
    if amendment.get("reporting_scope") != "provisional_research_results":
        issues.append("provisional_reporting_scope_invalid")
    if policy.get("reporting_scope") != amendment.get("reporting_scope"):
        issues.append("model_config_reporting_scope_mismatch")
    if not policy.get("allow_modelling_with_provisional_universe"):
        issues.append("model_config_provisional_modelling_disabled")
    if policy.get("licensed_provenance_claim_allowed") is not False:
        issues.append("model_config_must_forbid_licensed_claim")
    if not quality.get("allow_provisional_universe_provenance"):
        issues.append("quality_gate_provisional_provenance_disabled")
    if quality.get("allow_failed_universe_provenance"):
        issues.append("generic_failed_provenance_waiver_forbidden")

    allowed_original_blockers = {"authorized_point_in_time_universe_extract_absent"}
    if (
        universe.get("status") != "blocked"
        or set(universe.get("blockers", [])) != allowed_original_blockers
    ):
        issues.append("universe_has_nonwaivable_provenance_issue")
    if amendment.get("expected_public_universe_sha256") != universe.get("public_universe_sha256"):
        issues.append("provisional_amendment_universe_hash_mismatch")
    if amendment.get("expected_universe_source_manifest_sha256") != universe.get(
        "source_manifest_sha256"
    ):
        issues.append("provisional_amendment_source_manifest_hash_mismatch")

    if yahoo_manifest.get("status") != "provisional_non_licensed":
        issues.append("yahoo_reference_status_invalid")
    if yahoo_manifest.get("row_count") != 101:
        issues.append("yahoo_reference_row_count_invalid")
    if yahoo_manifest.get("metadata_found_count", 0) < amendment.get(
        "minimum_yahoo_metadata_matches", 101
    ):
        issues.append("yahoo_metadata_coverage_below_amendment_minimum")
    if yahoo_manifest.get("price_found_count", 0) < amendment.get(
        "minimum_yahoo_price_matches", 101
    ):
        issues.append("yahoo_price_coverage_below_amendment_minimum")
    failure_tickers = {item.get("ticker") for item in yahoo_manifest.get("request_failures", [])}
    if failure_tickers != set(amendment.get("accepted_yahoo_missing_tickers", [])):
        issues.append("yahoo_missing_tickers_differ_from_amendment")
    if (yahoo_manifest.get("gate_eligibility") or {}).get(
        "foundation_v2_universe_provenance"
    ) is not False:
        issues.append("yahoo_manifest_must_remain_nonlicensed")
    if current_file_hashes is not None and yahoo_manifest.get(
        "output_sha256"
    ) != current_file_hashes.get("yahoo_reference_sha256"):
        issues.append("yahoo_reference_file_hash_mismatch")
    return not issues, issues


def aggregate_foundation_status(
    access: dict[str, Any],
    universe: dict[str, Any],
    market: dict[str, Any],
    construction: dict[str, Any],
    arithmetic: dict[str, Any],
    model_config: dict[str, Any],
    current_file_hashes: dict[str, str] | None = None,
    provenance_amendment: dict[str, Any] | None = None,
    yahoo_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic aggregate status without erasing historical audits."""

    universe_pass = universe.get("status") == "pass"
    provisional_universe_pass, amendment_issues = validate_provisional_amendment(
        universe,
        model_config,
        provenance_amendment,
        yahoo_manifest,
        current_file_hashes,
    )
    universe_accepted = universe_pass or provisional_universe_pass
    market_pass = market.get("status") == "pass"
    construction_pass = construction.get("gate_status") == "pass"
    arithmetic_pass = arithmetic.get("status") == "pass"
    method_pass = model_config.get("protocol_status") == "frozen_before_out_of_sample_modelling"
    construction_inputs = construction.get("input_hashes", {})
    construction_provenance_bound = bool(
        construction.get("universe_provenance_status") == "pass"
        and construction.get("output_scope") == "confirmed_foundation_input"
        and construction_inputs.get("universe_json_sha256")
        == universe.get("public_universe_sha256")
        and construction_inputs.get("universe_source_manifest_sha256")
        == universe.get("source_manifest_sha256")
    )
    construction_provisional_bound = bool(
        provisional_universe_pass
        and construction.get("universe_provenance_status") == "blocked"
        and construction.get("output_scope") == "provisional_pending_universe_provenance"
        and construction_inputs.get("universe_json_sha256")
        == universe.get("public_universe_sha256")
        and construction_inputs.get("universe_source_manifest_sha256")
        == universe.get("source_manifest_sha256")
    )
    construction_universe_bound = bool(
        construction_provenance_bound or construction_provisional_bound
    )
    construction_market_bound = bool(
        construction_inputs.get("market_input_manifest_sha256")
        and construction_inputs.get("market_input_manifest_sha256")
        == market.get("public_manifest_sha256")
    )
    arithmetic_inputs = arithmetic.get("inputs", {})
    construction_outputs = construction.get("output_hashes", {})
    arithmetic_bound = bool(
        arithmetic_inputs.get("simple_return_panel", {}).get("sha256")
        == construction_outputs.get("portfolio_simple_return_panel_sha256")
        and arithmetic_inputs.get("active_universe", {}).get("sha256")
        == construction_outputs.get("active_universe_sha256")
        and arithmetic_inputs.get("final_universe", {}).get("sha256")
        == construction_outputs.get("final_universe_sha256")
        and construction_outputs.get("portfolio_simple_return_panel_sha256")
    )
    files = current_file_hashes
    universe_files_current = bool(
        files is None
        or (
            universe.get("public_universe_sha256") == files.get("universe_json_sha256")
            and universe.get("source_manifest_sha256")
            == files.get("universe_source_manifest_sha256")
        )
    )
    market_file_current = bool(
        files is None
        or market.get("public_manifest_sha256") == files.get("market_input_manifest_sha256")
    )
    construction_files_current = bool(
        files is None
        or (
            construction_inputs.get("universe_json_sha256") == files.get("universe_json_sha256")
            and construction_inputs.get("universe_source_manifest_sha256")
            == files.get("universe_source_manifest_sha256")
            and construction_inputs.get("market_input_manifest_sha256")
            == files.get("market_input_manifest_sha256")
            and construction_inputs.get("data_config_sha256") == files.get("data_config_sha256")
            and construction_inputs.get("security_master_sha256")
            == files.get("security_master_sha256")
            and construction_inputs.get("manual_corporate_actions_sha256")
            == files.get("manual_corporate_actions_sha256")
            and construction_inputs.get("lifecycle_events_sha256")
            == files.get("lifecycle_events_sha256")
            and construction_inputs.get("observation_reviews_sha256")
            == files.get("observation_reviews_sha256")
            and construction_outputs.get("portfolio_simple_return_panel_sha256")
            == files.get("portfolio_simple_return_panel_sha256")
            and construction_outputs.get("active_universe_sha256")
            == files.get("active_universe_sha256")
            and construction_outputs.get("final_universe_sha256")
            == files.get("final_universe_sha256")
        )
    )
    arithmetic_files_current = bool(
        files is None
        or (
            arithmetic_inputs.get("simple_return_panel", {}).get("sha256")
            == files.get("portfolio_simple_return_panel_sha256")
            and arithmetic_inputs.get("active_universe", {}).get("sha256")
            == files.get("active_universe_sha256")
            and arithmetic_inputs.get("final_universe", {}).get("sha256")
            == files.get("final_universe_sha256")
        )
    )

    blockers: list[str] = []
    if not universe_accepted:
        if provenance_amendment is not None:
            blockers.extend(amendment_issues)
        else:
            blockers.extend(str(item) for item in universe.get("blockers", []))
            if not universe.get("blockers"):
                blockers.append("universe_provenance_not_passed")
    elif not universe_files_current:
        blockers.append("universe_files_changed_since_provenance_audit")
    if not market_pass:
        blockers.append("market_input_integrity_not_passed")
    elif not market_file_current:
        blockers.append("market_manifest_changed_since_integrity_audit")
    if not construction_pass:
        blockers.append("data_construction_not_passed")
    elif universe_accepted and not construction_universe_bound:
        blockers.append("data_construction_not_bound_to_approved_universe")
    if market_pass and not construction_market_bound:
        blockers.append("data_construction_not_bound_to_verified_market_manifest")
    if construction_pass and not construction_files_current:
        blockers.append("construction_inputs_or_outputs_changed_since_audit")
    if not arithmetic_pass:
        blockers.append("portfolio_arithmetic_not_passed")
    elif not arithmetic_bound:
        blockers.append("portfolio_arithmetic_not_bound_to_current_panels")
    elif not arithmetic_files_current:
        blockers.append("portfolio_arithmetic_inputs_changed_since_audit")
    if not method_pass:
        blockers.append("method_freeze_not_passed")
    blockers = list(dict.fromkeys(blockers))
    foundation_ready = not blockers
    foundation_status = (
        "pass"
        if foundation_ready and universe_pass
        else "provisional_pass"
        if foundation_ready and provisional_universe_pass
        else "blocked"
    )
    modelling_status = (
        "pass"
        if foundation_status == "pass"
        else "pass_provisional"
        if foundation_status == "provisional_pass"
        else "blocked"
    )

    return {
        "schema_version": 2,
        "foundation_v2": {
            "status": foundation_status,
            "blockers": blockers,
            "reporting_scope": (
                "confirmatory"
                if foundation_status == "pass"
                else "provisional_research_results"
                if foundation_status == "provisional_pass"
                else "provisional_diagnostics_only"
            ),
            "licensed_universe_provenance": universe_pass,
            "note": (
                "All provenance, input-integrity, construction, arithmetic, and method-freeze "
                "gates pass; confirmatory modelling may begin."
                if foundation_status == "pass"
                else "Modelling may begin under protocol amendment PA-001. Universe provenance "
                "and every downstream result must remain labelled provisional."
                if foundation_status == "provisional_pass"
                else "Confirmatory modelling remains blocked until every listed foundation "
                "condition passes. Diagnostic data outputs are provisional."
            ),
        },
        "gates": {
            "data_access_feasibility_historical": {
                "status": access.get("overall_status", "unknown"),
                "source": "data/audit/data_access_audit.json",
                "scope": "Historical access/entitlement feasibility only",
                "superseded_by": "foundation_v2",
            },
            "universe_provenance": {
                "status": (
                    "pass"
                    if universe_pass
                    else "provisional_authorized_yahoo"
                    if provisional_universe_pass
                    else "blocked"
                ),
                "source": "data/audit/universe_provenance.json",
                "blockers": [str(item) for item in universe.get("blockers", [])],
                "current_file_binding": universe_files_current,
                "licensed_provenance": universe_pass,
            },
            "provenance_amendment": {
                "status": "pass" if provisional_universe_pass else "not_applied",
                "source": "config/provenance_amendment.json",
                "amendment_id": (
                    provenance_amendment.get("amendment_id") if provenance_amendment else None
                ),
                "issues": amendment_issues,
            },
            "market_input_integrity_v2": {
                "status": "pass" if market_pass else "blocked",
                "source": "data/audit/market_input_integrity.json",
                "current_file_binding": market_file_current,
            },
            "data_construction_v2": {
                "status": "pass" if construction_pass else "blocked",
                "source": "data/audit/data_quality_report.json",
                "output_scope": construction.get("output_scope", "unknown"),
                "approved_universe_binding": construction_universe_bound,
                "binding_scope": (
                    "licensed"
                    if construction_provenance_bound
                    else "provisional_amendment"
                    if construction_provisional_bound
                    else "none"
                ),
                "verified_market_binding": construction_market_bound,
                "current_file_binding": construction_files_current,
            },
            "portfolio_arithmetic_v1": {
                "status": "pass" if arithmetic_pass else "blocked",
                "source": "data/audit/portfolio_arithmetic.json",
                "current_panel_binding": arithmetic_bound,
                "current_file_binding": arithmetic_files_current,
            },
            "method_freeze": {
                "status": "pass" if method_pass else "blocked",
                "source": "config/model_config.toml",
            },
            "modelling_readiness": {
                "status": modelling_status,
                "reason": (
                    None
                    if modelling_status == "pass"
                    else "authorized_provisional_universe"
                    if modelling_status == "pass_provisional"
                    else "foundation_v2_not_passed"
                ),
            },
        },
        "historical_access_audit_is_preserved": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--require-modelling-ready", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/audit/current_gate_status.json"
    )
    args = parser.parse_args()
    if args.require_pass and args.require_modelling_ready:
        parser.error("choose at most one of --require-pass and --require-modelling-ready")

    with (PROJECT_ROOT / "config/model_config.toml").open("rb") as handle:
        model_config = tomllib.load(handle)
    provenance_policy = model_config.get("provenance", {})
    amendment_path = PROJECT_ROOT / provenance_policy.get(
        "amendment_path", "config/provenance_amendment.json"
    )
    yahoo_manifest_path = PROJECT_ROOT / provenance_policy.get(
        "yahoo_manifest_path", "data/manifests/yahoo_universe_reference_manifest.json"
    )
    yahoo_reference_path = PROJECT_ROOT / provenance_policy.get(
        "yahoo_reference_path", "data/interim/provisional/yahoo_universe_reference.csv"
    )
    amendment = _read_json(amendment_path) if amendment_path.is_file() else None
    yahoo_manifest = _read_json(yahoo_manifest_path) if yahoo_manifest_path.is_file() else None
    status = aggregate_foundation_status(
        _read_json(PROJECT_ROOT / "data/audit/data_access_audit.json"),
        _read_json(PROJECT_ROOT / "data/audit/universe_provenance.json"),
        _read_json(PROJECT_ROOT / "data/audit/market_input_integrity.json"),
        _read_json(PROJECT_ROOT / "data/audit/data_quality_report.json"),
        _read_json(PROJECT_ROOT / "data/audit/portfolio_arithmetic.json"),
        model_config,
        {
            "universe_json_sha256": _sha256(
                PROJECT_ROOT / "data/raw/universe_sp100_2020-01-02.json"
            ),
            "universe_source_manifest_sha256": _sha256(
                PROJECT_ROOT / "data/manifests/universe_source_manifest.json"
            ),
            "market_input_manifest_sha256": _sha256(
                PROJECT_ROOT / "data/manifests/market_input_manifest.json"
            ),
            "data_config_sha256": _sha256(PROJECT_ROOT / "config/data_config.toml"),
            "security_master_sha256": _sha256(PROJECT_ROOT / "config/security_master.json"),
            "manual_corporate_actions_sha256": _sha256(
                PROJECT_ROOT / "config/manual_corporate_actions.json"
            ),
            "lifecycle_events_sha256": _sha256(PROJECT_ROOT / "config/lifecycle_events.json"),
            "observation_reviews_sha256": _sha256(PROJECT_ROOT / "config/observation_reviews.json"),
            "portfolio_simple_return_panel_sha256": _sha256(
                PROJECT_ROOT / "data/processed/portfolio_constituent_simple_returns.parquet"
            ),
            "active_universe_sha256": _sha256(
                PROJECT_ROOT / "data/processed/active_universe_by_year.json"
            ),
            "final_universe_sha256": _sha256(PROJECT_ROOT / "data/processed/final_universe.json"),
            "yahoo_reference_sha256": _sha256(yahoo_reference_path),
        },
        amendment,
        yahoo_manifest,
    )
    _write_json_atomic(args.output, status)
    print(f"foundation_v2={status['foundation_v2']['status']}")
    if status["foundation_v2"]["blockers"]:
        print("blockers=" + ",".join(status["foundation_v2"]["blockers"]))
    if args.require_pass and status["foundation_v2"]["status"] != "pass":
        return 2
    if args.require_modelling_ready and status["gates"]["modelling_readiness"]["status"] not in {
        "pass",
        "pass_provisional",
    }:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
