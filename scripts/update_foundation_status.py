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


def _failed_checks(checks: tuple[tuple[bool, str], ...]) -> list[str]:
    """Return the issue codes for failed declarative gate checks."""

    return [issue for condition, issue in checks if not condition]


def _amendment_policy_issues(
    amendment: dict[str, Any],
    policy: dict[str, Any],
    quality: dict[str, Any],
) -> list[str]:
    """Validate the configuration and authorization fields of PA-001."""

    return _failed_checks(
        (
            (amendment.get("status") == "active", "provisional_amendment_not_active"),
            (
                amendment.get("amendment_id") == policy.get("amendment_id"),
                "provisional_amendment_id_mismatch",
            ),
            (
                amendment.get("amended_universe_provenance_status")
                == "provisional_authorized_yahoo",
                "provisional_amendment_status_invalid",
            ),
            (
                bool(amendment.get("modelling_allowed")),
                "provisional_amendment_does_not_allow_modelling",
            ),
            (
                amendment.get("licensed_provenance_claim_allowed") is False,
                "provisional_amendment_must_forbid_licensed_claim",
            ),
            (
                amendment.get("reporting_scope") == "provisional_research_results",
                "provisional_reporting_scope_invalid",
            ),
            (
                policy.get("reporting_scope") == amendment.get("reporting_scope"),
                "model_config_reporting_scope_mismatch",
            ),
            (
                bool(policy.get("allow_modelling_with_provisional_universe")),
                "model_config_provisional_modelling_disabled",
            ),
            (
                policy.get("licensed_provenance_claim_allowed") is False,
                "model_config_must_forbid_licensed_claim",
            ),
            (
                bool(quality.get("allow_provisional_universe_provenance")),
                "quality_gate_provisional_provenance_disabled",
            ),
            (
                not quality.get("allow_failed_universe_provenance"),
                "generic_failed_provenance_waiver_forbidden",
            ),
        )
    )


def _amendment_universe_issues(universe: dict[str, Any], amendment: dict[str, Any]) -> list[str]:
    """Ensure PA-001 waives only the single named provenance blocker."""

    allowed_blockers = {"authorized_point_in_time_universe_extract_absent"}
    return _failed_checks(
        (
            (
                universe.get("status") == "blocked"
                and set(universe.get("blockers", [])) == allowed_blockers,
                "universe_has_nonwaivable_provenance_issue",
            ),
            (
                amendment.get("expected_public_universe_sha256")
                == universe.get("public_universe_sha256"),
                "provisional_amendment_universe_hash_mismatch",
            ),
            (
                amendment.get("expected_universe_source_manifest_sha256")
                == universe.get("source_manifest_sha256"),
                "provisional_amendment_source_manifest_hash_mismatch",
            ),
        )
    )


def _amendment_yahoo_issues(
    yahoo_manifest: dict[str, Any],
    amendment: dict[str, Any],
    current_file_hashes: dict[str, str] | None,
) -> list[str]:
    """Validate the secondary Yahoo coverage and current file binding."""

    failures = {item.get("ticker") for item in yahoo_manifest.get("request_failures", [])}
    checks = [
        (
            yahoo_manifest.get("status") == "provisional_non_licensed",
            "yahoo_reference_status_invalid",
        ),
        (yahoo_manifest.get("row_count") == 101, "yahoo_reference_row_count_invalid"),
        (
            yahoo_manifest.get("metadata_found_count", 0)
            >= amendment.get("minimum_yahoo_metadata_matches", 101),
            "yahoo_metadata_coverage_below_amendment_minimum",
        ),
        (
            yahoo_manifest.get("price_found_count", 0)
            >= amendment.get("minimum_yahoo_price_matches", 101),
            "yahoo_price_coverage_below_amendment_minimum",
        ),
        (
            failures == set(amendment.get("accepted_yahoo_missing_tickers", [])),
            "yahoo_missing_tickers_differ_from_amendment",
        ),
        (
            (yahoo_manifest.get("gate_eligibility") or {}).get("foundation_v2_universe_provenance")
            is False,
            "yahoo_manifest_must_remain_nonlicensed",
        ),
    ]
    if current_file_hashes is not None:
        checks.append(
            (
                yahoo_manifest.get("output_sha256")
                == current_file_hashes.get("yahoo_reference_sha256"),
                "yahoo_reference_file_hash_mismatch",
            )
        )
    return _failed_checks(tuple(checks))


def validate_provisional_amendment(
    universe: dict[str, Any],
    model_config: dict[str, Any],
    amendment: dict[str, Any] | None,
    yahoo_manifest: dict[str, Any] | None,
    current_file_hashes: dict[str, str] | None,
) -> tuple[bool, list[str]]:
    """Validate the narrowly scoped user-authorized Yahoo amendment."""

    if amendment is None:
        return False, ["provisional_amendment_absent"]
    if yahoo_manifest is None:
        return False, ["yahoo_reference_manifest_absent"]
    policy = model_config.get("provenance", {})
    quality = model_config.get("quality_gates", {})
    issues = _amendment_policy_issues(amendment, policy, quality)
    issues.extend(_amendment_universe_issues(universe, amendment))
    issues.extend(_amendment_yahoo_issues(yahoo_manifest, amendment, current_file_hashes))
    return not issues, issues


def _same_universe_hashes(construction_inputs: dict[str, Any], universe: dict[str, Any]) -> bool:
    return bool(
        construction_inputs.get("universe_json_sha256") == universe.get("public_universe_sha256")
        and construction_inputs.get("universe_source_manifest_sha256")
        == universe.get("source_manifest_sha256")
    )


def _construction_provenance_binding(
    construction: dict[str, Any], universe: dict[str, Any]
) -> bool:
    inputs = construction.get("input_hashes", {})
    return bool(
        construction.get("universe_provenance_status") == "pass"
        and construction.get("output_scope") == "confirmed_foundation_input"
        and _same_universe_hashes(inputs, universe)
    )


def _construction_provisional_binding(
    construction: dict[str, Any],
    universe: dict[str, Any],
    provisional_universe_pass: bool,
) -> bool:
    inputs = construction.get("input_hashes", {})
    return bool(
        provisional_universe_pass
        and construction.get("universe_provenance_status") == "blocked"
        and construction.get("output_scope") == "provisional_pending_universe_provenance"
        and _same_universe_hashes(inputs, universe)
    )


def _construction_market_binding(
    construction_inputs: dict[str, Any], market: dict[str, Any]
) -> bool:
    recorded_hash = construction_inputs.get("market_input_manifest_sha256")
    return bool(recorded_hash and recorded_hash == market.get("public_manifest_sha256"))


def _arithmetic_binding(
    arithmetic_inputs: dict[str, Any], construction_outputs: dict[str, Any]
) -> bool:
    expected_panel_hash = construction_outputs.get("portfolio_simple_return_panel_sha256")
    return bool(
        expected_panel_hash
        and arithmetic_inputs.get("simple_return_panel", {}).get("sha256") == expected_panel_hash
        and arithmetic_inputs.get("active_universe", {}).get("sha256")
        == construction_outputs.get("active_universe_sha256")
        and arithmetic_inputs.get("final_universe", {}).get("sha256")
        == construction_outputs.get("final_universe_sha256")
    )


def _universe_files_current(universe: dict[str, Any], files: dict[str, str] | None) -> bool:
    return bool(
        files is None
        or (
            universe.get("public_universe_sha256") == files.get("universe_json_sha256")
            and universe.get("source_manifest_sha256")
            == files.get("universe_source_manifest_sha256")
        )
    )


def _construction_files_current(
    construction_inputs: dict[str, Any],
    construction_outputs: dict[str, Any],
    files: dict[str, str] | None,
) -> bool:
    if files is None:
        return True
    input_fields = (
        "universe_json_sha256",
        "universe_source_manifest_sha256",
        "market_input_manifest_sha256",
        "data_config_sha256",
        "security_master_sha256",
        "manual_corporate_actions_sha256",
        "lifecycle_events_sha256",
        "observation_reviews_sha256",
    )
    output_fields = (
        "portfolio_simple_return_panel_sha256",
        "active_universe_sha256",
        "final_universe_sha256",
    )
    return all(
        construction_inputs.get(field) == files.get(field) for field in input_fields
    ) and all(construction_outputs.get(field) == files.get(field) for field in output_fields)


def _arithmetic_files_current(
    arithmetic_inputs: dict[str, Any], files: dict[str, str] | None
) -> bool:
    if files is None:
        return True
    bindings = (
        ("simple_return_panel", "portfolio_simple_return_panel_sha256"),
        ("active_universe", "active_universe_sha256"),
        ("final_universe", "final_universe_sha256"),
    )
    return all(
        arithmetic_inputs.get(record_name, {}).get("sha256") == files.get(file_name)
        for record_name, file_name in bindings
    )


def _universe_blockers(
    universe_accepted: bool,
    universe_files_current: bool,
    universe: dict[str, Any],
    amendment: dict[str, Any] | None,
    amendment_issues: list[str],
) -> list[str]:
    """Return provenance blockers while retaining the original audit reasons."""

    if universe_accepted:
        return [] if universe_files_current else ["universe_files_changed_since_provenance_audit"]
    if amendment is not None:
        return amendment_issues
    blockers = [str(item) for item in universe.get("blockers", [])]
    return blockers or ["universe_provenance_not_passed"]


def _collect_foundation_blockers(
    *,
    universe_accepted: bool,
    universe_files_current: bool,
    universe: dict[str, Any],
    amendment: dict[str, Any] | None,
    amendment_issues: list[str],
    checks: dict[str, bool],
) -> list[str]:
    """Translate independent gate booleans into stable public blocker codes."""

    blockers = _universe_blockers(
        universe_accepted,
        universe_files_current,
        universe,
        amendment,
        amendment_issues,
    )
    conditional_codes = (
        (not checks["market_pass"], "market_input_integrity_not_passed"),
        (
            checks["market_pass"] and not checks["market_file_current"],
            "market_manifest_changed_since_integrity_audit",
        ),
        (not checks["construction_pass"], "data_construction_not_passed"),
        (
            checks["construction_pass"]
            and universe_accepted
            and not checks["construction_universe_bound"],
            "data_construction_not_bound_to_approved_universe",
        ),
        (
            checks["market_pass"] and not checks["construction_market_bound"],
            "data_construction_not_bound_to_verified_market_manifest",
        ),
        (
            checks["construction_pass"] and not checks["construction_files_current"],
            "construction_inputs_or_outputs_changed_since_audit",
        ),
        (not checks["arithmetic_pass"], "portfolio_arithmetic_not_passed"),
        (
            checks["arithmetic_pass"] and not checks["arithmetic_bound"],
            "portfolio_arithmetic_not_bound_to_current_panels",
        ),
        (
            checks["arithmetic_pass"]
            and checks["arithmetic_bound"]
            and not checks["arithmetic_files_current"],
            "portfolio_arithmetic_inputs_changed_since_audit",
        ),
        (not checks["method_pass"], "method_freeze_not_passed"),
    )
    blockers.extend(code for condition, code in conditional_codes if condition)
    return list(dict.fromkeys(blockers))


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
    construction_provenance_bound = _construction_provenance_binding(construction, universe)
    construction_provisional_bound = _construction_provisional_binding(
        construction,
        universe,
        provisional_universe_pass,
    )
    construction_universe_bound = construction_provenance_bound or construction_provisional_bound
    construction_market_bound = _construction_market_binding(construction_inputs, market)
    arithmetic_inputs = arithmetic.get("inputs", {})
    construction_outputs = construction.get("output_hashes", {})
    arithmetic_bound = _arithmetic_binding(arithmetic_inputs, construction_outputs)
    files = current_file_hashes
    universe_files_current = _universe_files_current(universe, files)
    market_file_current = bool(
        files is None
        or market.get("public_manifest_sha256") == files.get("market_input_manifest_sha256")
    )
    construction_files_current = _construction_files_current(
        construction_inputs,
        construction_outputs,
        files,
    )
    arithmetic_files_current = _arithmetic_files_current(arithmetic_inputs, files)

    checks = {
        "market_pass": market_pass,
        "market_file_current": market_file_current,
        "construction_pass": construction_pass,
        "construction_universe_bound": construction_universe_bound,
        "construction_market_bound": construction_market_bound,
        "construction_files_current": construction_files_current,
        "arithmetic_pass": arithmetic_pass,
        "arithmetic_bound": arithmetic_bound,
        "arithmetic_files_current": arithmetic_files_current,
        "method_pass": method_pass,
    }
    blockers = _collect_foundation_blockers(
        universe_accepted=universe_accepted,
        universe_files_current=universe_files_current,
        universe=universe,
        amendment=provenance_amendment,
        amendment_issues=amendment_issues,
        checks=checks,
    )
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
