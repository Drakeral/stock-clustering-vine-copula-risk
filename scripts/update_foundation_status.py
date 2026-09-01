#!/usr/bin/env python3
"""Aggregate the independent pre-modelling gates into one foundation_v2 status."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


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


def aggregate_foundation_status(
    access: dict[str, Any],
    universe: dict[str, Any],
    market: dict[str, Any],
    construction: dict[str, Any],
    arithmetic: dict[str, Any],
    model_config: dict[str, Any],
    current_file_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic aggregate status without erasing historical audits."""

    universe_pass = universe.get("status") == "pass"
    market_pass = market.get("status") == "pass"
    construction_pass = construction.get("gate_status") == "pass"
    arithmetic_pass = arithmetic.get("status") == "pass"
    method_pass = (
        model_config.get("protocol_status") == "frozen_before_out_of_sample_modelling"
    )
    construction_inputs = construction.get("input_hashes", {})
    construction_provenance_bound = bool(
        construction.get("universe_provenance_status") == "pass"
        and construction.get("output_scope") == "confirmed_foundation_input"
        and construction_inputs.get("universe_json_sha256")
        == universe.get("public_universe_sha256")
        and construction_inputs.get("universe_source_manifest_sha256")
        == universe.get("source_manifest_sha256")
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
        or market.get("public_manifest_sha256")
        == files.get("market_input_manifest_sha256")
    )
    construction_files_current = bool(
        files is None
        or (
            construction_inputs.get("universe_json_sha256")
            == files.get("universe_json_sha256")
            and construction_inputs.get("universe_source_manifest_sha256")
            == files.get("universe_source_manifest_sha256")
            and construction_inputs.get("market_input_manifest_sha256")
            == files.get("market_input_manifest_sha256")
            and construction_inputs.get("data_config_sha256")
            == files.get("data_config_sha256")
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
    if not universe_pass:
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
    elif universe_pass and not construction_provenance_bound:
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
    foundation_pass = not blockers

    return {
        "schema_version": 2,
        "foundation_v2": {
            "status": "pass" if foundation_pass else "blocked",
            "blockers": blockers,
            "note": (
                "All provenance, input-integrity, construction, arithmetic, and method-freeze "
                "gates pass; confirmatory modelling may begin."
                if foundation_pass
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
                "status": "pass" if universe_pass else "blocked",
                "source": "data/audit/universe_provenance.json",
                "blockers": [str(item) for item in universe.get("blockers", [])],
                "current_file_binding": universe_files_current,
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
                "approved_universe_binding": construction_provenance_bound,
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
                "status": "pass" if foundation_pass else "blocked",
                "reason": None if foundation_pass else "foundation_v2_not_passed",
            },
        },
        "historical_access_audit_is_preserved": True,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/audit/current_gate_status.json"
    )
    args = parser.parse_args()

    with (PROJECT_ROOT / "config/model_config.toml").open("rb") as handle:
        model_config = tomllib.load(handle)
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
            "security_master_sha256": _sha256(
                PROJECT_ROOT / "config/security_master.json"
            ),
            "manual_corporate_actions_sha256": _sha256(
                PROJECT_ROOT / "config/manual_corporate_actions.json"
            ),
            "lifecycle_events_sha256": _sha256(
                PROJECT_ROOT / "config/lifecycle_events.json"
            ),
            "observation_reviews_sha256": _sha256(
                PROJECT_ROOT / "config/observation_reviews.json"
            ),
            "portfolio_simple_return_panel_sha256": _sha256(
                PROJECT_ROOT
                / "data/processed/portfolio_constituent_simple_returns.parquet"
            ),
            "active_universe_sha256": _sha256(
                PROJECT_ROOT / "data/processed/active_universe_by_year.json"
            ),
            "final_universe_sha256": _sha256(
                PROJECT_ROOT / "data/processed/final_universe.json"
            ),
        },
    )
    _write_json_atomic(args.output, status)
    print(f"foundation_v2={status['foundation_v2']['status']}")
    if status["foundation_v2"]["blockers"]:
        print("blockers=" + ",".join(status["foundation_v2"]["blockers"]))
    return 2 if args.require_pass and status["foundation_v2"]["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
