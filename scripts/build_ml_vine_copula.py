#!/usr/bin/env python3
"""Fit exploratory M6/M8 tree-3 vines without touching primary vine artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyvinecopulib as pv

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.build_gaussian_copula import _validate_marginal_binding
    from scripts.build_ml_groupings import validate_ml_protocol
    from scripts.build_vine_copula import (
        _validate_upstream_audits,
        build_vine_outputs,
        validate_vine_protocol,
    )
    from scripts.ml_risk_common import (
        ML_RISK_ANALYSIS_ROLE,
        ML_VINE_MODEL_BY_GROUPING,
        artifact_record,
        load_json,
    )
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        write_json_atomic,
        write_parquet_atomic,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from build_gaussian_copula import _validate_marginal_binding
    from build_ml_groupings import validate_ml_protocol
    from build_vine_copula import (
        _validate_upstream_audits,
        build_vine_outputs,
        validate_vine_protocol,
    )
    from ml_risk_common import (
        ML_RISK_ANALYSIS_ROLE,
        ML_VINE_MODEL_BY_GROUPING,
        artifact_record,
        load_json,
    )
    from pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        write_json_atomic,
        write_parquet_atomic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-pits",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_monthly_copula_training_pits.parquet",
    )
    parser.add_argument(
        "--marginal-refits",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_marginal_refits.parquet",
    )
    parser.add_argument(
        "--daily-margins",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_marginal_daily_forecasts.parquet",
    )
    parser.add_argument(
        "--group-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_annual_group_returns.parquet",
    )
    parser.add_argument(
        "--gaussian-refits",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_gaussian_copula_refits.parquet",
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/simulation_seed_manifest.json",
    )
    parser.add_argument(
        "--marginal-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_marginal_model_quality.json",
    )
    parser.add_argument(
        "--gaussian-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_gaussian_copula_quality.json",
    )
    parser.add_argument(
        "--model-config", type=Path, default=PROJECT_ROOT / "config/model_config.toml"
    )
    parser.add_argument(
        "--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_extension_config.toml"
    )
    parser.add_argument(
        "--foundation-status",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_gate_status.json",
    )
    parser.add_argument(
        "--refits-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_vine_copula_refits.parquet",
    )
    parser.add_argument(
        "--forecasts-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_vine_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_vine_copula_quality.json",
    )
    parser.add_argument("--progress-every", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    with args.ml_config.open("rb") as handle:
        ml_config = validate_ml_protocol(tomllib.load(handle))
    vine, _, _, _, truncation = validate_vine_protocol(model_config)
    if truncation != 3:
        raise ValueError("the exploratory ML vine must use the frozen tree-3 truncation")
    foundation = load_json(args.foundation_status)
    require_current_hash_records(
        foundation,
        source_name=project_path(args.foundation_status),
    )
    scope = reporting_scope(foundation)
    marginal_audit = load_json(args.marginal_audit)
    require_current_hash_records(
        marginal_audit,
        source_name=project_path(args.marginal_audit),
    )
    _validate_marginal_binding(
        marginal_audit,
        training_pits=args.training_pits,
        marginal_refits=args.marginal_refits,
        daily_margins=args.daily_margins,
        group_returns=args.group_returns,
        model_config=args.model_config,
    )
    if marginal_audit.get("analysis_role") != ML_RISK_ANALYSIS_ROLE:
        raise RuntimeError("ML marginal audit has an incompatible analysis role")
    gaussian_audit = load_json(args.gaussian_audit)
    require_current_hash_records(
        gaussian_audit,
        source_name=project_path(args.gaussian_audit),
    )
    if gaussian_audit.get("analysis_role") != ML_RISK_ANALYSIS_ROLE:
        raise RuntimeError("ML Gaussian audit has an incompatible analysis role")
    _validate_upstream_audits(
        gaussian_audit=gaussian_audit,
        gaussian_refits=args.gaussian_refits,
        seed_manifest=args.seed_manifest,
    )
    seed_manifest = load_json(args.seed_manifest)
    refits, forecasts, audit = build_vine_outputs(
        pd.read_parquet(args.training_pits),
        pd.read_parquet(args.marginal_refits),
        pd.read_parquet(args.daily_margins),
        pd.read_parquet(args.group_returns),
        pd.read_parquet(args.gaussian_refits),
        seed_manifest,
        model_config,
        universe_variant=str(ml_config["grouping"]["universe_variant"]),
        truncation_level=truncation,
        progress_every=args.progress_every,
        model_by_grouping=ML_VINE_MODEL_BY_GROUPING,
        analysis_role=ML_RISK_ANALYSIS_ROLE,
    )
    quality_eligible = bool(audit["eligible_to_be_declared_best"])
    audit["eligible_for_exploratory_comparison"] = quality_eligible
    audit["eligible_to_be_declared_best"] = False
    restrictions = list(audit["best_model_restrictions"])
    restrictions.append("exploratory_analysis_cannot_revise_core_model_ranking")
    audit["best_model_restrictions"] = restrictions
    write_parquet_atomic(args.refits_output, refits)
    write_parquet_atomic(args.forecasts_output, forecasts)
    audit.update(
        {
            "gate_name": "ml_vine_copula_quality_v1",
            "reporting_scope": scope,
            "method": {
                "library": "pyvinecopulib",
                "library_version": pv.__version__,
                "structure_selection": vine["structure_selection"],
                "family_selection": vine["family_selection"],
                "families": vine["families"],
                "rotations_degrees": vine["rotations_degrees"],
                "truncation_level": truncation,
                "num_threads": 1,
            },
            "inputs": {
                "monthly_copula_training_pits": artifact_record(args.training_pits),
                "marginal_refits": artifact_record(args.marginal_refits),
                "marginal_daily_forecasts": artifact_record(args.daily_margins),
                "annual_group_returns": artifact_record(args.group_returns),
                "gaussian_copula_refits": artifact_record(args.gaussian_refits),
                "simulation_seed_manifest": artifact_record(args.seed_manifest),
                "marginal_model_quality": artifact_record(args.marginal_audit),
                "gaussian_copula_quality": artifact_record(args.gaussian_audit),
                "model_config": artifact_record(args.model_config),
                "ml_config": artifact_record(args.ml_config),
                "foundation_status": artifact_record(args.foundation_status),
            },
            "outputs": {
                "vine_copula_refits": artifact_record(args.refits_output, rows=len(refits)),
                "vine_risk_forecasts": artifact_record(args.forecasts_output, rows=len(forecasts)),
            },
        }
    )
    write_json_atomic(args.audit_output, audit)
    print(
        f"ml_vine_quality={audit['status']} refits={len(refits)} "
        f"forecasts={len(forecasts)} whole_fallbacks={audit['whole_vine_fallback_refit_count']}"
    )
    print(f"Audit: {args.audit_output}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
