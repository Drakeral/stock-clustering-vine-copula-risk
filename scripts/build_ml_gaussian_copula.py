#!/usr/bin/env python3
"""Fit M5/M7 Gaussian copulas using the frozen primary simulation uniforms."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.build_gaussian_copula import (
        _validate_marginal_binding,
        _validate_protocol,
        build_gaussian_outputs,
    )
    from scripts.build_ml_groupings import validate_ml_protocol
    from scripts.ml_risk_common import (
        ML_GAUSSIAN_MODEL_BY_GROUPING,
        ML_RISK_ANALYSIS_ROLE,
        artifact_record,
        load_json,
        require_audit_output,
        require_same_seed_manifest,
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
    from build_gaussian_copula import (
        _validate_marginal_binding,
        _validate_protocol,
        build_gaussian_outputs,
    )
    from build_ml_groupings import validate_ml_protocol
    from ml_risk_common import (
        ML_GAUSSIAN_MODEL_BY_GROUPING,
        ML_RISK_ANALYSIS_ROLE,
        artifact_record,
        load_json,
        require_audit_output,
        require_same_seed_manifest,
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
        "--marginal-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_marginal_model_quality.json",
    )
    parser.add_argument(
        "--primary-seed-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/simulation_seed_manifest.json",
    )
    parser.add_argument(
        "--primary-gaussian-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/gaussian_copula_quality.json",
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
        default=PROJECT_ROOT / "data/processed/ml_gaussian_copula_refits.parquet",
    )
    parser.add_argument(
        "--forecasts-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_gaussian_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_gaussian_copula_quality.json",
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
    _validate_protocol(model_config)
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
    primary_gaussian_audit = load_json(args.primary_gaussian_audit)
    require_current_hash_records(
        primary_gaussian_audit,
        source_name=project_path(args.primary_gaussian_audit),
    )
    require_audit_output(
        primary_gaussian_audit,
        audit_name="primary Gaussian audit",
        output_key="simulation_seed_manifest",
        output_path=args.primary_seed_manifest,
        reporting_scope=scope,
    )
    refits, forecasts, generated_seeds, audit = build_gaussian_outputs(
        pd.read_parquet(args.training_pits),
        pd.read_parquet(args.marginal_refits),
        pd.read_parquet(args.daily_margins),
        pd.read_parquet(args.group_returns),
        model_config,
        universe_variant=str(ml_config["grouping"]["universe_variant"]),
        progress_every=args.progress_every,
        model_by_grouping=ML_GAUSSIAN_MODEL_BY_GROUPING,
    )
    primary_seeds = load_json(args.primary_seed_manifest)
    require_same_seed_manifest(generated_seeds, primary_seeds)
    write_parquet_atomic(args.refits_output, refits)
    write_parquet_atomic(args.forecasts_output, forecasts)
    audit.update(
        {
            "gate_name": "ml_gaussian_copula_quality_v1",
            "analysis_role": ML_RISK_ANALYSIS_ROLE,
            "reporting_scope": scope,
            "method": {
                "estimator": model_config["gaussian"]["estimator"],
                "repair_method": model_config["gaussian"]["repair_method"],
                "eigenvalue_floor": model_config["gaussian"]["eigenvalue_floor"],
                "factorization": model_config["gaussian"]["factorization"],
                "simulation_bit_generator": model_config["simulation"]["bit_generator"],
                "common_random_numbers": str(
                    ml_config["downstream_risk_models"]["common_random_numbers"]
                ),
            },
            "inputs": {
                "monthly_copula_training_pits": artifact_record(args.training_pits),
                "marginal_refits": artifact_record(args.marginal_refits),
                "marginal_daily_forecasts": artifact_record(args.daily_margins),
                "annual_group_returns": artifact_record(args.group_returns),
                "marginal_model_quality": artifact_record(args.marginal_audit),
                "primary_simulation_seed_manifest": artifact_record(args.primary_seed_manifest),
                "primary_gaussian_audit": artifact_record(args.primary_gaussian_audit),
                "model_config": artifact_record(args.model_config),
                "ml_config": artifact_record(args.ml_config),
                "foundation_status": artifact_record(args.foundation_status),
            },
            "outputs": {
                "gaussian_copula_refits": artifact_record(args.refits_output, rows=len(refits)),
                "gaussian_risk_forecasts": artifact_record(
                    args.forecasts_output, rows=len(forecasts)
                ),
                "simulation_seed_manifest": {
                    **artifact_record(args.primary_seed_manifest),
                    "records": len(primary_seeds["records"]),
                },
            },
        }
    )
    write_json_atomic(args.audit_output, audit)
    print(
        f"ml_gaussian_quality={audit['status']} refits={len(refits)} "
        f"forecasts={len(forecasts)} repaired={audit['repaired_refit_count']}"
    )
    print(f"Audit: {args.audit_output}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
