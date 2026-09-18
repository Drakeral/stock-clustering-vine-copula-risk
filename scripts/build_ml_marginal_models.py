#!/usr/bin/env python3
"""Fit M5--M8 exploratory marginal inputs without touching core artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.build_marginal_models import _validate_protocol, build_marginal_outputs
    from scripts.build_ml_groupings import validate_ml_protocol
    from scripts.ml_risk_common import (
        ML_RISK_ANALYSIS_ROLE,
        artifact_record,
        load_json,
        require_audit_output,
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
    from build_marginal_models import _validate_protocol, build_marginal_outputs
    from build_ml_groupings import validate_ml_protocol
    from ml_risk_common import (
        ML_RISK_ANALYSIS_ROLE,
        artifact_record,
        load_json,
        require_audit_output,
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
        "--group-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_annual_group_returns.parquet",
    )
    parser.add_argument(
        "--grouping-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_clustering_diagnostics.json",
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
        default=PROJECT_ROOT / "data/processed/ml_marginal_refits.parquet",
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_marginal_daily_forecasts.parquet",
    )
    parser.add_argument(
        "--training-pits-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_monthly_copula_training_pits.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_marginal_model_quality.json",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    with args.ml_config.open("rb") as handle:
        ml_config = validate_ml_protocol(tomllib.load(handle))
    marginal = _validate_protocol(model_config)
    foundation = load_json(args.foundation_status)
    require_current_hash_records(
        foundation,
        source_name=project_path(args.foundation_status),
    )
    scope = reporting_scope(foundation)
    grouping_audit = load_json(args.grouping_audit)
    require_current_hash_records(
        grouping_audit,
        source_name=project_path(args.grouping_audit),
    )
    require_audit_output(
        grouping_audit,
        audit_name="ML grouping audit",
        output_key="ml_annual_group_returns",
        output_path=args.group_returns,
        reporting_scope=scope,
        analysis_role="exploratory_unsupervised_robustness",
    )
    refits, daily, training_pits, audit = build_marginal_outputs(
        pd.read_parquet(args.group_returns),
        model_config,
        universe_variant=str(ml_config["grouping"]["universe_variant"]),
        progress_every=args.progress_every,
    )
    write_parquet_atomic(args.refits_output, refits)
    write_parquet_atomic(args.daily_output, daily)
    write_parquet_atomic(args.training_pits_output, training_pits)
    audit.update(
        {
            "gate_name": "ml_marginal_model_quality_v1",
            "analysis_role": ML_RISK_ANALYSIS_ROLE,
            "reporting_scope": scope,
            "model_ids": ["M5", "M6", "M7", "M8"],
            "method": {
                "primary": marginal["primary"],
                "fallback_order": marginal["fallback_order"],
                "estimation_return_scale": marginal["estimation_return_scale"],
                "ewma_lambda": marginal["ewma_lambda"],
                "pit_clip_lower": marginal["pit_clip_lower"],
                "pit_clip_upper": marginal["pit_clip_upper"],
            },
            "inputs": {
                "annual_group_returns": artifact_record(args.group_returns),
                "ml_grouping_audit": artifact_record(args.grouping_audit),
                "model_config": artifact_record(args.model_config),
                "ml_config": artifact_record(args.ml_config),
                "foundation_status": artifact_record(args.foundation_status),
            },
            "outputs": {
                "marginal_refits": artifact_record(args.refits_output, rows=len(refits)),
                "marginal_daily_forecasts": artifact_record(args.daily_output, rows=len(daily)),
                "monthly_copula_training_pits": artifact_record(
                    args.training_pits_output, rows=len(training_pits)
                ),
            },
        }
    )
    write_json_atomic(args.audit_output, audit)
    print(
        f"ml_marginal_quality={audit['status']} refits={len(refits)} "
        f"daily={len(daily)} training_pits={len(training_pits)} "
        f"ewma_fraction={audit['ewma_fit_fraction']:.6f}"
    )
    print(f"Audit: {args.audit_output}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
