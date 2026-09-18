#!/usr/bin/env python3
"""Construct annually reset, buy-and-hold GICS/cluster-balanced portfolios."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.ml_risk_common import artifact_record
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        write_json_atomic,
        write_parquet_atomic,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from ml_risk_common import artifact_record
    from pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        write_json_atomic,
        write_parquet_atomic,
    )


ANALYSIS_ROLE = "exploratory_portfolio_weighting_robustness"
GROUP_RETURN_COLUMNS = (
    "date",
    "year",
    "universe_variant",
    "sample_role",
    "grouping_id",
    "group_id",
    "group_size",
    "portfolio_weight",
    "simple_return",
    "log_return",
)
PORTFOLIO_RETURN_COLUMNS = (
    "date",
    "year",
    "universe_variant",
    "sample_role",
    "portfolio_id",
    "grouping_id",
    "active_security_count",
    "group_count",
    "simple_return",
    "log_return",
)


@dataclass(frozen=True)
class PortfolioSpec:
    portfolio_id: str
    input_grouping_id: str
    model_grouping_id: str
    historical_model_id: str
    gaussian_model_id: str
    vine_model_id: str


@dataclass(frozen=True)
class BalancedSegment:
    group_simple_returns: pd.DataFrame
    group_pre_return_weights: pd.DataFrame
    portfolio_simple_returns: pd.Series
    group_sizes: pd.Series
    reset_dates: tuple[pd.Timestamp, ...]
    maximum_identity_error: float


@dataclass(frozen=True)
class AnnualSample:
    year: int
    active: tuple[str, ...]
    training: pd.DataFrame
    evaluation: pd.DataFrame
    assignments_by_ticker: Mapping[str, Mapping[str, Any]]
    expected_group_count: int


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a table")
    return cast(Mapping[str, Any], value)


def load_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), project_path(path))


def _require_exact(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(f"{name} differs from the frozen group-balanced protocol")


def validate_group_balanced_protocol(config: Mapping[str, Any]) -> tuple[PortfolioSpec, ...]:
    """Validate the complete pre-result portfolio and model specification."""

    _require_exact(config.get("schema_version"), 1, "schema_version")
    _require_exact(
        config.get("protocol_status"),
        "frozen_before_group_balanced_results",
        "protocol_status",
    )
    _require_exact(config.get("analysis_role"), ANALYSIS_ROLE, "analysis_role")
    portfolio = _mapping(config.get("portfolio"), "portfolio")
    expected_portfolio = {
        "return_scale": "simple_total_return",
        "rebalance_frequency": "annual_first_trading_day",
        "initial_group_weight": "equal_one_over_nonempty_group_count",
        "initial_within_group_weight": "equal_one_over_group_size",
        "between_rebalance_rule": "buy_and_hold_total_return_weights_drift",
        "daily_weight_timing": "pre_return",
        "dividend_treatment": "inherited_total_return_reinvestment",
        "verified_halt_and_terminal_cash_return": 0.0,
        "membership_and_labels": "hold_primary_annual_assignments_fixed",
        "training_backcast": "evaluation_year_active_set_and_labels",
        "training_weight_resets": "first_available_session_of_each_calendar_year",
        "transaction_costs": 0.0,
        "portfolio_identity_tolerance": 1e-12,
    }
    for key, expected in expected_portfolio.items():
        _require_exact(portfolio.get(key), expected, f"portfolio.{key}")
    models = _mapping(config.get("models"), "models")
    expected_models = {
        "marginal_copula_simulation_protocol": "inherit_config/model_config.toml",
        "historical_window_protocol": "inherit_config/model_config.toml",
        "vine_truncation_tree": 3,
        "common_random_numbers": "reuse_primary_simulation_seed_manifest",
        "full_vine_cross_robustness": False,
        "ml_grouping_cross_robustness": False,
    }
    for key, expected in expected_models.items():
        _require_exact(models.get(key), expected, f"models.{key}")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    expected_evaluation = {
        "loss_scores": ["quantile_loss_95", "quantile_loss_99", "fz0_975"],
        "dm_comparison": "vine_minus_gaussian_within_same_portfolio",
        "dm_hac_lag": 7,
        "dm_multiplicity": "holm_familywise",
        "dm_family_size": 6,
        "calibration_confidence_levels": [0.95, 0.99],
        "calibration_tests": [
            "kupiec_unconditional_coverage",
            "christoffersen_independence",
        ],
        "calibration_multiplicity": "holm_familywise",
        "calibration_family_size": 24,
        "familywise_alpha": 0.05,
        "ranking": "separate_within_each_portfolio_equal_weight_average_primary_score_rank",
        "hypothesis_treatment": "robustness_does_not_revise_core_H1_H2_H3",
        "result_role": "exploratory_robustness",
    }
    for key, expected in expected_evaluation.items():
        _require_exact(evaluation.get(key), expected, f"evaluation.{key}")
    raw_specs = config.get("portfolios")
    if not isinstance(raw_specs, list):
        raise TypeError("portfolios must be an array of tables")
    specs = tuple(
        PortfolioSpec(**{key: str(value) for key, value in _mapping(row, "portfolio").items()})
        for row in raw_specs
    )
    expected_specs = (
        PortfolioSpec(
            "gics_balanced",
            "gics_sector",
            "gics_balanced",
            "B_GICS_HS",
            "B_GICS_GAUSSIAN",
            "B_GICS_VINE",
        ),
        PortfolioSpec(
            "hierarchical_balanced",
            "hierarchical_cluster",
            "hierarchical_balanced",
            "B_HIER_HS",
            "B_HIER_GAUSSIAN",
            "B_HIER_VINE",
        ),
    )
    _require_exact(specs, expected_specs, "portfolios")
    return specs


def _validated_return_segment(
    stock_simple_returns: pd.DataFrame,
    labels: Mapping[str, str],
    tolerance: float,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("portfolio identity tolerance must be finite and positive")
    frame = stock_simple_returns.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"), name="date")
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("stock return segment must have unique, sorted dates")
    if frame.columns.has_duplicates or not all(isinstance(column, str) for column in frame.columns):
        raise ValueError("stock return columns must be unique strings")
    if set(frame.columns) != set(labels):
        raise ValueError("group labels must match the stock return columns exactly")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or bool((values <= -1.0).any()):
        raise ValueError("buy-and-hold stock returns must be finite and greater than -1")
    return frame, values, frame.columns.tolist()


def _group_memberships(
    columns: list[str], labels: Mapping[str, str]
) -> tuple[list[str], dict[str, np.ndarray], pd.Series]:
    normalized_labels = {ticker: str(labels[ticker]).strip() for ticker in columns}
    if any(not group for group in normalized_labels.values()):
        raise ValueError("group labels must be non-empty strings")
    groups = sorted(set(normalized_labels.values()))
    member_indices = {
        group: np.asarray(
            [index for index, ticker in enumerate(columns) if normalized_labels[ticker] == group],
            dtype=int,
        )
        for group in groups
    }
    group_sizes = pd.Series(
        {group: len(indices) for group, indices in member_indices.items()},
        name="group_size",
        dtype="int64",
    )
    return groups, member_indices, group_sizes


def _equal_group_security_weights(
    security_count: int, member_indices: Mapping[str, np.ndarray]
) -> np.ndarray:
    weights = np.empty(security_count, dtype=float)
    for indices in member_indices.values():
        weights[indices] = 1.0 / (len(member_indices) * len(indices))
    return weights


def _period_group_values(
    row: np.ndarray,
    weights: np.ndarray,
    groups: list[str],
    member_indices: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    period_returns = np.empty(len(groups), dtype=float)
    period_weights = np.empty(len(groups), dtype=float)
    reconstructed = 0.0
    for group_number, group in enumerate(groups):
        indices = member_indices[group]
        group_weight = float(weights[indices].sum())
        group_return = float((weights[indices] / group_weight) @ row[indices])
        period_weights[group_number] = group_weight
        period_returns[group_number] = group_return
        reconstructed += group_weight * group_return
    return period_returns, period_weights, reconstructed


def _advance_security_weights(
    weights: np.ndarray,
    row: np.ndarray,
    portfolio_return: float,
    tolerance: float,
) -> np.ndarray:
    wealth_multiplier = 1.0 + portfolio_return
    if wealth_multiplier <= 0:
        raise ValueError("buy-and-hold portfolio wealth became nonpositive")
    updated = weights * (1.0 + row) / wealth_multiplier
    if not np.isfinite(updated).all() or not np.isclose(
        updated.sum(), 1.0, rtol=0.0, atol=tolerance
    ):
        raise ValueError("buy-and-hold security weights became invalid")
    return updated


def group_balanced_buy_and_hold(
    stock_simple_returns: pd.DataFrame,
    labels: Mapping[str, str],
    *,
    tolerance: float = 1e-12,
) -> BalancedSegment:
    """Build a self-financing equal-group portfolio with annual weight resets."""

    frame, values, columns = _validated_return_segment(stock_simple_returns, labels, tolerance)
    groups, member_indices, group_sizes = _group_memberships(columns, labels)
    group_returns = np.empty((len(frame), len(groups)), dtype=float)
    group_weights = np.empty((len(frame), len(groups)), dtype=float)
    portfolio_returns = np.empty(len(frame), dtype=float)
    reset_dates: list[pd.Timestamp] = []
    weights = np.empty(len(columns), dtype=float)
    previous_calendar_year: int | None = None
    maximum_error = 0.0
    for row_number, (date, row) in enumerate(zip(frame.index, values, strict=True)):
        calendar_year = int(date.year)
        if calendar_year != previous_calendar_year:
            weights = _equal_group_security_weights(len(columns), member_indices)
            reset_dates.append(pd.Timestamp(date))
            previous_calendar_year = calendar_year
        portfolio_return = float(weights @ row)
        portfolio_returns[row_number] = portfolio_return
        period_returns, period_weights, reconstructed = _period_group_values(
            row, weights, groups, member_indices
        )
        group_returns[row_number] = period_returns
        group_weights[row_number] = period_weights
        error = abs(reconstructed - portfolio_return)
        maximum_error = max(maximum_error, error)
        if error > tolerance:
            raise AssertionError(f"group-balanced portfolio identity error {error}")
        weights = _advance_security_weights(weights, row, portfolio_return, tolerance)
    return BalancedSegment(
        group_simple_returns=pd.DataFrame(group_returns, index=frame.index, columns=groups),
        group_pre_return_weights=pd.DataFrame(group_weights, index=frame.index, columns=groups),
        portfolio_simple_returns=pd.Series(
            portfolio_returns, index=frame.index, name="simple_return"
        ),
        group_sizes=group_sizes,
        reset_dates=tuple(reset_dates),
        maximum_identity_error=maximum_error,
    )


def _group_records(
    segment: BalancedSegment,
    *,
    evaluation_year: int,
    sample_role: str,
    grouping_id: str,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for group_id in segment.group_simple_returns.columns:
        simple = segment.group_simple_returns[group_id]
        records.append(
            pd.DataFrame(
                {
                    "date": simple.index,
                    "year": evaluation_year,
                    "universe_variant": "security_primary",
                    "sample_role": sample_role,
                    "grouping_id": grouping_id,
                    "group_id": group_id,
                    "group_size": int(segment.group_sizes[group_id]),
                    "portfolio_weight": segment.group_pre_return_weights[group_id].to_numpy(),
                    "simple_return": simple.to_numpy(),
                    "log_return": np.log1p(simple.to_numpy()),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def _portfolio_records(
    segment: BalancedSegment,
    *,
    evaluation_year: int,
    sample_role: str,
    spec: PortfolioSpec,
) -> pd.DataFrame:
    simple = segment.portfolio_simple_returns
    return pd.DataFrame(
        {
            "date": simple.index,
            "year": evaluation_year,
            "universe_variant": "security_primary",
            "sample_role": sample_role,
            "portfolio_id": spec.portfolio_id,
            "grouping_id": spec.model_grouping_id,
            "active_security_count": int(segment.group_sizes.sum()),
            "group_count": len(segment.group_sizes),
            "simple_return": simple.to_numpy(),
            "log_return": np.log1p(simple.to_numpy()),
        }
    )


def _assignment_years(assignments: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    years = assignments.get("years")
    if not isinstance(years, list) or not years:
        raise TypeError("assignment artifact must contain annual records")
    normalized = [_mapping(item, "assignment year") for item in years]
    observed = [int(item["year"]) for item in normalized]
    if observed != list(range(2020, 2026)):
        raise ValueError("group-balanced portfolios require assignment years 2020--2025")
    return normalized


def _annual_sample(panel: pd.DataFrame, annual: Mapping[str, Any]) -> AnnualSample:
    year = int(annual["year"])
    assignment_rows = annual.get("assignments")
    if not isinstance(assignment_rows, list):
        raise TypeError("annual record must contain assignments")
    normalized = [_mapping(row, "security assignment") for row in assignment_rows]
    active = tuple(sorted(str(row["ticker"]) for row in normalized))
    if (
        not active
        or len(active) != len(set(active))
        or len(active) != int(annual["active_security_count"])
    ):
        raise ValueError(f"invalid active assignment set for {year}")
    missing = sorted(set(active) - set(panel.columns))
    if missing:
        raise ValueError(f"stock return panel is missing active securities for {year}: {missing}")
    training_start = pd.Timestamp(str(annual["training_start_inclusive"]))
    rebalance_date = pd.Timestamp(str(annual["rebalance_date"]))
    if training_start >= rebalance_date or rebalance_date.year != year:
        raise ValueError(f"invalid group-balanced date bounds for {year}")
    training = panel.loc[
        (panel.index >= training_start) & (panel.index < rebalance_date), list(active)
    ]
    training = training.loc[np.isfinite(training.to_numpy(dtype=float)).all(axis=1)]
    evaluation = panel.loc[
        (panel.index >= rebalance_date) & (panel.index.year == year), list(active)
    ]
    if training.empty or evaluation.empty or evaluation.index[0] != rebalance_date:
        raise ValueError(f"empty or misaligned group-balanced sample for {year}")
    if not np.isfinite(evaluation.to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite group-balanced evaluation return for {year}")
    assignments_by_ticker = {str(row["ticker"]): row for row in normalized}
    return AnnualSample(
        year=year,
        active=active,
        training=training,
        evaluation=evaluation,
        assignments_by_ticker=assignments_by_ticker,
        expected_group_count=int(annual["group_count"]),
    )


def _portfolio_spec_outputs(
    sample: AnnualSample,
    spec: PortfolioSpec,
    tolerance: float,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, Any]]:
    labels: dict[str, str] = {}
    for ticker in sample.active:
        value = sample.assignments_by_ticker[ticker].get(spec.input_grouping_id)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{spec.portfolio_id} has an invalid group label for {ticker}/{sample.year}"
            )
        labels[ticker] = value.strip()
    if len(set(labels.values())) != sample.expected_group_count:
        raise ValueError(f"{spec.portfolio_id} has the wrong group count in {sample.year}")
    training_segment = group_balanced_buy_and_hold(sample.training, labels, tolerance=tolerance)
    evaluation_segment = group_balanced_buy_and_hold(sample.evaluation, labels, tolerance=tolerance)
    group_frames = [
        _group_records(
            training_segment,
            evaluation_year=sample.year,
            sample_role="training",
            grouping_id=spec.model_grouping_id,
        ),
        _group_records(
            evaluation_segment,
            evaluation_year=sample.year,
            sample_role="evaluation",
            grouping_id=spec.model_grouping_id,
        ),
    ]
    portfolio_frames = [
        _portfolio_records(
            training_segment,
            evaluation_year=sample.year,
            sample_role="training",
            spec=spec,
        ),
        _portfolio_records(
            evaluation_segment,
            evaluation_year=sample.year,
            sample_role="evaluation",
            spec=spec,
        ),
    ]
    diagnostic = {
        "year": sample.year,
        "portfolio_id": spec.portfolio_id,
        "active_security_count": len(sample.active),
        "group_count": len(set(labels.values())),
        "training_observations": len(sample.training),
        "evaluation_observations": len(sample.evaluation),
        "training_calendar_reset_dates": [
            date.date().isoformat() for date in training_segment.reset_dates
        ],
        "evaluation_reset_date": evaluation_segment.reset_dates[0].date().isoformat(),
        "maximum_portfolio_identity_error": max(
            training_segment.maximum_identity_error,
            evaluation_segment.maximum_identity_error,
        ),
        "evaluation_minimum_group_weight": float(
            evaluation_segment.group_pre_return_weights.min().min()
        ),
        "evaluation_maximum_group_weight": float(
            evaluation_segment.group_pre_return_weights.max().max()
        ),
    }
    return group_frames, portfolio_frames, diagnostic


def _construction_issues(
    group_returns: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    maximum_identity: float,
    tolerance: float,
) -> list[str]:
    issues: list[str] = []
    group_key = ["date", "year", "sample_role", "grouping_id", "group_id"]
    portfolio_key = ["date", "year", "sample_role", "portfolio_id"]
    if group_returns.duplicated(group_key).any():
        issues.append("duplicate_group_return")
    if portfolio_returns.duplicated(portfolio_key).any():
        issues.append("duplicate_portfolio_return")
    group_numeric = group_returns[["portfolio_weight", "simple_return", "log_return"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(group_numeric).all():
        issues.append("nonfinite_group_return")
    portfolio_numeric = portfolio_returns[["simple_return", "log_return"]].to_numpy(dtype=float)
    if not np.isfinite(portfolio_numeric).all():
        issues.append("nonfinite_portfolio_return")
    if maximum_identity > tolerance:
        issues.append("portfolio_identity_failed")
    return issues


def build_group_balanced_outputs(
    stock_simple_returns: pd.DataFrame,
    assignments: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build training/evaluation group and portfolio return panels."""

    specs = validate_group_balanced_protocol(config)
    tolerance = float(config["portfolio"]["portfolio_identity_tolerance"])
    panel = stock_simple_returns.copy()
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index, errors="raise"), name="date")
    if (
        panel.empty
        or panel.index.has_duplicates
        or panel.columns.has_duplicates
        or not panel.index.is_monotonic_increasing
    ):
        raise ValueError("stock return panel must have unique, sorted dates")
    group_frames: list[pd.DataFrame] = []
    portfolio_frames: list[pd.DataFrame] = []
    annual_audits: list[dict[str, Any]] = []
    for annual in _assignment_years(assignments):
        sample = _annual_sample(panel, annual)
        for spec in specs:
            groups, portfolios, diagnostic = _portfolio_spec_outputs(sample, spec, tolerance)
            group_frames.extend(groups)
            portfolio_frames.extend(portfolios)
            annual_audits.append(diagnostic)
    group_returns = pd.concat(group_frames, ignore_index=True).loc[:, GROUP_RETURN_COLUMNS]
    group_returns = group_returns.sort_values(["year", "date", "grouping_id", "group_id"])
    portfolio_returns = pd.concat(portfolio_frames, ignore_index=True).loc[
        :, PORTFOLIO_RETURN_COLUMNS
    ]
    portfolio_returns = portfolio_returns.sort_values(["year", "date", "portfolio_id"])
    maximum_identity = max(
        float(record["maximum_portfolio_identity_error"]) for record in annual_audits
    )
    issues = _construction_issues(group_returns, portfolio_returns, maximum_identity, tolerance)
    audit = {
        "schema_version": 1,
        "gate_name": "group_balanced_portfolio_construction_v1",
        "analysis_role": ANALYSIS_ROLE,
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "portfolio_ids": [spec.portfolio_id for spec in specs],
        "evaluation_years": [2020, 2025],
        "group_return_rows": len(group_returns),
        "portfolio_return_rows": len(portfolio_returns),
        "maximum_portfolio_identity_error": maximum_identity,
        "portfolio_identity_tolerance": tolerance,
        "annual_diagnostics": annual_audits,
    }
    return group_returns.reset_index(drop=True), portfolio_returns.reset_index(drop=True), audit


def _require_binding(
    audit: Mapping[str, Any], section: str, key: str, path: Path, audit_name: str
) -> None:
    require_current_hash_records(audit, source_name=audit_name)
    if audit.get("status") != "pass":
        raise RuntimeError(f"{audit_name} is not passed")
    record = _mapping(_mapping(audit.get(section), section).get(key), key)
    if record.get("sha256") != artifact_record(path)["sha256"]:
        raise RuntimeError(f"{audit_name} does not bind {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/portfolio_constituent_simple_returns.parquet",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=PROJECT_ROOT / "data/processed/annual_group_assignments.json",
    )
    parser.add_argument(
        "--clustering-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/clustering_diagnostics.json",
    )
    parser.add_argument(
        "--foundation-status",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_gate_status.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/group_balanced_robustness.toml",
    )
    parser.add_argument(
        "--group-returns-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/group_balanced_group_returns.parquet",
    )
    parser.add_argument(
        "--portfolio-returns-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/group_balanced_portfolio_returns.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/group_balanced_portfolio_construction.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    validate_group_balanced_protocol(config)
    foundation = load_json(args.foundation_status)
    require_current_hash_records(foundation, source_name=project_path(args.foundation_status))
    scope = reporting_scope(foundation)
    clustering_audit = load_json(args.clustering_audit)
    _require_binding(
        clustering_audit,
        "outputs",
        "annual_group_assignments",
        args.assignments,
        project_path(args.clustering_audit),
    )
    _require_binding(
        clustering_audit,
        "inputs",
        "simple_return_panel",
        args.returns,
        project_path(args.clustering_audit),
    )
    group_returns, portfolio_returns, audit = build_group_balanced_outputs(
        pd.read_parquet(args.returns), load_json(args.assignments), config
    )
    write_parquet_atomic(args.group_returns_output, group_returns)
    write_parquet_atomic(args.portfolio_returns_output, portfolio_returns)
    audit.update(
        {
            "reporting_scope": scope,
            "method": dict(config["portfolio"]),
            "inputs": {
                "stock_simple_returns": artifact_record(args.returns),
                "annual_group_assignments": artifact_record(args.assignments),
                "clustering_audit": artifact_record(args.clustering_audit),
                "foundation_status": artifact_record(args.foundation_status),
                "group_balanced_config": artifact_record(args.config),
            },
            "outputs": {
                "group_returns": artifact_record(
                    args.group_returns_output, rows=len(group_returns)
                ),
                "portfolio_returns": artifact_record(
                    args.portfolio_returns_output, rows=len(portfolio_returns)
                ),
            },
        }
    )
    write_json_atomic(args.audit_output, audit)
    print(
        f"group_balanced_construction={audit['status']} group_rows={len(group_returns)} "
        f"portfolio_rows={len(portfolio_returns)} identity={audit['maximum_portfolio_identity_error']:.3e}"
    )
    print(f"Audit: {args.audit_output}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
