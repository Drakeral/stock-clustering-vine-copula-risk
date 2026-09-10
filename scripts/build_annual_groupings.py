#!/usr/bin/env python3
"""Build leakage-free annual GICS and hierarchical stock groupings.

Clusters for each evaluation year use only the preceding three calendar years.
The resulting group returns preserve the frozen daily-rebalanced equal-weight
stock portfolio exactly and are suitable inputs for the later marginal/copula
stage. Every artifact inherits the foundation gate's reporting scope.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path as _project_path,
        reporting_scope as _reporting_scope,
        sha256_file as _sha256,
        write_json_atomic as _write_json_atomic,
        write_parquet_atomic as _write_parquet_atomic,
    )
    from scripts.research_methods import (
        adjusted_rand_index,
        alphabet_issuer_composite,
        average_linkage_clusters,
        correlation_distance,
        dependence_gap,
        group_log_returns,
        group_simple_returns,
        normalized_mutual_information,
        pairwise_spearman,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import (
        PROJECT_ROOT,
        project_path as _project_path,
        reporting_scope as _reporting_scope,
        sha256_file as _sha256,
        write_json_atomic as _write_json_atomic,
        write_parquet_atomic as _write_parquet_atomic,
    )
    from research_methods import (
        adjusted_rand_index,
        alphabet_issuer_composite,
        average_linkage_clusters,
        correlation_distance,
        dependence_gap,
        group_log_returns,
        group_simple_returns,
        normalized_mutual_information,
        pairwise_spearman,
    )


def _gics_labels(universe: Mapping[str, Any]) -> dict[str, str]:
    rows = universe.get("constituents")
    if not isinstance(rows, list):
        raise ValueError("final universe must contain a constituents list")
    labels: dict[str, str] = {}
    for row in rows:
        if row.get("initial_universe_status") != "included":
            continue
        ticker = str(row["ticker"])
        if ticker in labels:
            raise ValueError(f"duplicate included ticker: {ticker}")
        labels[ticker] = str(row["gics_sector"])
    if not labels:
        raise ValueError("final universe contains no included securities")
    return labels


def _schedule_by_year(schedule: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = schedule.get("years")
    if not isinstance(rows, list) or not rows:
        raise ValueError("active universe must contain a non-empty years list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        year = int(row["year"])
        if year in result:
            raise ValueError(f"duplicate active-universe year: {year}")
        result[year] = row
    return result


def _validate_protocol(config: Mapping[str, Any]) -> Mapping[str, Any]:
    clustering = config.get("clustering")
    if not isinstance(clustering, Mapping):
        raise ValueError("model configuration has no clustering section")
    expected = {
        "primary_method": "average_linkage_spearman_distance",
        "distance": "sqrt((1-rho)/2)",
        "group_count_rule": "nonempty_gics_sector_count",
        "linkage_update": "unweighted_pair_group_average",
        "linkage_tie_break": "lexicographic_sorted_member_tuples",
        "cluster_label_order": "lexicographic_sorted_member_tuples",
        "nmi_normalization": "arithmetic_mean_entropy",
    }
    mismatches = {
        key: {"expected": value, "observed": clustering.get(key)}
        for key, value in expected.items()
        if clustering.get(key) != value
    }
    if mismatches:
        raise ValueError(f"unsupported clustering protocol: {mismatches}")
    return clustering


def _validate_arithmetic_binding(
    audit: Mapping[str, Any],
    returns_path: Path,
    active_universe_path: Path,
    universe_path: Path,
) -> None:
    """Refuse inputs that differ from the passed portfolio-arithmetic audit."""

    if audit.get("status") != "pass":
        raise RuntimeError("portfolio-arithmetic audit is not passed")
    expected = {
        "simple_return_panel": _sha256(returns_path),
        "active_universe": _sha256(active_universe_path),
        "final_universe": _sha256(universe_path),
    }
    recorded = audit.get("inputs", {})
    mismatches = [
        name for name, digest in expected.items() if recorded.get(name, {}).get("sha256") != digest
    ]
    if mismatches:
        raise RuntimeError(
            "clustering inputs differ from the portfolio-arithmetic audit: " + ",".join(mismatches)
        )


def _gap_record(result: Any) -> dict[str, Any]:
    return {
        "within_pair_mean": result.within_pair_mean,
        "between_pair_mean": result.between_pair_mean,
        "gap": result.gap,
        "within_pair_count": result.within_pair_count,
        "between_pair_count": result.between_pair_count,
        "contributing_within_groups": result.contributing_within_groups,
        "group_balanced_within_mean": result.group_balanced_within_mean,
        "group_balanced_between_mean": result.group_balanced_between_mean,
        "group_balanced_gap": result.group_balanced_gap,
    }


def _group_return_records(
    annual: pd.DataFrame,
    labels: Mapping[str, str],
    *,
    year: int,
    grouping_id: str,
    universe_variant: str,
    sample_role: str,
) -> pd.DataFrame:
    grouped = group_simple_returns(annual, labels)
    logged = group_log_returns(grouped)
    records: list[pd.DataFrame] = []
    total = int(grouped.group_sizes.sum())
    for group_id in grouped.simple_returns.columns:
        size = int(grouped.group_sizes[group_id])
        records.append(
            pd.DataFrame(
                {
                    "date": annual.index,
                    "year": year,
                    "universe_variant": universe_variant,
                    "sample_role": sample_role,
                    "grouping_id": grouping_id,
                    "group_id": group_id,
                    "group_size": size,
                    "portfolio_weight": size / total,
                    "simple_return": grouped.simple_returns[group_id].to_numpy(),
                    "log_return": logged[group_id].to_numpy(),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def build_annual_groupings(
    stock_simple_returns: pd.DataFrame,
    active_schedule: Mapping[str, Any],
    gics_labels: Mapping[str, str],
    clustering_config: Mapping[str, Any],
    *,
    reporting_scope: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Return assignments, long-form group returns, and clustering diagnostics."""

    panel = stock_simple_returns.copy()
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index), name="date")
    if panel.empty or panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("return panel must be non-empty with unique sorted dates")
    if panel.columns.has_duplicates:
        raise ValueError("return panel security columns must be unique")

    schedule = _schedule_by_year(active_schedule)
    start_year = int(clustering_config["evaluation_start_year"])
    end_year = int(clustering_config["evaluation_end_year"])
    training_years = int(clustering_config["training_window_calendar_years"])
    paired_fraction = float(clustering_config["minimum_paired_fraction"])
    if start_year > end_year or training_years <= 0:
        raise ValueError("invalid clustering evaluation or training window")

    assignment_years: list[dict[str, Any]] = []
    diagnostic_years: list[dict[str, Any]] = []
    group_return_frames: list[pd.DataFrame] = []
    previous_clusters: dict[str, str] | None = None
    previous_issuer_clusters: dict[str, str] | None = None
    previous_year: int | None = None

    for year in range(start_year, end_year + 1):
        if year not in schedule:
            raise ValueError(f"active-universe schedule is missing evaluation year {year}")
        schedule_row = schedule[year]
        active = [str(value) for value in schedule_row["active_tickers"]]
        if len(active) != len(set(active)) or int(schedule_row["security_count"]) != len(active):
            raise ValueError(f"invalid active set for {year}")
        missing_panel = sorted(set(active) - set(panel.columns))
        missing_gics = sorted(set(active) - set(gics_labels))
        if missing_panel or missing_gics:
            raise ValueError(
                f"active set is not represented for {year}; panel={missing_panel}, gics={missing_gics}"
            )
        active = sorted(active)
        annual_gics = {ticker: str(gics_labels[ticker]) for ticker in active}
        group_count = len(set(annual_gics.values()))
        rebalance_date = pd.Timestamp(str(schedule_row["rebalance_date"]))
        training_start = rebalance_date - pd.DateOffset(years=training_years)
        training = panel.loc[
            (panel.index >= training_start) & (panel.index < rebalance_date), active
        ]
        if training.empty:
            raise ValueError(f"empty training window for {year}")
        training_finite = np.isfinite(training.to_numpy(dtype=float)).all(axis=1)
        training_complete = training.loc[training_finite]
        if training_complete.empty:
            raise ValueError(f"training window has no complete return rows for {year}")
        training_correlation = pairwise_spearman(training, minimum_paired_fraction=paired_fraction)
        cluster_labels, merges = average_linkage_clusters(
            correlation_distance(training_correlation), group_count
        )
        issuer_training, issuer_gics = alphabet_issuer_composite(training, annual_gics)
        issuer_training_complete = issuer_training.loc[
            np.isfinite(issuer_training.to_numpy(dtype=float)).all(axis=1)
        ]
        issuer_training_correlation = pairwise_spearman(
            issuer_training, minimum_paired_fraction=paired_fraction
        )
        issuer_cluster_labels, issuer_merges = average_linkage_clusters(
            correlation_distance(issuer_training_correlation), group_count
        )

        annual = panel.loc[(panel.index >= rebalance_date) & (panel.index.year == year), active]
        if annual.empty:
            raise ValueError(f"empty evaluation window for {year}")
        if not np.isfinite(annual.to_numpy(dtype=float)).all():
            bad_dates = annual.index[~np.isfinite(annual.to_numpy(dtype=float)).all(axis=1)]
            raise ValueError(
                f"non-finite active returns in {year}: "
                + ",".join(date.date().isoformat() for date in bad_dates[:10])
            )
        evaluation_correlation = pairwise_spearman(annual, minimum_paired_fraction=paired_fraction)
        issuer_annual, issuer_evaluation_gics = alphabet_issuer_composite(annual, annual_gics)
        if issuer_evaluation_gics != issuer_gics:
            raise AssertionError("issuer-deduplicated GICS labels changed across windows")
        issuer_evaluation_correlation = pairwise_spearman(
            issuer_annual, minimum_paired_fraction=paired_fraction
        )
        gics_gap = dependence_gap(evaluation_correlation, annual_gics)
        cluster_gap = dependence_gap(evaluation_correlation, cluster_labels)
        issuer_gics_gap = dependence_gap(issuer_evaluation_correlation, issuer_evaluation_gics)
        issuer_cluster_gap = dependence_gap(issuer_evaluation_correlation, issuer_cluster_labels)

        gics_returns = group_simple_returns(annual, annual_gics)
        cluster_returns = group_simple_returns(annual, cluster_labels)
        direct = annual.mean(axis=1)
        gics_reconstructed = gics_returns.simple_returns.mul(
            gics_returns.group_sizes / len(active), axis="columns"
        ).sum(axis=1)
        cluster_reconstructed = cluster_returns.simple_returns.mul(
            cluster_returns.group_sizes / len(active), axis="columns"
        ).sum(axis=1)
        gics_error = float((direct - gics_reconstructed).abs().max())
        cluster_error = float((direct - cluster_reconstructed).abs().max())
        issuer_gics_returns = group_simple_returns(issuer_annual, issuer_evaluation_gics)
        issuer_cluster_returns = group_simple_returns(issuer_annual, issuer_cluster_labels)
        issuer_direct = issuer_annual.mean(axis=1)
        issuer_gics_reconstructed = issuer_gics_returns.simple_returns.mul(
            issuer_gics_returns.group_sizes / len(issuer_evaluation_gics), axis="columns"
        ).sum(axis=1)
        issuer_cluster_reconstructed = issuer_cluster_returns.simple_returns.mul(
            issuer_cluster_returns.group_sizes / len(issuer_evaluation_gics), axis="columns"
        ).sum(axis=1)
        issuer_gics_error = float((issuer_direct - issuer_gics_reconstructed).abs().max())
        issuer_cluster_error = float((issuer_direct - issuer_cluster_reconstructed).abs().max())
        if (
            max(
                gics_error,
                cluster_error,
                issuer_gics_error,
                issuer_cluster_error,
            )
            > 1e-12
        ):
            raise AssertionError(f"annual portfolio identity failed in {year}")

        intersection_count = None
        consecutive_ari = None
        if previous_clusters is not None:
            common = sorted(set(previous_clusters) & set(cluster_labels))
            intersection_count = len(common)
            consecutive_ari = adjusted_rand_index(
                {ticker: previous_clusters[ticker] for ticker in common},
                {ticker: cluster_labels[ticker] for ticker in common},
            )
        issuer_intersection_count = None
        issuer_consecutive_ari = None
        if previous_issuer_clusters is not None:
            issuer_common = sorted(set(previous_issuer_clusters) & set(issuer_cluster_labels))
            issuer_intersection_count = len(issuer_common)
            issuer_consecutive_ari = adjusted_rand_index(
                {ticker: previous_issuer_clusters[ticker] for ticker in issuer_common},
                {ticker: issuer_cluster_labels[ticker] for ticker in issuer_common},
            )
        nmi = normalized_mutual_information(cluster_labels, annual_gics)
        issuer_nmi = normalized_mutual_information(issuer_cluster_labels, issuer_evaluation_gics)
        cluster_sizes = {
            key: int(value)
            for key, value in pd.Series(cluster_labels).value_counts().sort_index().items()
        }
        gics_sizes = {
            key: int(value)
            for key, value in pd.Series(annual_gics).value_counts().sort_index().items()
        }
        assignments = [
            {
                "ticker": ticker,
                "gics_sector": annual_gics[ticker],
                "hierarchical_cluster": cluster_labels[ticker],
            }
            for ticker in active
        ]
        issuer_assignments = [
            {
                "position_id": position,
                "gics_sector": issuer_evaluation_gics[position],
                "hierarchical_cluster": issuer_cluster_labels[position],
            }
            for position in sorted(issuer_evaluation_gics)
        ]
        assignment_years.append(
            {
                "year": year,
                "rebalance_date": rebalance_date.date().isoformat(),
                "training_start_inclusive": training_start.date().isoformat(),
                "training_end_exclusive": rebalance_date.date().isoformat(),
                "training_observations": len(training),
                "training_group_return_observations": len(training_complete),
                "training_incomplete_dates_excluded_from_group_returns": [
                    date.date().isoformat() for date in training.index[~training_finite]
                ],
                "active_security_count": len(active),
                "group_count": group_count,
                "assignments": assignments,
                "linkage_merges": merges,
                "issuer_deduplicated_robustness": {
                    "position_count": len(issuer_evaluation_gics),
                    "alphabet_position_definition": "50_50_mean_GOOG_GOOGL_simple_returns",
                    "assignments": issuer_assignments,
                    "linkage_merges": issuer_merges,
                },
            }
        )
        diagnostic_years.append(
            {
                "year": year,
                "rebalance_date": rebalance_date.date().isoformat(),
                "training_start_inclusive": training_start.date().isoformat(),
                "training_end_exclusive": rebalance_date.date().isoformat(),
                "training_observations": len(training),
                "training_group_return_observations": len(training_complete),
                "training_incomplete_date_count": int((~training_finite).sum()),
                "evaluation_observations": len(annual),
                "active_security_count": len(active),
                "group_count": group_count,
                "gics_group_sizes": gics_sizes,
                "hierarchical_cluster_sizes": cluster_sizes,
                "gics_dependence_gap": _gap_record(gics_gap),
                "hierarchical_dependence_gap": _gap_record(cluster_gap),
                "cluster_minus_gics_pair_weighted_gap": cluster_gap.gap - gics_gap.gap,
                "nmi_hierarchical_vs_gics": nmi,
                "ari_vs_previous_year": consecutive_ari,
                "ari_previous_year": previous_year,
                "ari_active_intersection_count": intersection_count,
                "maximum_portfolio_identity_error": {
                    "gics_sector": gics_error,
                    "hierarchical_cluster": cluster_error,
                },
                "issuer_deduplicated_robustness": {
                    "position_count": len(issuer_evaluation_gics),
                    "gics_dependence_gap": _gap_record(issuer_gics_gap),
                    "hierarchical_dependence_gap": _gap_record(issuer_cluster_gap),
                    "cluster_minus_gics_pair_weighted_gap": (
                        issuer_cluster_gap.gap - issuer_gics_gap.gap
                    ),
                    "nmi_hierarchical_vs_gics": issuer_nmi,
                    "ari_vs_previous_year": issuer_consecutive_ari,
                    "ari_previous_year": previous_year,
                    "ari_active_intersection_count": issuer_intersection_count,
                    "maximum_portfolio_identity_error": {
                        "gics_sector": issuer_gics_error,
                        "hierarchical_cluster": issuer_cluster_error,
                    },
                },
            }
        )
        group_return_frames.extend(
            [
                _group_return_records(
                    training_complete,
                    annual_gics,
                    year=year,
                    grouping_id="gics_sector",
                    universe_variant="security_primary",
                    sample_role="training",
                ),
                _group_return_records(
                    training_complete,
                    cluster_labels,
                    year=year,
                    grouping_id="hierarchical_cluster",
                    universe_variant="security_primary",
                    sample_role="training",
                ),
                _group_return_records(
                    issuer_training_complete,
                    issuer_gics,
                    year=year,
                    grouping_id="gics_sector",
                    universe_variant="issuer_deduplicated_robustness",
                    sample_role="training",
                ),
                _group_return_records(
                    issuer_training_complete,
                    issuer_cluster_labels,
                    year=year,
                    grouping_id="hierarchical_cluster",
                    universe_variant="issuer_deduplicated_robustness",
                    sample_role="training",
                ),
                _group_return_records(
                    annual,
                    annual_gics,
                    year=year,
                    grouping_id="gics_sector",
                    universe_variant="security_primary",
                    sample_role="evaluation",
                ),
                _group_return_records(
                    annual,
                    cluster_labels,
                    year=year,
                    grouping_id="hierarchical_cluster",
                    universe_variant="security_primary",
                    sample_role="evaluation",
                ),
                _group_return_records(
                    issuer_annual,
                    issuer_evaluation_gics,
                    year=year,
                    grouping_id="gics_sector",
                    universe_variant="issuer_deduplicated_robustness",
                    sample_role="evaluation",
                ),
                _group_return_records(
                    issuer_annual,
                    issuer_cluster_labels,
                    year=year,
                    grouping_id="hierarchical_cluster",
                    universe_variant="issuer_deduplicated_robustness",
                    sample_role="evaluation",
                ),
            ]
        )
        previous_clusters = cluster_labels
        previous_issuer_clusters = issuer_cluster_labels
        previous_year = year

    group_returns = pd.concat(group_return_frames, ignore_index=True)
    assignment_payload = {
        "schema_version": 1,
        "artifact": "annual_group_assignments",
        "status": "pass",
        "reporting_scope": reporting_scope,
        "evaluation_years": [start_year, end_year],
        "grouping_ids": ["gics_sector", "hierarchical_cluster"],
        "universe_variants": [
            "security_primary",
            "issuer_deduplicated_robustness",
        ],
        "years": assignment_years,
    }
    differences = [row["cluster_minus_gics_pair_weighted_gap"] for row in diagnostic_years]
    issuer_differences = [
        row["issuer_deduplicated_robustness"]["cluster_minus_gics_pair_weighted_gap"]
        for row in diagnostic_years
    ]
    diagnostics = {
        "schema_version": 1,
        "gate_name": "annual_grouping_v1",
        "status": "pass",
        "reporting_scope": reporting_scope,
        "method": {
            "correlation": "spearman_average_ranks",
            "distance": "sqrt((1-rho)/2)",
            "linkage": "unweighted_pair_group_average",
            "linkage_tie_break": "lexicographic_sorted_member_tuples",
            "group_count_rule": "annual_nonempty_gics_sector_count",
            "training_window_calendar_years": training_years,
            "minimum_paired_fraction": paired_fraction,
            "nmi_normalization": "arithmetic_mean_entropy",
            "issuer_robustness": "equal_weight_GOOG_GOOGL_simple_returns",
        },
        "years": diagnostic_years,
        "summary": {
            "evaluation_year_count": len(diagnostic_years),
            "mean_cluster_minus_gics_pair_weighted_gap": float(np.mean(differences)),
            "minimum_cluster_minus_gics_pair_weighted_gap": float(np.min(differences)),
            "maximum_cluster_minus_gics_pair_weighted_gap": float(np.max(differences)),
            "mean_issuer_deduplicated_cluster_minus_gics_pair_weighted_gap": float(
                np.mean(issuer_differences)
            ),
            "h1_inference_status": "not_run_bootstrap_is_later_evaluation_step",
        },
        "issues": [],
    }
    return assignment_payload, group_returns, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/portfolio_constituent_simple_returns.parquet",
    )
    parser.add_argument(
        "--active-universe",
        type=Path,
        default=PROJECT_ROOT / "data/processed/active_universe_by_year.json",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=PROJECT_ROOT / "data/processed/final_universe.json",
    )
    parser.add_argument(
        "--model-config", type=Path, default=PROJECT_ROOT / "config/model_config.toml"
    )
    parser.add_argument(
        "--foundation-status",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_gate_status.json",
    )
    parser.add_argument(
        "--portfolio-arithmetic-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/portfolio_arithmetic.json",
    )
    parser.add_argument(
        "--assignments-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/annual_group_assignments.json",
    )
    parser.add_argument(
        "--group-returns-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/annual_group_returns.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/clustering_diagnostics.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    clustering_config = _validate_protocol(model_config)
    gate = json.loads(args.foundation_status.read_text(encoding="utf-8"))
    scope = _reporting_scope(gate)
    arithmetic_audit = json.loads(args.portfolio_arithmetic_audit.read_text(encoding="utf-8"))
    _validate_arithmetic_binding(
        arithmetic_audit, args.returns, args.active_universe, args.universe
    )
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    schedule = json.loads(args.active_universe.read_text(encoding="utf-8"))
    panel = pd.read_parquet(args.returns)

    assignments, group_returns, audit = build_annual_groupings(
        panel,
        schedule,
        _gics_labels(universe),
        clustering_config,
        reporting_scope=scope,
    )
    _write_json_atomic(args.assignments_output, assignments)
    _write_parquet_atomic(args.group_returns_output, group_returns)
    audit["inputs"] = {
        "simple_return_panel": {
            "path": _project_path(args.returns),
            "sha256": _sha256(args.returns),
        },
        "active_universe": {
            "path": _project_path(args.active_universe),
            "sha256": _sha256(args.active_universe),
        },
        "final_universe": {
            "path": _project_path(args.universe),
            "sha256": _sha256(args.universe),
        },
        "model_config": {
            "path": _project_path(args.model_config),
            "sha256": _sha256(args.model_config),
        },
        "foundation_status": {
            "path": _project_path(args.foundation_status),
            "sha256": _sha256(args.foundation_status),
        },
        "portfolio_arithmetic_audit": {
            "path": _project_path(args.portfolio_arithmetic_audit),
            "sha256": _sha256(args.portfolio_arithmetic_audit),
        },
    }
    audit["outputs"] = {
        "annual_group_assignments": {
            "path": _project_path(args.assignments_output),
            "sha256": _sha256(args.assignments_output),
        },
        "annual_group_returns": {
            "path": _project_path(args.group_returns_output),
            "sha256": _sha256(args.group_returns_output),
            "rows": len(group_returns),
        },
    }
    _write_json_atomic(args.audit_output, audit)
    print(
        f"annual_grouping={audit['status']} scope={scope} "
        f"years={audit['summary']['evaluation_year_count']} "
        f"group_return_rows={len(group_returns)}"
    )
    print(f"Audit: {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
