#!/usr/bin/env python3
"""Build leakage-free annual spectral and PCA-plus-k-means groupings.

This exploratory extension changes only the annual grouping method. It uses the
same active securities, trailing windows, group count, return arithmetic, and
reporting scope as the frozen primary grouping analysis. Core M0--M4 artifacts
are inputs for comparison and are never modified by this command.
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
    from scripts.build_annual_groupings import (
        _gap_record,
        _gics_labels,
        _group_return_records,
        _schedule_by_year,
        _validate_arithmetic_binding,
    )
    from scripts.ml_clustering_methods import (
        array_sha256,
        canonical_cluster_labels,
        deterministic_kmeans,
        labels_sha256,
        pca_correlation_profile_embedding,
        spectral_embedding,
    )
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        sha256_file,
        write_json_atomic,
        write_parquet_atomic,
    )
    from scripts.research_methods import (
        adjusted_rand_index,
        dependence_gap,
        group_simple_returns,
        normalized_mutual_information,
        pairwise_spearman,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from build_annual_groupings import (
        _gap_record,
        _gics_labels,
        _group_return_records,
        _schedule_by_year,
        _validate_arithmetic_binding,
    )
    from ml_clustering_methods import (
        array_sha256,
        canonical_cluster_labels,
        deterministic_kmeans,
        labels_sha256,
        pca_correlation_profile_embedding,
        spectral_embedding,
    )
    from pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        sha256_file,
        write_json_atomic,
        write_parquet_atomic,
    )
    from research_methods import (
        adjusted_rand_index,
        dependence_gap,
        group_simple_returns,
        normalized_mutual_information,
        pairwise_spearman,
    )

GROUPING_IDS = ("spectral_cluster", "pca_kmeans_cluster")
METHOD_CODES = {"spectral_cluster": 1, "pca_kmeans_cluster": 2}


def _require_values(section: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    mismatches = {
        key: {"expected": value, "observed": section.get(key)}
        for key, value in expected.items()
        if section.get(key) != value
    }
    if mismatches:
        raise ValueError(f"unsupported {name} protocol: {mismatches}")


def validate_ml_protocol(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed if a production ML choice differs from the frozen protocol."""

    _require_values(
        config,
        {
            "schema_version": 1,
            "protocol_status": "frozen_before_ml_grouping_results",
            "analysis_role": "exploratory_unsupervised_robustness",
            "reporting_scope_rule": "inherit_foundation_gate",
        },
        "top-level ML",
    )
    grouping = config.get("grouping")
    kmeans = config.get("kmeans")
    spectral = config.get("spectral")
    pca = config.get("pca_kmeans")
    downstream = config.get("downstream_risk_models")
    inference = config.get("exploratory_inference")
    if not all(
        isinstance(section, Mapping)
        for section in (grouping, kmeans, spectral, pca, downstream, inference)
    ):
        raise ValueError("ML configuration is missing a required table")
    _require_values(
        grouping,
        {
            "methods": list(GROUPING_IDS),
            "evaluation_start_year": 2020,
            "evaluation_end_year": 2025,
            "training_window_calendar_years": 3,
            "minimum_training_observations": 700,
            "correlation": "spearman_average_ranks",
            "distance": "sqrt((1-rho)/2)",
            "minimum_paired_fraction": 0.8,
            "group_count_rule": "annual_nonempty_gics_sector_count",
            "universe_variant": "security_primary",
            "portfolio_return_scale": "simple",
            "portfolio_rebalancing": "daily",
            "group_weighting": "group_size",
            "cluster_label_order": "lexicographic_sorted_member_tuples",
            "portfolio_identity_tolerance": 1e-12,
        },
        "ML grouping",
    )
    _require_values(
        kmeans,
        {
            "algorithm": "lloyd_kmeans_plus_plus",
            "n_init": 100,
            "maximum_iterations": 500,
            "convergence_tolerance": 1e-10,
            "bit_generator": "PCG64DXSM",
            "base_seed": 5110,
            "seed_components": ["base_seed", "year", "method_code"],
            "empty_cluster_rule": ("move_farthest_point_from_cluster_with_more_than_one_member"),
            "best_run_rule": "minimum_inertia_then_lexicographic_partition",
        },
        "ML k-means",
    )
    _require_values(
        spectral,
        {
            "method_code": 1,
            "affinity": "rbf_of_spearman_correlation_distance",
            "bandwidth": "median_positive_off_diagonal_training_distance",
            "affinity_diagonal": 0.0,
            "laplacian": "symmetric_normalized",
            "embedding": "eigenvectors_of_smallest_laplacian_eigenvalues",
            "row_normalize_embedding": True,
        },
        "spectral clustering",
    )
    _require_values(
        pca,
        {
            "method_code": 2,
            "input_features": "annual_training_spearman_correlation_profiles",
            "self_correlation_treatment": ("replace_diagonal_with_column_off_diagonal_mean"),
            "feature_preprocessing": "column_center_only",
            "decomposition": "full_svd",
            "component_rule": ("minimum_components_reaching_cumulative_explained_variance"),
            "explained_variance_threshold": 0.8,
            "minimum_components": 2,
            "embedding": "left_singular_vectors_times_singular_values",
        },
        "PCA plus k-means",
    )
    _require_values(
        downstream,
        {
            "status": "reserved_not_yet_implemented",
            "inherit_marginal_copula_simulation_protocol_from": "config/model_config.toml",
        },
        "downstream ML risk-model",
    )
    model_rows = downstream.get("models")
    expected_models = [
        {"model_id": "M5", "grouping_id": "spectral", "dependence": "gaussian"},
        {"model_id": "M6", "grouping_id": "spectral", "dependence": "vine_tree_3"},
        {"model_id": "M7", "grouping_id": "pca_kmeans", "dependence": "gaussian"},
        {"model_id": "M8", "grouping_id": "pca_kmeans", "dependence": "vine_tree_3"},
    ]
    if model_rows != expected_models:
        raise ValueError("unsupported downstream ML model registry")
    _require_values(
        inference,
        {
            "loss_scores": ["quantile_loss_95", "quantile_loss_99", "fz0_975"],
            "matched_baselines": ["gics", "hierarchical"],
            "dm_hac_lag": 7,
            "multiplicity": "benjamini_hochberg",
            "family_size": 24,
            "false_discovery_rate": 0.05,
            "calibration_role": "descriptive",
        },
        "exploratory ML inference",
    )
    return config


def _hierarchical_labels_by_year(payload: Mapping[str, Any]) -> dict[int, dict[str, str]]:
    if payload.get("status") != "pass":
        raise RuntimeError("primary annual grouping assignments are not passed")
    result: dict[int, dict[str, str]] = {}
    for year_row in payload.get("years", []):
        year = int(year_row["year"])
        labels = {
            str(row["ticker"]): str(row["hierarchical_cluster"]) for row in year_row["assignments"]
        }
        if len(labels) != int(year_row["active_security_count"]):
            raise ValueError(f"invalid primary assignments for {year}")
        result[year] = labels
    return result


def _validate_primary_grouping_binding(
    audit: Mapping[str, Any], assignments_path: Path, returns_path: Path
) -> None:
    if audit.get("status") != "pass":
        raise RuntimeError("primary annual grouping audit is not passed")
    if audit.get("inputs", {}).get("simple_return_panel", {}).get("sha256") != sha256_file(
        returns_path
    ):
        raise RuntimeError("primary grouping audit is bound to a different return panel")
    if audit.get("outputs", {}).get("annual_group_assignments", {}).get("sha256") != sha256_file(
        assignments_path
    ):
        raise RuntimeError("primary grouping assignments differ from their passed audit")


def _cluster_sizes(labels: Mapping[str, str]) -> dict[str, int]:
    counts = pd.Series(labels, dtype="object").value_counts().sort_index()
    return {str(group): int(count) for group, count in counts.items()}


def _portfolio_identity_error(annual: pd.DataFrame, labels: Mapping[str, str]) -> float:
    grouped = group_simple_returns(annual, labels)
    weights = grouped.group_sizes / grouped.group_sizes.sum()
    reconstructed = grouped.simple_returns.mul(weights, axis="columns").sum(axis=1)
    return float((annual.mean(axis=1) - reconstructed).abs().max())


def build_ml_groupings(
    stock_simple_returns: pd.DataFrame,
    active_schedule: Mapping[str, Any],
    gics_labels: Mapping[str, str],
    hierarchical_assignments: Mapping[int, Mapping[str, str]],
    config: Mapping[str, Any],
    *,
    reporting_scope_value: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Return ML assignments, group returns, diagnostics, and seed records."""

    panel = stock_simple_returns.copy()
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index), name="date")
    if panel.empty or panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("return panel must be non-empty with unique sorted dates")
    if panel.columns.has_duplicates:
        raise ValueError("return panel security columns must be unique")

    grouping = config["grouping"]
    kmeans = config["kmeans"]
    spectral_config = config["spectral"]
    pca_config = config["pca_kmeans"]
    start_year = int(grouping["evaluation_start_year"])
    end_year = int(grouping["evaluation_end_year"])
    training_years = int(grouping["training_window_calendar_years"])
    minimum_observations = int(grouping["minimum_training_observations"])
    paired_fraction = float(grouping["minimum_paired_fraction"])
    identity_tolerance = float(grouping["portfolio_identity_tolerance"])
    schedule = _schedule_by_year(active_schedule)

    assignment_years: list[dict[str, Any]] = []
    diagnostic_years: list[dict[str, Any]] = []
    group_return_frames: list[pd.DataFrame] = []
    seed_records: list[dict[str, Any]] = []
    previous: dict[str, dict[str, str]] = {}
    previous_year: int | None = None

    for year in range(start_year, end_year + 1):
        if year not in schedule or year not in hierarchical_assignments:
            raise ValueError(f"required active or hierarchical assignment year is missing: {year}")
        schedule_row = schedule[year]
        active = sorted(str(value) for value in schedule_row["active_tickers"])
        if len(active) != len(set(active)) or len(active) != int(schedule_row["security_count"]):
            raise ValueError(f"invalid active set for {year}")
        missing = sorted(set(active) - set(panel.columns))
        missing_gics = sorted(set(active) - set(gics_labels))
        if missing or missing_gics:
            raise ValueError(
                f"active set is not represented for {year}; panel={missing}, gics={missing_gics}"
            )
        annual_gics = {ticker: str(gics_labels[ticker]) for ticker in active}
        hierarchical = {ticker: str(hierarchical_assignments[year][ticker]) for ticker in active}
        group_count = len(set(annual_gics.values()))
        if not 1 < group_count < len(active):
            raise ValueError(f"invalid group count for {year}: {group_count}")

        rebalance_date = pd.Timestamp(str(schedule_row["rebalance_date"]))
        training_start = rebalance_date - pd.DateOffset(years=training_years)
        training = panel.loc[
            (panel.index >= training_start) & (panel.index < rebalance_date), active
        ]
        if len(training) < minimum_observations:
            raise ValueError(
                f"training window for {year} has {len(training)} rows; "
                f"minimum is {minimum_observations}"
            )
        training_finite = np.isfinite(training.to_numpy(dtype=float)).all(axis=1)
        training_complete = training.loc[training_finite]
        if len(training_complete) < minimum_observations:
            raise ValueError(
                f"complete training window for {year} has {len(training_complete)} rows; "
                f"minimum is {minimum_observations}"
            )
        correlation = pairwise_spearman(training, minimum_paired_fraction=paired_fraction)

        spectral = spectral_embedding(correlation.to_numpy(dtype=float), group_count)
        pca = pca_correlation_profile_embedding(
            correlation.to_numpy(dtype=float),
            explained_variance_threshold=float(pca_config["explained_variance_threshold"]),
            minimum_components=int(pca_config["minimum_components"]),
        )
        embeddings = {
            "spectral_cluster": spectral.values,
            "pca_kmeans_cluster": pca.values,
        }
        labels_by_method: dict[str, dict[str, str]] = {}
        method_details: dict[str, dict[str, Any]] = {}
        for grouping_id, values in embeddings.items():
            method_code = METHOD_CODES[grouping_id]
            seed_components = [int(kmeans["base_seed"]), year, method_code]
            result = deterministic_kmeans(
                values,
                group_count,
                seed_components=seed_components,
                n_init=int(kmeans["n_init"]),
                maximum_iterations=int(kmeans["maximum_iterations"]),
                convergence_tolerance=float(kmeans["convergence_tolerance"]),
            )
            prefix = "spectral" if grouping_id == "spectral_cluster" else "pca_kmeans"
            labels = canonical_cluster_labels(active, result.labels, prefix=prefix)
            if len(set(labels.values())) != group_count:
                raise AssertionError(f"{grouping_id} returned the wrong group count for {year}")
            labels_by_method[grouping_id] = labels
            detail: dict[str, Any] = {
                "embedding_sha256": array_sha256(values),
                "embedding_dimension": int(values.shape[1]),
                "selected_initialization_zero_based": result.selected_initialization,
                "iterations": result.iterations,
                "inertia": result.inertia,
            }
            if grouping_id == "spectral_cluster":
                detail.update(
                    {
                        "affinity_bandwidth": spectral.bandwidth,
                        "laplacian_eigenvalues": spectral.laplacian_eigenvalues.tolist(),
                    }
                )
            else:
                detail.update(
                    {
                        "pca_component_count": pca.component_count,
                        "pca_cumulative_explained_variance": (pca.cumulative_explained_variance),
                        "pca_explained_variance_ratios": (pca.explained_variance_ratios.tolist()),
                    }
                )
            method_details[grouping_id] = detail
            seed_records.append(
                {
                    "year": year,
                    "grouping_id": grouping_id,
                    "method_code": method_code,
                    "bit_generator": str(kmeans["bit_generator"]),
                    "seed_components": seed_components,
                    "n_init": int(kmeans["n_init"]),
                    "selected_initialization_zero_based": result.selected_initialization,
                    "iterations": result.iterations,
                    "inertia": result.inertia,
                    "embedding_sha256": array_sha256(values),
                    "labels_sha256": labels_sha256(labels),
                }
            )

        annual = panel.loc[(panel.index >= rebalance_date) & (panel.index.year == year), active]
        if annual.empty or not np.isfinite(annual.to_numpy(dtype=float)).all():
            raise ValueError(f"evaluation window for {year} is empty or non-finite")
        evaluation_correlation = pairwise_spearman(annual, minimum_paired_fraction=paired_fraction)

        year_diagnostics: dict[str, Any] = {
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
            "methods": {},
        }
        assignment_rows: list[dict[str, str]] = []
        for ticker in active:
            assignment_rows.append(
                {
                    "ticker": ticker,
                    "gics_sector": annual_gics[ticker],
                    "hierarchical_cluster": hierarchical[ticker],
                    "spectral_cluster": labels_by_method["spectral_cluster"][ticker],
                    "pca_kmeans_cluster": labels_by_method["pca_kmeans_cluster"][ticker],
                }
            )
        for grouping_id, labels in labels_by_method.items():
            identity_error = _portfolio_identity_error(annual, labels)
            if identity_error > identity_tolerance:
                raise AssertionError(
                    f"{grouping_id} portfolio identity failed in {year}: {identity_error}"
                )
            common_previous = sorted(set(previous.get(grouping_id, {})) & set(labels))
            ari = None
            if previous_year is not None:
                ari = adjusted_rand_index(
                    {ticker: previous[grouping_id][ticker] for ticker in common_previous},
                    {ticker: labels[ticker] for ticker in common_previous},
                )
            gap = dependence_gap(evaluation_correlation, labels)
            gics_gap = dependence_gap(evaluation_correlation, annual_gics)
            hierarchical_gap = dependence_gap(evaluation_correlation, hierarchical)
            year_diagnostics["methods"][grouping_id] = {
                **method_details[grouping_id],
                "cluster_sizes": _cluster_sizes(labels),
                "dependence_gap": _gap_record(gap),
                "gap_minus_gics_pair_weighted": gap.gap - gics_gap.gap,
                "gap_minus_hierarchical_pair_weighted": gap.gap - hierarchical_gap.gap,
                "nmi_vs_gics": normalized_mutual_information(labels, annual_gics),
                "nmi_vs_hierarchical": normalized_mutual_information(labels, hierarchical),
                "ari_vs_previous_year": ari,
                "ari_previous_year": previous_year,
                "ari_active_intersection_count": (
                    len(common_previous) if previous_year is not None else None
                ),
                "maximum_portfolio_identity_error": identity_error,
            }
            group_return_frames.extend(
                [
                    _group_return_records(
                        training_complete,
                        labels,
                        year=year,
                        grouping_id=grouping_id,
                        universe_variant="security_primary",
                        sample_role="training",
                    ),
                    _group_return_records(
                        annual,
                        labels,
                        year=year,
                        grouping_id=grouping_id,
                        universe_variant="security_primary",
                        sample_role="evaluation",
                    ),
                ]
            )

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
                "assignments": assignment_rows,
            }
        )
        diagnostic_years.append(year_diagnostics)
        previous = labels_by_method
        previous_year = year

    group_returns = pd.concat(group_return_frames, ignore_index=True)
    assignments_payload = {
        "schema_version": 1,
        "artifact": "ml_annual_group_assignments",
        "status": "pass",
        "analysis_role": "exploratory_unsupervised_robustness",
        "reporting_scope": reporting_scope_value,
        "evaluation_years": [start_year, end_year],
        "grouping_ids": list(GROUPING_IDS),
        "universe_variants": ["security_primary"],
        "years": assignment_years,
    }
    method_summaries: dict[str, dict[str, float]] = {}
    for grouping_id in GROUPING_IDS:
        rows = [row["methods"][grouping_id] for row in diagnostic_years]
        method_summaries[grouping_id] = {
            "mean_gap_minus_gics_pair_weighted": float(
                np.mean([row["gap_minus_gics_pair_weighted"] for row in rows])
            ),
            "mean_gap_minus_hierarchical_pair_weighted": float(
                np.mean([row["gap_minus_hierarchical_pair_weighted"] for row in rows])
            ),
            "mean_nmi_vs_gics": float(np.mean([row["nmi_vs_gics"] for row in rows])),
            "mean_nmi_vs_hierarchical": float(
                np.mean([row["nmi_vs_hierarchical"] for row in rows])
            ),
            "maximum_portfolio_identity_error": float(
                max(row["maximum_portfolio_identity_error"] for row in rows)
            ),
        }
    diagnostics = {
        "schema_version": 1,
        "gate_name": "ml_annual_grouping_v1",
        "status": "pass",
        "analysis_role": "exploratory_unsupervised_robustness",
        "reporting_scope": reporting_scope_value,
        "method": {
            "grouping_ids": list(GROUPING_IDS),
            "training_window_calendar_years": training_years,
            "minimum_training_observations": minimum_observations,
            "minimum_paired_fraction": paired_fraction,
            "group_count_rule": "annual_nonempty_gics_sector_count",
            "portfolio_return_scale": "simple",
            "portfolio_rebalancing": "daily",
            "group_weighting": "group_size",
            "bit_generator": str(kmeans["bit_generator"]),
            "seed_sequence": "SeedSequence([5110, year, method_code])",
            "spectral_affinity": str(spectral_config["affinity"]),
            "pca_component_rule": str(pca_config["component_rule"]),
        },
        "years": diagnostic_years,
        "summary": {
            "evaluation_year_count": len(diagnostic_years),
            "method_summaries": method_summaries,
            "downstream_risk_models_status": "reserved_not_yet_implemented",
        },
        "issues": [],
    }
    seed_manifest = {
        "schema_version": 1,
        "manifest_type": "ml_clustering_seed_manifest",
        "status": "pass",
        "analysis_role": "exploratory_unsupervised_robustness",
        "reporting_scope": reporting_scope_value,
        "bit_generator": str(kmeans["bit_generator"]),
        "seed_sequence": "SeedSequence([5110, year, method_code])",
        "records": seed_records,
    }
    return assignments_payload, group_returns, diagnostics, seed_manifest


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
        "--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_extension_config.toml"
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
        "--primary-grouping-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/clustering_diagnostics.json",
    )
    parser.add_argument(
        "--primary-assignments",
        type=Path,
        default=PROJECT_ROOT / "data/processed/annual_group_assignments.json",
    )
    parser.add_argument(
        "--assignments-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_annual_group_assignments.json",
    )
    parser.add_argument(
        "--group-returns-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_annual_group_returns.parquet",
    )
    parser.add_argument(
        "--seed-manifest-output",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/ml_clustering_seed_manifest.json",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_clustering_diagnostics.json",
    )
    return parser.parse_args()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": project_path(path), "sha256": sha256_file(path)}


def main() -> int:
    args = parse_args()
    with args.ml_config.open("rb") as handle:
        config = validate_ml_protocol(tomllib.load(handle))
    gate = json.loads(args.foundation_status.read_text(encoding="utf-8"))
    scope = reporting_scope(gate)
    arithmetic_audit = json.loads(args.portfolio_arithmetic_audit.read_text(encoding="utf-8"))
    _validate_arithmetic_binding(
        arithmetic_audit, args.returns, args.active_universe, args.universe
    )
    primary_audit = json.loads(args.primary_grouping_audit.read_text(encoding="utf-8"))
    _validate_primary_grouping_binding(primary_audit, args.primary_assignments, args.returns)
    primary_assignments = json.loads(args.primary_assignments.read_text(encoding="utf-8"))
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    schedule = json.loads(args.active_universe.read_text(encoding="utf-8"))
    panel = pd.read_parquet(args.returns)

    assignments, group_returns, audit, seed_manifest = build_ml_groupings(
        panel,
        schedule,
        _gics_labels(universe),
        _hierarchical_labels_by_year(primary_assignments),
        config,
        reporting_scope_value=scope,
    )
    write_json_atomic(args.assignments_output, assignments)
    write_parquet_atomic(args.group_returns_output, group_returns)
    write_json_atomic(args.seed_manifest_output, seed_manifest)
    audit["inputs"] = {
        "simple_return_panel": _artifact_record(args.returns),
        "active_universe": _artifact_record(args.active_universe),
        "final_universe": _artifact_record(args.universe),
        "ml_config": _artifact_record(args.ml_config),
        "foundation_status": _artifact_record(args.foundation_status),
        "portfolio_arithmetic_audit": _artifact_record(args.portfolio_arithmetic_audit),
        "primary_grouping_audit": _artifact_record(args.primary_grouping_audit),
        "primary_assignments": _artifact_record(args.primary_assignments),
    }
    audit["outputs"] = {
        "ml_annual_group_assignments": _artifact_record(args.assignments_output),
        "ml_annual_group_returns": {
            **_artifact_record(args.group_returns_output),
            "rows": len(group_returns),
        },
        "ml_clustering_seed_manifest": _artifact_record(args.seed_manifest_output),
    }
    write_json_atomic(args.audit_output, audit)
    print(
        f"ml_annual_grouping={audit['status']} scope={scope} "
        f"years={audit['summary']['evaluation_year_count']} "
        f"group_return_rows={len(group_returns)}"
    )
    print(f"Audit: {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
