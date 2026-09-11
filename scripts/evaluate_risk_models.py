#!/usr/bin/env python3
"""Evaluate M0-M4 forecasts and run the frozen H1-H3 inference protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        sha256_file,
        write_json_atomic,
        write_parquet_atomic,
    )
    from scripts.research_methods import (
        christoffersen_conditional_coverage_p_value,
        christoffersen_independence,
        diebold_mariano,
        fz0_loss,
        holm_adjust,
        kupiec_unconditional_coverage,
        quantile_loss,
        var_exceptions,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        sha256_file,
        write_json_atomic,
        write_parquet_atomic,
    )
    from research_methods import (
        christoffersen_conditional_coverage_p_value,
        christoffersen_independence,
        diebold_mariano,
        fz0_loss,
        holm_adjust,
        kupiec_unconditional_coverage,
        quantile_loss,
        var_exceptions,
    )


MODEL_IDS = ("M0", "M1", "M2", "M3", "M4")
MODEL_GROUPINGS = {
    "M0": "none",
    "M1": "gics",
    "M2": "gics",
    "M3": "hierarchical",
    "M4": "hierarchical",
}
FORECAST_COLUMNS = (
    "date",
    "model_id",
    "grouping_id",
    "refit_id",
    "var_95",
    "var_975",
    "var_99",
    "es_975",
    "realised_simple_return",
    "realised_loss",
    "seed_components",
    "margin_fallback_count",
    "whole_vine_fallback",
    "copula_log_score",
    "forecast_status",
)
DAILY_SCORE_COLUMNS = (
    "date",
    "year",
    "model_id",
    "grouping_id",
    "realised_loss",
    "var_95",
    "var_975",
    "var_99",
    "es_975",
    "quantile_loss_95",
    "quantile_loss_99",
    "fz0_975",
    "exception_95",
    "exception_975",
    "exception_99",
    "copula_log_score",
    "forecast_status",
    "margin_fallback_count",
    "whole_vine_fallback",
)
PRIMARY_SCORES = ("quantile_loss_95", "quantile_loss_99", "fz0_975")
VAR_COLUMNS = {0.95: "var_95", 0.975: "var_975", 0.99: "var_99"}
EXCEPTION_COLUMNS = {0.95: "exception_95", 0.975: "exception_975", 0.99: "exception_99"}


@dataclass(frozen=True)
class H1Year:
    year: int
    dates: pd.DatetimeIndex
    values: np.ndarray
    upper_rows: np.ndarray
    upper_columns: np.ndarray
    gics_within: np.ndarray
    cluster_within: np.ndarray


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a table")
    return cast(Mapping[str, Any], value)


def validate_evaluation_protocol(
    evaluation_config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    """Require the exact pre-result evaluation and inference choices."""

    if evaluation_config.get("schema_version") != 1:
        raise ValueError("unsupported evaluation configuration schema")
    if evaluation_config.get("protocol_status") != "frozen_before_aggregate_result_evaluation":
        raise ValueError("evaluation protocol was not frozen before aggregate evaluation")
    evaluation = _mapping(evaluation_config.get("evaluation"), "evaluation")
    calibration = _mapping(evaluation_config.get("calibration"), "calibration")
    h1 = _mapping(evaluation_config.get("h1"), "h1")
    h2 = _mapping(evaluation_config.get("h2"), "h2")
    h3 = _mapping(evaluation_config.get("h3"), "h3")
    inference = _mapping(model_config.get("inference"), "model inference")
    forecast = _mapping(model_config.get("forecast"), "model forecast")
    clustering = _mapping(model_config.get("clustering"), "model clustering")

    expected = {
        "evaluation.model_ids": (evaluation.get("model_ids"), list(MODEL_IDS)),
        "evaluation.primary_score_columns": (
            evaluation.get("primary_score_columns"),
            list(PRIMARY_SCORES),
        ),
        "evaluation.model_ranking": (
            evaluation.get("model_ranking"),
            "equal_weight_average_rank_across_primary_scores",
        ),
        "calibration.tests": (
            calibration.get("tests"),
            ["kupiec_unconditional_coverage", "christoffersen_independence"],
        ),
        "calibration.multiplicity": (calibration.get("multiplicity"), "holm_familywise"),
        "h1.annual_aggregation": (
            h1.get("annual_aggregation"),
            "equal_weight_mean_across_evaluation_years",
        ),
        "h1.bootstrap": (h1.get("bootstrap"), "paired_circular_moving_block_within_year"),
        "h1.bit_generator": (h1.get("bit_generator"), "PCG64DXSM"),
        "h1.seed_components": (h1.get("seed_components"), [5110, 1]),
        "h1.confidence_interval": (
            h1.get("confidence_interval"),
            "two_sided_percentile_linear",
        ),
        "h1.degenerate_sample_action": (
            h1.get("degenerate_sample_action"),
            "discard_whole_paired_replication_and_continue_rng_stream",
        ),
        "h2.loss_difference": (h2.get("loss_difference"), "vine_minus_gaussian"),
        "h2.score_columns": (h2.get("score_columns"), list(PRIMARY_SCORES)),
        "h2.hac": (h2.get("hac"), "newey_west_bartlett"),
        "h2.multiplicity": (h2.get("multiplicity"), "holm_familywise"),
        "h3.candidate_model": (h3.get("candidate_model"), "M4"),
        "h3.ranking": (
            h3.get("ranking"),
            "unique_lowest_equal_weight_average_primary_score_rank",
        ),
    }
    mismatches = {
        name: {"observed": observed, "expected": required}
        for name, (observed, required) in expected.items()
        if observed != required
    }
    if mismatches:
        raise ValueError(f"evaluation protocol differs from the implemented freeze: {mismatches}")

    start_year = int(evaluation.get("evaluation_start_year", -1))
    end_year = int(evaluation.get("evaluation_end_year", -1))
    if (start_year, end_year) != (
        clustering.get("evaluation_start_year"),
        clustering.get("evaluation_end_year"),
    ) or (start_year, end_year) != (2020, 2025):
        raise ValueError("evaluation years do not match the frozen model protocol")
    if [float(value) for value in forecast.get("var_confidence_levels", [])] != [
        0.95,
        0.975,
        0.99,
    ]:
        raise ValueError("forecast confidence levels differ from the evaluation protocol")
    if [float(value) for value in calibration.get("confirmatory_confidence_levels", [])] != [
        0.95,
        0.99,
    ]:
        raise ValueError("confirmatory calibration levels must be 95% and 99%")
    if [float(value) for value in calibration.get("descriptive_annual_confidence_levels", [])] != [
        0.95,
        0.975,
        0.99,
    ]:
        raise ValueError("annual calibration levels differ from the frozen protocol")

    numeric_pairs = {
        "calibration.family_size": (calibration.get("family_size"), 20),
        "calibration.familywise_alpha": (calibration.get("familywise_alpha"), 0.05),
        "h1.block_length_trading_days": (
            h1.get("block_length_trading_days"),
            inference.get("h1_block_length_trading_days"),
        ),
        "h1.replications": (h1.get("replications"), inference.get("h1_bootstrap_replications")),
        "h1.confidence_level": (
            h1.get("confidence_level"),
            inference.get("h1_confidence_level"),
        ),
        "h2.hac_lag": (h2.get("hac_lag"), inference.get("dm_hac_lag")),
        "h2.family_size": (h2.get("family_size"), inference.get("dm_holm_family_size")),
        "h2.familywise_alpha": (
            h2.get("familywise_alpha"),
            inference.get("confirmatory_familywise_alpha"),
        ),
    }
    invalid_numeric = {
        name: {"observed": observed, "expected": required}
        for name, (observed, required) in numeric_pairs.items()
        if observed != required
    }
    if invalid_numeric:
        raise ValueError(f"evaluation and model inference settings disagree: {invalid_numeric}")
    frozen_numeric = {
        "calibration.family_size": (calibration.get("family_size"), 20),
        "calibration.familywise_alpha": (calibration.get("familywise_alpha"), 0.05),
        "h1.block_length_trading_days": (h1.get("block_length_trading_days"), 20),
        "h1.replications": (h1.get("replications"), 10_000),
        "h1.confidence_level": (h1.get("confidence_level"), 0.95),
        "h2.hac_lag": (h2.get("hac_lag"), 7),
        "h2.family_size": (h2.get("family_size"), 6),
        "h2.familywise_alpha": (h2.get("familywise_alpha"), 0.05),
    }
    unfrozen = {
        name: {"observed": observed, "expected": required}
        for name, (observed, required) in frozen_numeric.items()
        if observed != required
    }
    if unfrozen:
        raise ValueError(f"evaluation numeric settings differ from the freeze: {unfrozen}")
    if calibration.get("family_size") != len(MODEL_IDS) * 2 * 2:
        raise ValueError(
            "calibration family size must equal five models by two levels by two tests"
        )
    comparisons = h2.get("comparisons")
    expected_comparisons = [
        {"vine_model": "M2", "gaussian_model": "M1", "grouping_id": "gics"},
        {"vine_model": "M4", "gaussian_model": "M3", "grouping_id": "hierarchical"},
    ]
    if comparisons != expected_comparisons or h2.get("family_size") != 6:
        raise ValueError("H2 must contain the six frozen matched comparisons")
    tolerance = evaluation.get("realised_loss_identity_tolerance")
    if not isinstance(tolerance, int | float) or isinstance(tolerance, bool) or tolerance <= 0:
        raise ValueError("evaluation realised-loss tolerance must be positive")
    return {
        "evaluation": evaluation,
        "calibration": calibration,
        "h1": h1,
        "h2": h2,
        "h3": h3,
    }


def combine_forecasts(
    historical: pd.DataFrame,
    gaussian: pd.DataFrame,
    vine: pd.DataFrame,
    evaluation: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate schemas, identities, and matched dates before combining M0-M4."""

    sources = ((historical, {"M0"}), (gaussian, {"M1", "M3"}), (vine, {"M2", "M4"}))
    normalized: list[pd.DataFrame] = []
    for number, (source, expected_models) in enumerate(sources, start=1):
        if set(source.columns) != set(FORECAST_COLUMNS):
            missing = sorted(set(FORECAST_COLUMNS) - set(source.columns))
            extra = sorted(set(source.columns) - set(FORECAST_COLUMNS))
            raise ValueError(
                f"forecast source {number} schema mismatch; missing={missing}, extra={extra}"
            )
        frame = source.loc[:, FORECAST_COLUMNS].copy()
        try:
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"forecast source {number} contains invalid dates") from exc
        if frame["date"].isna().any():
            raise ValueError(f"forecast source {number} contains missing dates")
        models = set(frame["model_id"].astype(str))
        if models != expected_models:
            raise ValueError(
                f"forecast source {number} models differ; observed={sorted(models)}, "
                f"expected={sorted(expected_models)}"
            )
        if frame.duplicated(["model_id", "date"]).any():
            raise ValueError(f"forecast source {number} contains duplicate model-date rows")
        normalized.append(frame)

    # M0 has no copula score. Build the nullable numeric column explicitly so
    # concatenation does not depend on pandas' evolving all-null dtype inference.
    copula_log_scores = np.concatenate(
        [frame["copula_log_score"].to_numpy() for frame in normalized]
    )
    combined = pd.concat(
        [frame.drop(columns="copula_log_score") for frame in normalized],
        ignore_index=True,
    )
    combined["copula_log_score"] = pd.to_numeric(copula_log_scores, errors="coerce")
    start_year = int(evaluation["evaluation_start_year"])
    end_year = int(evaluation["evaluation_end_year"])
    if set(combined["date"].dt.year) != set(range(start_year, end_year + 1)):
        raise ValueError("forecast dates do not cover every frozen evaluation year")
    if not combined["forecast_status"].isin(evaluation["required_forecast_statuses"]).all():
        raise ValueError("forecast evaluation cannot include failed or unsupported statuses")

    for model_id, expected_grouping in MODEL_GROUPINGS.items():
        rows = combined.loc[combined["model_id"] == model_id]
        if set(rows["grouping_id"].astype(str)) != {expected_grouping}:
            raise ValueError(f"{model_id} does not use grouping {expected_grouping}")
    numeric_columns = [
        "var_95",
        "var_975",
        "var_99",
        "es_975",
        "realised_simple_return",
        "realised_loss",
    ]
    numeric = combined[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("forecast risk and realised values must be finite")
    if bool((combined["var_95"] > combined["var_975"]).any()) or bool(
        (combined["var_975"] > combined["var_99"]).any()
    ):
        raise ValueError("forecast VaR levels must be nondecreasing")
    if bool((combined["es_975"] <= 0).any()) or bool(
        (combined["es_975"] < combined["var_975"]).any()
    ):
        raise ValueError("97.5% ES must be positive and at least its matched VaR")
    tolerance = float(evaluation["realised_loss_identity_tolerance"])
    sign_error = np.abs(combined["realised_loss"] + combined["realised_simple_return"])
    if float(sign_error.max()) > tolerance:
        raise ValueError("realised loss does not equal negative realised simple return")
    non_m0 = combined["model_id"] != "M0"
    if not np.isfinite(combined.loc[non_m0, "copula_log_score"].to_numpy(dtype=float)).all():
        raise ValueError("copula models require finite copula log scores")
    if combined.loc[~non_m0, "copula_log_score"].notna().any():
        raise ValueError("M0 must not contain a copula log score")

    dates_by_model = {
        model: pd.DatetimeIndex(combined.loc[combined["model_id"] == model, "date"])
        for model in MODEL_IDS
    }
    reference_dates = dates_by_model["M0"].sort_values()
    for model, dates in dates_by_model.items():
        if not dates.sort_values().equals(reference_dates):
            raise ValueError(f"{model} forecast dates do not match M0")
    realised = combined.pivot(index="date", columns="model_id", values="realised_loss")
    maximum_identity_error = float(realised.sub(realised["M0"], axis="index").abs().max().max())
    if maximum_identity_error > tolerance:
        raise ValueError("models do not share the same realised portfolio loss")

    order = {model: index for index, model in enumerate(MODEL_IDS)}
    combined["_model_order"] = combined["model_id"].map(order)
    combined = combined.sort_values(["date", "_model_order"]).drop(columns="_model_order")
    combined = combined.reset_index(drop=True)
    validation = {
        "evaluation_date_count": len(reference_dates),
        "forecast_count": len(combined),
        "model_date_counts": {
            model: int((combined["model_id"] == model).sum()) for model in MODEL_IDS
        },
        "evaluation_start": reference_dates.min().date().isoformat(),
        "evaluation_end": reference_dates.max().date().isoformat(),
        "maximum_realised_loss_identity_error": maximum_identity_error,
        "realised_loss_identity_tolerance": tolerance,
    }
    return combined, validation


def score_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily frozen losses and VaR exception indicators."""

    loss = forecasts["realised_loss"].to_numpy(dtype=float)
    scored = forecasts[
        [
            "date",
            "model_id",
            "grouping_id",
            "realised_loss",
            "var_95",
            "var_975",
            "var_99",
            "es_975",
            "copula_log_score",
            "forecast_status",
            "margin_fallback_count",
            "whole_vine_fallback",
        ]
    ].copy()
    scored.insert(1, "year", scored["date"].dt.year.astype("int64"))
    scored["quantile_loss_95"] = quantile_loss(loss, scored["var_95"], 0.95)
    scored["quantile_loss_99"] = quantile_loss(loss, scored["var_99"], 0.99)
    scored["fz0_975"] = fz0_loss(loss, scored["var_975"], scored["es_975"], 0.975)
    for confidence, var_column in VAR_COLUMNS.items():
        scored[EXCEPTION_COLUMNS[confidence]] = var_exceptions(loss, scored[var_column])
    return scored.loc[:, DAILY_SCORE_COLUMNS]


def calibration_results(
    scores: pd.DataFrame, calibration: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return full-period confirmatory and annual descriptive VaR diagnostics."""

    alpha = float(calibration["familywise_alpha"])
    full: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        model = scores.loc[scores["model_id"] == model_id].sort_values("date")
        for confidence in calibration["confirmatory_confidence_levels"]:
            confidence = float(confidence)
            indicators = model[EXCEPTION_COLUMNS[confidence]].to_numpy(dtype=bool)
            coverage = kupiec_unconditional_coverage(indicators, confidence)
            independence = christoffersen_independence(indicators)
            full.append(
                {
                    "model_id": model_id,
                    "confidence": confidence,
                    "observations": coverage.observations,
                    "exceptions": coverage.exceptions,
                    "exception_rate": coverage.exception_rate,
                    "kupiec_likelihood_ratio": coverage.likelihood_ratio,
                    "kupiec_p_value": coverage.p_value,
                    "christoffersen_independence_likelihood_ratio": (independence.likelihood_ratio),
                    "christoffersen_independence_p_value": independence.p_value,
                    "transitions_00": independence.transitions_00,
                    "transitions_01": independence.transitions_01,
                    "transitions_10": independence.transitions_10,
                    "transitions_11": independence.transitions_11,
                    "conditional_coverage_likelihood_ratio": (
                        coverage.likelihood_ratio + independence.likelihood_ratio
                    ),
                    "conditional_coverage_p_value": (
                        christoffersen_conditional_coverage_p_value(coverage, independence)
                    ),
                }
            )
    raw_p_values = [
        value
        for row in full
        for value in (row["kupiec_p_value"], row["christoffersen_independence_p_value"])
    ]
    if len(raw_p_values) != int(calibration["family_size"]):
        raise AssertionError("calibration test family has the wrong size")
    adjusted = iter(holm_adjust(raw_p_values).tolist())
    for row in full:
        row["kupiec_holm_p_value"] = next(adjusted)
        row["kupiec_reject_after_holm"] = row["kupiec_holm_p_value"] < alpha
        row["christoffersen_independence_holm_p_value"] = next(adjusted)
        row["christoffersen_independence_reject_after_holm"] = (
            row["christoffersen_independence_holm_p_value"] < alpha
        )

    annual: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        model = scores.loc[scores["model_id"] == model_id].sort_values("date")
        for year, annual_model in model.groupby("year", sort=True):
            for confidence in calibration["descriptive_annual_confidence_levels"]:
                confidence = float(confidence)
                indicators = annual_model[EXCEPTION_COLUMNS[confidence]].to_numpy(dtype=bool)
                coverage = kupiec_unconditional_coverage(indicators, confidence)
                independence = christoffersen_independence(indicators)
                annual.append(
                    {
                        "model_id": model_id,
                        "year": int(year),
                        "confidence": confidence,
                        "inference_role": "descriptive",
                        "observations": coverage.observations,
                        "exceptions": coverage.exceptions,
                        "exception_rate": coverage.exception_rate,
                        "kupiec_p_value_unadjusted": coverage.p_value,
                        "christoffersen_independence_p_value_unadjusted": independence.p_value,
                        "conditional_coverage_p_value_unadjusted": (
                            christoffersen_conditional_coverage_p_value(coverage, independence)
                        ),
                    }
                )
    return full, annual


def dm_results(scores: pd.DataFrame, h2: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run the six frozen matched vine-minus-Gaussian DM tests."""

    results: list[dict[str, Any]] = []
    hac_lag = int(h2["hac_lag"])
    for comparison in h2["comparisons"]:
        vine_id = str(comparison["vine_model"])
        gaussian_id = str(comparison["gaussian_model"])
        vine = scores.loc[scores["model_id"] == vine_id].set_index("date")
        gaussian = scores.loc[scores["model_id"] == gaussian_id].set_index("date")
        if not vine.index.equals(gaussian.index):
            raise ValueError(f"DM dates are not matched for {vine_id} and {gaussian_id}")
        for score_column in h2["score_columns"]:
            differential = vine[score_column].to_numpy() - gaussian[score_column].to_numpy()
            result = diebold_mariano(differential, hac_lag=hac_lag)
            results.append(
                {
                    "comparison_id": f"{vine_id}_minus_{gaussian_id}:{score_column}",
                    "vine_model": vine_id,
                    "gaussian_model": gaussian_id,
                    "grouping_id": comparison["grouping_id"],
                    "score": score_column,
                    "observations": result.observations,
                    "mean_loss_difference": result.mean_difference,
                    "dm_statistic": result.statistic,
                    "p_value_two_sided": result.p_value_two_sided,
                    "hac_lag": result.hac_lag,
                }
            )
    if len(results) != int(h2["family_size"]):
        raise AssertionError("DM test family has the wrong size")
    adjusted = holm_adjust([row["p_value_two_sided"] for row in results])
    alpha = float(h2["familywise_alpha"])
    for row, adjusted_p in zip(results, adjusted, strict=True):
        row["holm_p_value"] = float(adjusted_p)
        row["significant_after_holm"] = bool(adjusted_p < alpha)
        difference = float(row["mean_loss_difference"])
        row["direction"] = (
            "vine_lower_loss"
            if difference < 0
            else "gaussian_lower_loss"
            if difference > 0
            else "equal_loss"
        )
        row["favours_vine_after_holm"] = bool(adjusted_p < alpha and difference < 0)
        row["favours_gaussian_after_holm"] = bool(adjusted_p < alpha and difference > 0)
    return results


def model_summaries(
    scores: pd.DataFrame, calibration: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Summarize forecast accuracy and compute the frozen average-rank ordering."""

    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        model = scores.loc[scores["model_id"] == model_id]
        row: dict[str, Any] = {
            "model_id": model_id,
            "grouping_id": MODEL_GROUPINGS[model_id],
            "observations": len(model),
            "mean_quantile_loss_95": float(model["quantile_loss_95"].mean()),
            "mean_quantile_loss_99": float(model["quantile_loss_99"].mean()),
            "mean_fz0_975": float(model["fz0_975"].mean()),
            "mean_copula_log_score": (
                None if model_id == "M0" else float(model["copula_log_score"].mean())
            ),
        }
        for confidence, exception_column in EXCEPTION_COLUMNS.items():
            suffix = str(confidence).replace("0.", "").replace(".", "")
            row[f"exceptions_{suffix}"] = int(model[exception_column].sum())
            row[f"exception_rate_{suffix}"] = float(model[exception_column].mean())
        failures = [
            item
            for item in calibration
            if item["model_id"] == model_id
            and (
                item["kupiec_reject_after_holm"]
                or item["christoffersen_independence_reject_after_holm"]
            )
        ]
        row["systematic_calibration_failure"] = bool(failures)
        row["adjusted_calibration_rejection_count"] = sum(
            int(item["kupiec_reject_after_holm"])
            + int(item["christoffersen_independence_reject_after_holm"])
            for item in calibration
            if item["model_id"] == model_id
        )
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("model_id")
    mean_columns = [f"mean_{score}" for score in PRIMARY_SCORES]
    rank_columns: list[str] = []
    for mean_column in mean_columns:
        rank_column = f"rank_{mean_column.removeprefix('mean_')}"
        frame[rank_column] = frame[mean_column].rank(method="average", ascending=True)
        rank_columns.append(rank_column)
    frame["average_primary_score_rank"] = frame[rank_columns].mean(axis=1)
    frame["overall_rank"] = frame["average_primary_score_rank"].rank(
        method="average", ascending=True
    )
    records = cast(list[dict[str, Any]], frame.reset_index().to_dict(orient="records"))
    for row in records:
        if row["model_id"] == "M0":
            row["mean_copula_log_score"] = None
    return records


def _gap(
    correlation: np.ndarray, rows: np.ndarray, columns: np.ndarray, within: np.ndarray
) -> float:
    pairs = correlation[rows, columns]
    if not np.isfinite(pairs).all() or not bool(within.any()) or bool(within.all()):
        raise ValueError("dependence-gap correlation or grouping is invalid")
    return float(pairs[within].mean() - pairs[~within].mean())


def _spearman_correlation(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, axis=0, method="average")
    correlation = np.corrcoef(ranked, rowvar=False)
    if correlation.ndim != 2 or not np.isfinite(correlation).all():
        raise ValueError("bootstrap sample produced an invalid Spearman correlation")
    return correlation


def prepare_h1_years(
    stock_returns: pd.DataFrame,
    assignments: Mapping[str, Any],
    clustering_diagnostics: Mapping[str, Any],
) -> list[H1Year]:
    """Bind annual stock panels and fixed labels to existing OOS diagnostics."""

    panel = stock_returns.copy()
    if not isinstance(panel.index, pd.DatetimeIndex):
        try:
            panel.index = pd.to_datetime(panel.index, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("stock-return panel has an invalid date index") from exc
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("stock-return dates must be unique and increasing")
    assignment_rows = assignments.get("years")
    diagnostic_rows = clustering_diagnostics.get("years")
    if not isinstance(assignment_rows, list) or not isinstance(diagnostic_rows, list):
        raise ValueError("assignment and clustering artifacts require annual rows")
    diagnostics_by_year = {int(row["year"]): row for row in diagnostic_rows}
    result: list[H1Year] = []
    for annual_assignment in assignment_rows:
        year = int(annual_assignment["year"])
        raw_labels = annual_assignment.get("assignments")
        if not isinstance(raw_labels, list) or not raw_labels:
            raise ValueError(f"year {year} has no security assignments")
        tickers = [str(row["ticker"]) for row in raw_labels]
        if len(tickers) != len(set(tickers)) or not set(tickers).issubset(panel.columns):
            raise ValueError(f"year {year} contains duplicate or unavailable assigned securities")
        annual = panel.loc[panel.index.year == year, tickers]
        values = annual.to_numpy(dtype=float)
        if annual.empty or not np.isfinite(values).all():
            raise ValueError(f"year {year} H1 evaluation panel is empty or incomplete")
        gics = np.asarray([str(row["gics_sector"]) for row in raw_labels])
        clusters = np.asarray([str(row["hierarchical_cluster"]) for row in raw_labels])
        upper_rows, upper_columns = np.triu_indices(len(tickers), k=1)
        gics_within = gics[upper_rows] == gics[upper_columns]
        cluster_within = clusters[upper_rows] == clusters[upper_columns]
        correlation = _spearman_correlation(values)
        observed_gics = _gap(correlation, upper_rows, upper_columns, gics_within)
        observed_cluster = _gap(correlation, upper_rows, upper_columns, cluster_within)
        diagnostic = diagnostics_by_year.get(year)
        if diagnostic is None:
            raise ValueError(f"year {year} is absent from clustering diagnostics")
        expected_gics = float(diagnostic["gics_dependence_gap"]["gap"])
        expected_cluster = float(diagnostic["hierarchical_dependence_gap"]["gap"])
        if not np.isclose(observed_gics, expected_gics, atol=1e-12, rtol=0) or not np.isclose(
            observed_cluster, expected_cluster, atol=1e-12, rtol=0
        ):
            raise ValueError(f"year {year} H1 inputs do not reproduce clustering diagnostics")
        result.append(
            H1Year(
                year,
                pd.DatetimeIndex(annual.index),
                values,
                upper_rows,
                upper_columns,
                gics_within,
                cluster_within,
            )
        )
    years = [annual.year for annual in result]
    if years != list(range(min(years), max(years) + 1)) or years != sorted(diagnostics_by_year):
        raise ValueError("H1 annual inputs must cover identical consecutive years")
    return result


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _circular_block_candidate(
    dates: pd.DatetimeIndex, generator: np.random.Generator, block_length: int
) -> np.ndarray:
    """Draw one paired circular-block index vector without crossing years."""

    candidate = np.empty(len(dates), dtype=np.int32)
    years = np.asarray(dates.year)
    write_at = 0
    for year in dict.fromkeys(years.tolist()):
        positions = np.flatnonzero(years == year)
        count = len(positions)
        sampled: list[int] = []
        while len(sampled) < count:
            start = int(generator.integers(0, count))
            sampled.extend(positions[(start + np.arange(block_length)) % count].tolist())
        candidate[write_at : write_at + count] = sampled[:count]
        write_at += count
    return candidate


def accepted_h1_bootstrap_indices(
    years: Sequence[H1Year],
    *,
    replications: int,
    block_length: int,
    base_seed: int = 5110,
) -> tuple[np.ndarray, int, int]:
    """Generate valid paired indices, rejecting whole degenerate replications."""

    if not years or replications <= 0 or block_length <= 0:
        raise ValueError("H1 bootstrap dimensions must be positive")
    dates = pd.DatetimeIndex(np.concatenate([annual.dates.to_numpy() for annual in years]))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("H1 dates must be unique and increasing")
    generator = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([base_seed, 1])))
    accepted = np.empty((replications, len(dates)), dtype=np.int32)
    offsets = np.cumsum([0, *[len(annual.dates) for annual in years]])
    accepted_count = 0
    candidate_count = 0
    maximum_candidates = replications * 100
    while accepted_count < replications:
        if candidate_count >= maximum_candidates:
            raise RuntimeError("too many degenerate H1 bootstrap candidates")
        candidate = _circular_block_candidate(dates, generator, block_length)
        candidate_count += 1
        degenerate = False
        for year_number, annual in enumerate(years):
            start, end = int(offsets[year_number]), int(offsets[year_number + 1])
            local_indices = candidate[start:end] - start
            sample = annual.values[local_indices]
            if bool((np.ptp(sample, axis=0) == 0).any()):
                degenerate = True
                break
        if degenerate:
            continue
        accepted[accepted_count] = candidate
        accepted_count += 1
    return accepted, candidate_count, candidate_count - replications


def h1_bootstrap(
    years: Sequence[H1Year],
    *,
    replications: int,
    block_length: int,
    confidence_level: float,
    base_seed: int = 5110,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the paired within-year H1 bootstrap and return its seed manifest."""

    if not years:
        raise ValueError("H1 bootstrap requires annual panels")
    dates = pd.DatetimeIndex(np.concatenate([annual.dates.to_numpy() for annual in years]))
    indices, candidate_count, rejected_count = accepted_h1_bootstrap_indices(
        years,
        replications=replications,
        block_length=block_length,
        base_seed=base_seed,
    )
    observed_gics: list[float] = []
    observed_cluster: list[float] = []
    annual_records: list[dict[str, Any]] = []
    for annual in years:
        correlation = _spearman_correlation(annual.values)
        gics_gap = _gap(correlation, annual.upper_rows, annual.upper_columns, annual.gics_within)
        cluster_gap = _gap(
            correlation, annual.upper_rows, annual.upper_columns, annual.cluster_within
        )
        observed_gics.append(gics_gap)
        observed_cluster.append(cluster_gap)
        annual_records.append(
            {
                "year": annual.year,
                "observations": len(annual.dates),
                "gics_dependence_gap": gics_gap,
                "cluster_dependence_gap": cluster_gap,
                "cluster_minus_gics_gap": cluster_gap - gics_gap,
            }
        )

    bootstrap_cluster = np.empty(replications, dtype=np.float64)
    bootstrap_difference = np.empty(replications, dtype=np.float64)
    offsets = np.cumsum([0, *[len(annual.dates) for annual in years]])
    for replication in range(replications):
        cluster_gaps: list[float] = []
        differences: list[float] = []
        for year_number, annual in enumerate(years):
            start, end = int(offsets[year_number]), int(offsets[year_number + 1])
            local_indices = indices[replication, start:end] - start
            correlation = _spearman_correlation(annual.values[local_indices])
            gics_gap = _gap(
                correlation, annual.upper_rows, annual.upper_columns, annual.gics_within
            )
            cluster_gap = _gap(
                correlation, annual.upper_rows, annual.upper_columns, annual.cluster_within
            )
            cluster_gaps.append(cluster_gap)
            differences.append(cluster_gap - gics_gap)
        bootstrap_cluster[replication] = float(np.mean(cluster_gaps))
        bootstrap_difference[replication] = float(np.mean(differences))

    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_difference, [tail, 1.0 - tail], method="linear")
    mean_cluster = float(np.mean(observed_cluster))
    mean_gics = float(np.mean(observed_gics))
    observed_difference = mean_cluster - mean_gics
    supported = bool(mean_cluster > 0 and lower > 0)
    result = {
        "annual_aggregation": "equal_weight_mean_across_evaluation_years",
        "annual_results": annual_records,
        "mean_gics_dependence_gap": mean_gics,
        "mean_cluster_dependence_gap": mean_cluster,
        "mean_cluster_minus_gics_gap": observed_difference,
        "bootstrap_replications": replications,
        "bootstrap_candidate_draws": candidate_count,
        "degenerate_candidate_rejections": rejected_count,
        "degenerate_sample_action": ("discard_whole_paired_replication_and_continue_rng_stream"),
        "block_length_trading_days": block_length,
        "confidence_level": confidence_level,
        "confidence_interval_method": "two_sided_percentile_linear",
        "cluster_minus_gics_confidence_interval": {
            "lower": float(lower),
            "upper": float(upper),
        },
        "bootstrap_mean_cluster_gap": float(bootstrap_cluster.mean()),
        "bootstrap_mean_cluster_minus_gics_gap": float(bootstrap_difference.mean()),
        "bootstrap_standard_error_cluster_minus_gics_gap": float(bootstrap_difference.std(ddof=1)),
        "decision": "supported" if supported else "not_supported",
        "support_conditions": {
            "mean_cluster_gap_positive": bool(mean_cluster > 0),
            "difference_confidence_interval_strictly_positive": bool(lower > 0),
        },
    }
    manifest = {
        "schema_version": 1,
        "manifest_type": "h1_bootstrap_seed_manifest",
        "bit_generator": "PCG64DXSM",
        "seed_sequence": f"SeedSequence([{base_seed}, 1])",
        "seed_components": [base_seed, 1],
        "replications": replications,
        "candidate_draws": candidate_count,
        "degenerate_candidate_rejections": rejected_count,
        "degenerate_sample_action": ("discard_whole_paired_replication_and_continue_rng_stream"),
        "block_length_trading_days": block_length,
        "evaluation_years": [annual.year for annual in years],
        "observation_count": len(dates),
        "indices_shape": list(indices.shape),
        "indices_dtype": str(indices.dtype),
        "indices_sha256": _array_sha256(indices),
        "replicate_statistics_sha256": _array_sha256(
            np.column_stack([bootstrap_cluster, bootstrap_difference])
        ),
    }
    return result, manifest


def hypothesis_decisions(
    h1: Mapping[str, Any],
    dm: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    *,
    vine_eligible: bool,
) -> dict[str, Any]:
    """Apply the frozen H1-H3 support and ranking rules."""

    vine_supports = sum(bool(row["favours_vine_after_holm"]) for row in dm)
    gaussian_supports = sum(bool(row["favours_gaussian_after_holm"]) for row in dm)
    if vine_supports == len(dm):
        h2_status = "full_support"
    elif vine_supports > 0 and gaussian_supports == 0:
        h2_status = "partial_support"
    else:
        h2_status = "no_support"

    ranks = {str(row["model_id"]): float(row["average_primary_score_rank"]) for row in summaries}
    minimum_rank = min(ranks.values())
    first_models = sorted(model for model, rank in ranks.items() if rank == minimum_rank)
    m4 = next(row for row in summaries if row["model_id"] == "M4")
    h3_supported = bool(
        first_models == ["M4"] and not m4["systematic_calibration_failure"] and vine_eligible
    )
    eligible = [
        str(row["model_id"])
        for row in summaries
        if not row["systematic_calibration_failure"] and (row["model_id"] != "M4" or vine_eligible)
    ]
    best_eligible = (
        sorted(eligible, key=lambda model: (ranks[model], model))[0] if eligible else None
    )
    return {
        "H1_cluster_information": {
            "status": h1["decision"],
            "rule": "positive mean cluster gap and strictly positive paired bootstrap CI",
        },
        "H2_copula_specification": {
            "status": h2_status,
            "vine_favouring_adjusted_tests": vine_supports,
            "gaussian_favouring_adjusted_tests": gaussian_supports,
            "required_tests_for_full_support": len(dm),
        },
        "H3_integrated_model": {
            "status": "supported" if h3_supported else "not_supported",
            "unique_first_rank_models": first_models,
            "m4_systematic_calibration_failure": bool(m4["systematic_calibration_failure"]),
            "m4_vine_quality_eligible": vine_eligible,
            "best_eligible_model": best_eligible,
        },
    }


def build_evaluation_outputs(
    historical: pd.DataFrame,
    gaussian: pd.DataFrame,
    vine: pd.DataFrame,
    stock_returns: pd.DataFrame,
    assignments: Mapping[str, Any],
    clustering_diagnostics: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    *,
    vine_eligible: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Build daily scores, statistical results, and the bootstrap seed manifest."""

    protocol = validate_evaluation_protocol(evaluation_config, model_config)
    forecasts, validation = combine_forecasts(historical, gaussian, vine, protocol["evaluation"])
    scores = score_forecasts(forecasts)
    calibration, annual_calibration = calibration_results(scores, protocol["calibration"])
    dm = dm_results(scores, protocol["h2"])
    summaries = model_summaries(scores, calibration)
    h1_years = prepare_h1_years(stock_returns, assignments, clustering_diagnostics)
    h1, seed_manifest = h1_bootstrap(
        h1_years,
        replications=int(protocol["h1"]["replications"]),
        block_length=int(protocol["h1"]["block_length_trading_days"]),
        confidence_level=float(protocol["h1"]["confidence_level"]),
        base_seed=int(protocol["h1"]["seed_components"][0]),
    )
    audit = {
        "schema_version": 1,
        "gate_name": "model_evaluation_v1",
        "status": "pass",
        "issues": [],
        "forecast_validation": validation,
        "model_summaries": summaries,
        "full_period_calibration": calibration,
        "annual_calibration_descriptive": annual_calibration,
        "diebold_mariano_comparisons": dm,
        "h1_bootstrap": h1,
        "hypotheses": hypothesis_decisions(h1, dm, summaries, vine_eligible=vine_eligible),
    }
    return scores, audit, seed_manifest


def _validate_upstream_output(
    audit: Mapping[str, Any],
    *,
    audit_name: str,
    output_key: str,
    output_path: Path,
    scope: str,
) -> None:
    if audit.get("status") != "pass" or audit.get("reporting_scope") != scope:
        raise RuntimeError(f"{audit_name} is not a passed, scope-matched upstream audit")
    outputs = _mapping(audit.get("outputs"), f"{audit_name}.outputs")
    record = _mapping(outputs.get(output_key), f"{audit_name}.{output_key}")
    if record.get("sha256") != sha256_file(output_path):
        raise RuntimeError(f"{audit_name} does not bind the current {output_key} artifact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-forecasts",
        type=Path,
        default=PROJECT_ROOT / "data/processed/historical_simulation_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--gaussian-forecasts",
        type=Path,
        default=PROJECT_ROOT / "data/processed/gaussian_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--vine-forecasts",
        type=Path,
        default=PROJECT_ROOT / "data/processed/vine_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--stock-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/portfolio_constituent_simple_returns.parquet",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=PROJECT_ROOT / "data/processed/annual_group_assignments.json",
    )
    parser.add_argument(
        "--historical-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/historical_simulation_quality.json",
    )
    parser.add_argument(
        "--gaussian-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/gaussian_copula_quality.json",
    )
    parser.add_argument(
        "--vine-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/vine_copula_quality.json",
    )
    parser.add_argument(
        "--clustering-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/clustering_diagnostics.json",
    )
    parser.add_argument(
        "--model-config", type=Path, default=PROJECT_ROOT / "config/model_config.toml"
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=PROJECT_ROOT / "config/evaluation_config.toml",
    )
    parser.add_argument(
        "--foundation-status",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_gate_status.json",
    )
    parser.add_argument(
        "--daily-scores-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/risk_evaluation_daily.parquet",
    )
    parser.add_argument(
        "--seed-manifest-output",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/inference_seed_manifest.json",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/model_evaluation.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    with args.evaluation_config.open("rb") as handle:
        evaluation_config = tomllib.load(handle)
    scope = reporting_scope(json.loads(args.foundation_status.read_text(encoding="utf-8")))
    historical_audit = json.loads(args.historical_audit.read_text(encoding="utf-8"))
    gaussian_audit = json.loads(args.gaussian_audit.read_text(encoding="utf-8"))
    vine_audit = json.loads(args.vine_audit.read_text(encoding="utf-8"))
    clustering_audit = json.loads(args.clustering_audit.read_text(encoding="utf-8"))
    _validate_upstream_output(
        historical_audit,
        audit_name="historical audit",
        output_key="historical_simulation_risk_forecasts",
        output_path=args.historical_forecasts,
        scope=scope,
    )
    _validate_upstream_output(
        gaussian_audit,
        audit_name="Gaussian audit",
        output_key="gaussian_risk_forecasts",
        output_path=args.gaussian_forecasts,
        scope=scope,
    )
    _validate_upstream_output(
        vine_audit,
        audit_name="vine audit",
        output_key="vine_risk_forecasts",
        output_path=args.vine_forecasts,
        scope=scope,
    )
    _validate_upstream_output(
        clustering_audit,
        audit_name="clustering audit",
        output_key="annual_group_assignments",
        output_path=args.assignments,
        scope=scope,
    )
    clustering_inputs = _mapping(clustering_audit.get("inputs"), "clustering audit inputs")
    returns_record = _mapping(
        clustering_inputs.get("simple_return_panel"), "clustering simple-return input"
    )
    if returns_record.get("sha256") != sha256_file(args.stock_returns):
        raise RuntimeError("clustering audit does not bind the current stock-return panel")

    scores, audit, seed_manifest = build_evaluation_outputs(
        pd.read_parquet(args.historical_forecasts),
        pd.read_parquet(args.gaussian_forecasts),
        pd.read_parquet(args.vine_forecasts),
        pd.read_parquet(args.stock_returns),
        json.loads(args.assignments.read_text(encoding="utf-8")),
        clustering_audit,
        evaluation_config,
        model_config,
        vine_eligible=bool(vine_audit.get("eligible_to_be_declared_best")),
    )
    seed_manifest["reporting_scope"] = scope
    write_parquet_atomic(args.daily_scores_output, scores)
    write_json_atomic(args.seed_manifest_output, seed_manifest)
    audit.update(
        {
            "reporting_scope": scope,
            "method": {
                "evaluation_protocol": project_path(args.evaluation_config),
                "primary_scores": list(PRIMARY_SCORES),
                "calibration_multiplicity": "Holm family-wise correction",
                "dm_difference": "vine minus matched Gaussian",
                "h1_annual_aggregation": "equal-year mean",
            },
            "inputs": {
                "historical_forecasts": {
                    "path": project_path(args.historical_forecasts),
                    "sha256": sha256_file(args.historical_forecasts),
                },
                "gaussian_forecasts": {
                    "path": project_path(args.gaussian_forecasts),
                    "sha256": sha256_file(args.gaussian_forecasts),
                },
                "vine_forecasts": {
                    "path": project_path(args.vine_forecasts),
                    "sha256": sha256_file(args.vine_forecasts),
                },
                "stock_returns": {
                    "path": project_path(args.stock_returns),
                    "sha256": sha256_file(args.stock_returns),
                },
                "annual_group_assignments": {
                    "path": project_path(args.assignments),
                    "sha256": sha256_file(args.assignments),
                },
                "historical_audit": {
                    "path": project_path(args.historical_audit),
                    "sha256": sha256_file(args.historical_audit),
                },
                "gaussian_audit": {
                    "path": project_path(args.gaussian_audit),
                    "sha256": sha256_file(args.gaussian_audit),
                },
                "vine_audit": {
                    "path": project_path(args.vine_audit),
                    "sha256": sha256_file(args.vine_audit),
                },
                "clustering_audit": {
                    "path": project_path(args.clustering_audit),
                    "sha256": sha256_file(args.clustering_audit),
                },
                "model_config": {
                    "path": project_path(args.model_config),
                    "sha256": sha256_file(args.model_config),
                },
                "evaluation_config": {
                    "path": project_path(args.evaluation_config),
                    "sha256": sha256_file(args.evaluation_config),
                },
                "foundation_status": {
                    "path": project_path(args.foundation_status),
                    "sha256": sha256_file(args.foundation_status),
                },
            },
            "outputs": {
                "risk_evaluation_daily": {
                    "path": project_path(args.daily_scores_output),
                    "sha256": sha256_file(args.daily_scores_output),
                    "rows": len(scores),
                },
                "inference_seed_manifest": {
                    "path": project_path(args.seed_manifest_output),
                    "sha256": sha256_file(args.seed_manifest_output),
                },
            },
        }
    )
    write_json_atomic(args.audit_output, audit)
    decisions = audit["hypotheses"]
    print(
        f"model_evaluation={audit['status']} forecasts={len(scores)} "
        f"H1={decisions['H1_cluster_information']['status']} "
        f"H2={decisions['H2_copula_specification']['status']} "
        f"H3={decisions['H3_integrated_model']['status']}"
    )
    print(f"Audit: {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
