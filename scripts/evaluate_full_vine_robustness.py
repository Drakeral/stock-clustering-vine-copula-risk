#!/usr/bin/env python3
"""Evaluate full ten-tree vines against the matched primary three-tree vines."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.evaluate_risk_models import (
        EXCEPTION_COLUMNS,
        FORECAST_COLUMNS,
        PRIMARY_SCORES,
        score_forecasts,
    )
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        sha256_file,
        write_json_atomic,
        write_parquet_atomic,
    )
    from scripts.research_methods import (
        christoffersen_conditional_coverage_p_value,
        christoffersen_independence,
        diebold_mariano,
        holm_adjust,
        kupiec_unconditional_coverage,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from evaluate_risk_models import (
        EXCEPTION_COLUMNS,
        FORECAST_COLUMNS,
        PRIMARY_SCORES,
        score_forecasts,
    )
    from pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        sha256_file,
        write_json_atomic,
        write_parquet_atomic,
    )
    from research_methods import (
        christoffersen_conditional_coverage_p_value,
        christoffersen_independence,
        diebold_mariano,
        holm_adjust,
        kupiec_unconditional_coverage,
    )


MODEL_GROUPINGS = {"M2": "gics", "M4": "hierarchical"}
VARIANT_ORDER = ("M2_t3", "M2_t10", "M4_t3", "M4_t10")
DAILY_COLUMNS = (
    "date",
    "year",
    "variant_id",
    "model_id",
    "grouping_id",
    "truncation_level",
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
REFIT_IDENTITY_COLUMNS = (
    "source_copula_refit_id",
    "matched_gaussian_refit_id",
    "refit_date",
    "training_start",
    "training_end",
    "training_observations",
    "dimension",
    "group_order_json",
    "fit_seed_components",
    "base_uniform_sha256",
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a table")
    return cast(Mapping[str, Any], value)


def _require_settings(settings: Mapping[str, tuple[object, object]], error_prefix: str) -> None:
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in settings.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"{error_prefix}: {mismatches}")


def validate_robustness_protocol(
    robustness_config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    """Require the exact pre-result full-vine robustness protocol."""

    if robustness_config.get("schema_version") != 1:
        raise ValueError("unsupported full-vine robustness configuration schema")
    if robustness_config.get("protocol_status") != "frozen_before_full_vine_robustness_results":
        raise ValueError("full-vine protocol was not frozen before robustness results")
    if robustness_config.get("analysis_role") != "exploratory_robustness":
        raise ValueError("full-vine analysis role must remain exploratory robustness")

    evaluation = _mapping(robustness_config.get("evaluation"), "evaluation")
    dm = _mapping(robustness_config.get("dm"), "dm")
    calibration = _mapping(robustness_config.get("calibration"), "calibration")
    interpretation = _mapping(robustness_config.get("interpretation"), "interpretation")
    vine = _mapping(model_config.get("vine"), "model vine")
    primary_level = int(vine.get("primary_truncation_tree", -1))
    full_level = int(vine.get("robustness_truncation_tree", -1))

    _require_settings(
        {
            "evaluation.model_ids": (evaluation.get("model_ids"), ["M2", "M4"]),
            "evaluation.primary_truncation_level": (
                evaluation.get("primary_truncation_level"),
                primary_level,
            ),
            "evaluation.full_truncation_level": (
                evaluation.get("full_truncation_level"),
                full_level,
            ),
            "evaluation.required_forecast_statuses": (
                evaluation.get("required_forecast_statuses"),
                ["ok", "fallback"],
            ),
            "evaluation.primary_score_columns": (
                evaluation.get("primary_score_columns"),
                list(PRIMARY_SCORES),
            ),
            "evaluation.copula_log_score_direction": (
                evaluation.get("copula_log_score_direction"),
                "higher_is_better",
            ),
            "dm.loss_difference": (
                dm.get("loss_difference"),
                "full_vine_minus_truncated_vine",
            ),
            "dm.comparisons": (
                dm.get("comparisons"),
                [
                    {"model_id": "M2", "grouping_id": "gics"},
                    {"model_id": "M4", "grouping_id": "hierarchical"},
                ],
            ),
            "dm.score_columns": (dm.get("score_columns"), list(PRIMARY_SCORES)),
            "dm.test": (
                dm.get("test"),
                "diebold_mariano_two_sided_then_direction_check",
            ),
            "dm.hac": (dm.get("hac"), "newey_west_bartlett"),
            "dm.hac_lag": (dm.get("hac_lag"), 7),
            "dm.multiplicity": (dm.get("multiplicity"), "holm_familywise"),
            "dm.family_size": (dm.get("family_size"), 6),
            "calibration.confidence_levels": (
                calibration.get("confidence_levels"),
                [0.95, 0.99],
            ),
            "calibration.tests": (
                calibration.get("tests"),
                ["kupiec_unconditional_coverage", "christoffersen_independence"],
            ),
            "calibration.variants": (
                calibration.get("variants"),
                list(VARIANT_ORDER),
            ),
            "calibration.multiplicity": (
                calibration.get("multiplicity"),
                "holm_familywise",
            ),
            "calibration.family_size": (calibration.get("family_size"), 16),
            "interpretation.role": (
                interpretation.get("role"),
                "robustness_not_new_confirmatory_hypothesis",
            ),
            "interpretation.decision_rule": (
                interpretation.get("decision_rule"),
                "report_adjusted_pairwise_loss_directions_without_changing_H1_H3",
            ),
            "interpretation.h2_rule": (
                interpretation.get("h2_rule"),
                "primary_tree_3_confirmatory_results_remain_authoritative",
            ),
            "interpretation.log_score_comparison": (
                interpretation.get("log_score_comparison"),
                "descriptive_within_matched_grouping_only",
            ),
        },
        "full-vine robustness protocol differs from the implementation",
    )
    tolerance = float(evaluation.get("realised_loss_identity_tolerance", float("nan")))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("realised-loss identity tolerance must be positive and finite")
    if float(dm.get("familywise_alpha", -1)) != 0.05:
        raise ValueError("DM family-wise alpha must remain 0.05")
    if float(calibration.get("familywise_alpha", -1)) != 0.05:
        raise ValueError("calibration family-wise alpha must remain 0.05")
    if not (1 <= primary_level < full_level < int(vine.get("dimension", 0))):
        raise ValueError("model configuration has invalid frozen vine truncation levels")
    return {
        "evaluation": evaluation,
        "dm": dm,
        "calibration": calibration,
        "interpretation": interpretation,
    }


def _normalize_forecasts(
    source: pd.DataFrame,
    *,
    source_name: str,
    evaluation: Mapping[str, Any],
) -> pd.DataFrame:
    if set(source.columns) != set(FORECAST_COLUMNS):
        missing = sorted(set(FORECAST_COLUMNS) - set(source.columns))
        extra = sorted(set(source.columns) - set(FORECAST_COLUMNS))
        raise ValueError(f"{source_name} schema mismatch; missing={missing}, extra={extra}")
    frame = source.loc[:, FORECAST_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    _validate_forecast_structure(frame, source_name, evaluation)
    _validate_forecast_numbers(frame, source_name, evaluation)
    return frame


def _validate_forecast_structure(
    frame: pd.DataFrame, source_name: str, evaluation: Mapping[str, Any]
) -> None:
    if frame["date"].isna().any() or frame.duplicated(["model_id", "date"]).any():
        raise ValueError(f"{source_name} has missing dates or duplicate model-date rows")
    if set(frame["model_id"].astype(str)) != set(MODEL_GROUPINGS):
        raise ValueError(f"{source_name} must contain exactly M2 and M4")
    for model_id, grouping_id in MODEL_GROUPINGS.items():
        observed = set(frame.loc[frame["model_id"] == model_id, "grouping_id"].astype(str))
        if observed != {grouping_id}:
            raise ValueError(f"{source_name} {model_id} grouping is not {grouping_id}")
    if not frame["forecast_status"].isin(evaluation["required_forecast_statuses"]).all():
        raise ValueError(f"{source_name} contains failed or unsupported forecasts")


def _validate_forecast_numbers(
    frame: pd.DataFrame, source_name: str, evaluation: Mapping[str, Any]
) -> None:
    numeric_columns = [
        "var_95",
        "var_975",
        "var_99",
        "es_975",
        "realised_simple_return",
        "realised_loss",
        "copula_log_score",
    ]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{source_name} contains non-finite forecast values")
    if bool(
        (
            (frame["var_95"] > frame["var_975"])
            | (frame["var_975"] > frame["var_99"])
            | (frame["es_975"] < frame["var_975"])
            | (frame["es_975"] <= 0)
        ).any()
    ):
        raise ValueError(f"{source_name} contains invalid VaR or ES ordering")
    sign_error = np.abs(frame["realised_loss"] + frame["realised_simple_return"])
    if float(sign_error.max()) > float(evaluation["realised_loss_identity_tolerance"]):
        raise ValueError(f"{source_name} violates the realised-loss sign identity")
    for row in frame.itertuples(index=False):
        if list(row.seed_components) != [5110, row.date.year, row.date.month]:
            raise ValueError(f"{source_name} has a seed that does not match its date")


def _stable_value(value: object) -> str:
    if isinstance(value, list | tuple | np.ndarray):
        normalized = [item.item() if isinstance(item, np.generic) else item for item in value]
        return json.dumps(normalized, separators=(",", ":"))
    if isinstance(value, np.generic):
        return str(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def validate_refit_identity(
    primary: pd.DataFrame,
    full: pd.DataFrame,
    *,
    primary_level: int,
    full_level: int,
) -> dict[str, Any]:
    """Verify that primary and full vines share the same monthly information sets."""

    required = {
        "year",
        "month",
        "universe_variant",
        "model_id",
        "grouping_id",
        "truncation_level",
        "pair_count",
        "number_parameters",
        "aic",
        "bic",
        "whole_vine_fallback",
        *REFIT_IDENTITY_COLUMNS,
    }
    for name, frame in (("primary", primary), ("full", full)):
        _validate_refit_frame(frame, name=name, required=required)

    if set(primary["truncation_level"].astype(int)) != {primary_level}:
        raise ValueError("primary refits have the wrong truncation level")
    if set(full["truncation_level"].astype(int)) != {full_level}:
        raise ValueError("full refits have the wrong truncation level")
    keys = ["year", "month", "universe_variant", "model_id", "grouping_id"]
    primary_keys = primary.loc[:, keys].sort_values(keys).reset_index(drop=True)
    full_keys = full.loc[:, keys].sort_values(keys).reset_index(drop=True)
    if not primary_keys.equals(full_keys):
        raise ValueError("primary and full refit keys are not identical")

    left = primary.set_index(keys).sort_index()
    right = full.set_index(keys).sort_index()
    mismatched: list[str] = []
    for column in REFIT_IDENTITY_COLUMNS:
        left_values = left[column].map(_stable_value)
        right_values = right[column].map(_stable_value)
        if not left_values.equals(right_values):
            mismatched.append(column)
    if mismatched:
        raise ValueError(f"primary and full refit identities differ: {mismatched}")

    dimension = left["dimension"].astype(int)
    expected_primary_pairs = _expected_pair_counts(dimension, primary_level)
    expected_full_pairs = _expected_pair_counts(dimension, full_level)
    if not left["pair_count"].astype(int).equals(expected_primary_pairs):
        raise ValueError("primary refit pair counts do not match their truncation level")
    if not right["pair_count"].astype(int).equals(expected_full_pairs):
        raise ValueError("full refit pair counts do not match their truncation level")
    return {
        "matched_refit_count": len(left),
        "model_refit_counts": {
            model_id: int((primary["model_id"] == model_id).sum()) for model_id in MODEL_GROUPINGS
        },
        "common_information_set_columns": list(REFIT_IDENTITY_COLUMNS),
        "primary_pair_count_total": int(primary["pair_count"].sum()),
        "full_pair_count_total": int(full["pair_count"].sum()),
    }


def _validate_refit_frame(frame: pd.DataFrame, *, name: str, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} refits are missing columns: {missing}")
    if frame.duplicated(["year", "month", "model_id"]).any():
        raise ValueError(f"{name} refits contain duplicate model-month rows")
    if set(frame["model_id"].astype(str)) != set(MODEL_GROUPINGS):
        raise ValueError(f"{name} refits must contain exactly M2 and M4")
    for model_id, grouping_id in MODEL_GROUPINGS.items():
        groupings = set(frame.loc[frame["model_id"] == model_id, "grouping_id"].astype(str))
        if groupings != {grouping_id}:
            raise ValueError(f"{name} refit grouping differs for {model_id}")


def _expected_pair_counts(dimension: pd.Series, truncation_level: int) -> pd.Series:
    return dimension.map(lambda value: sum(value - tree for tree in range(1, truncation_level + 1)))


def _refit_complexity_summary(primary: pd.DataFrame, full: pd.DataFrame) -> dict[str, Any]:
    """Summarize matched fitted complexity without treating fit criteria as OOS loss."""

    summaries: dict[str, dict[str, float | int]] = {}
    for label, frame in (("primary_tree_3", primary), ("full_tree_10", full)):
        accepted = frame.loc[~frame["whole_vine_fallback"]]
        metrics = accepted[["number_parameters", "aic", "bic"]].to_numpy(dtype=float)
        if accepted.empty or not np.isfinite(metrics).all():
            raise ValueError(f"{label} accepted refits have invalid fit diagnostics")
        summaries[label] = {
            "accepted_refits": len(accepted),
            "mean_number_parameters": float(accepted["number_parameters"].mean()),
            "median_number_parameters": float(accepted["number_parameters"].median()),
            "minimum_number_parameters": float(accepted["number_parameters"].min()),
            "maximum_number_parameters": float(accepted["number_parameters"].max()),
            "mean_aic": float(accepted["aic"].mean()),
            "mean_bic": float(accepted["bic"].mean()),
        }
    primary_summary = summaries["primary_tree_3"]
    full_summary = summaries["full_tree_10"]
    return {
        **summaries,
        "mean_parameter_increase": (
            full_summary["mean_number_parameters"] - primary_summary["mean_number_parameters"]
        ),
        "interpretation": "descriptive_in_sample_fit_complexity_not_forecast_loss_evidence",
    }


def combine_robustness_forecasts(
    primary: pd.DataFrame,
    full: pd.DataFrame,
    evaluation: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match the two truncation variants and return scored daily records."""

    primary_frame = _normalize_forecasts(
        primary, source_name="primary forecasts", evaluation=evaluation
    )
    full_frame = _normalize_forecasts(
        full, source_name="full-vine forecasts", evaluation=evaluation
    )
    keys = ["date", "model_id", "grouping_id"]
    left = primary_frame.set_index(keys).sort_index()
    right = full_frame.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("primary and full-vine forecast keys are not identical")

    tolerance = float(evaluation["realised_loss_identity_tolerance"])
    realised_error = float(np.max(np.abs(left["realised_loss"] - right["realised_loss"])))
    simple_error = float(
        np.max(np.abs(left["realised_simple_return"] - right["realised_simple_return"]))
    )
    if max(realised_error, simple_error) > tolerance:
        raise ValueError("primary and full vines do not share identical realised returns")
    exact_columns = ("seed_components", "margin_fallback_count")
    differences = [
        column
        for column in exact_columns
        if not left[column].map(_stable_value).equals(right[column].map(_stable_value))
    ]
    if differences:
        raise ValueError(f"primary and full forecast foundations differ: {differences}")

    levels = (
        (primary_frame, int(evaluation["primary_truncation_level"])),
        (full_frame, int(evaluation["full_truncation_level"])),
    )
    scored_frames: list[pd.DataFrame] = []
    for frame, level in levels:
        scored = score_forecasts(frame)
        scored.insert(2, "variant_id", scored["model_id"] + f"_t{level}")
        scored.insert(5, "truncation_level", level)
        scored_frames.append(scored)
    combined = pd.concat(scored_frames, ignore_index=True)
    variant_order = {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    combined["_variant_order"] = combined["variant_id"].map(variant_order)
    combined = combined.sort_values(["date", "_variant_order"]).drop(columns="_variant_order")
    combined = combined.loc[:, DAILY_COLUMNS].reset_index(drop=True)
    observed_variants = set(combined["variant_id"].astype(str))
    if observed_variants != set(VARIANT_ORDER):
        raise ValueError("scored robustness variants are incomplete")
    counts = combined.groupby("variant_id")["date"].nunique().to_dict()
    if len(set(counts.values())) != 1:
        raise ValueError("robustness variants do not share the same dates")
    dates = pd.DatetimeIndex(combined["date"].drop_duplicates().sort_values())
    return combined, {
        "evaluation_date_count": len(dates),
        "forecast_count": len(combined),
        "variant_date_counts": {key: int(counts[key]) for key in VARIANT_ORDER},
        "evaluation_start": dates.min().date().isoformat(),
        "evaluation_end": dates.max().date().isoformat(),
        "maximum_realised_loss_identity_error": realised_error,
        "maximum_realised_simple_return_identity_error": simple_error,
        "realised_loss_identity_tolerance": tolerance,
        "common_random_numbers_verified": True,
        "margin_state_identity_verified": True,
    }


def robustness_calibration(
    scores: pd.DataFrame, calibration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Run the frozen 16-test calibration family across four vine variants."""

    rows: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        variant = scores.loc[scores["variant_id"] == variant_id].sort_values("date")
        for raw_confidence in calibration["confidence_levels"]:
            confidence = float(raw_confidence)
            indicators = variant[EXCEPTION_COLUMNS[confidence]].to_numpy(dtype=bool)
            coverage = kupiec_unconditional_coverage(indicators, confidence)
            independence = christoffersen_independence(indicators)
            rows.append(
                {
                    "variant_id": variant_id,
                    "model_id": str(variant["model_id"].iloc[0]),
                    "grouping_id": str(variant["grouping_id"].iloc[0]),
                    "truncation_level": int(variant["truncation_level"].iloc[0]),
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
        for row in rows
        for value in (row["kupiec_p_value"], row["christoffersen_independence_p_value"])
    ]
    if len(raw_p_values) != int(calibration["family_size"]):
        raise AssertionError("full-vine calibration family has the wrong size")
    adjusted = iter(holm_adjust(raw_p_values).tolist())
    alpha = float(calibration["familywise_alpha"])
    for row in rows:
        row["kupiec_holm_p_value"] = next(adjusted)
        row["kupiec_reject_after_holm"] = row["kupiec_holm_p_value"] < alpha
        row["christoffersen_independence_holm_p_value"] = next(adjusted)
        row["christoffersen_independence_reject_after_holm"] = (
            row["christoffersen_independence_holm_p_value"] < alpha
        )
    return rows


def robustness_dm(scores: pd.DataFrame, dm: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare full-minus-truncated losses using the frozen six-test family."""

    rows: list[dict[str, Any]] = []
    primary_level = int(scores["truncation_level"].min())
    full_level = int(scores["truncation_level"].max())
    for comparison in dm["comparisons"]:
        model_id = str(comparison["model_id"])
        model = scores.loc[scores["model_id"] == model_id]
        primary = model.loc[model["truncation_level"] == primary_level].set_index("date")
        full = model.loc[model["truncation_level"] == full_level].set_index("date")
        if not primary.index.equals(full.index):
            raise ValueError(f"DM dates are not matched for {model_id}")
        for score_column in dm["score_columns"]:
            differential = full[str(score_column)].to_numpy(dtype=float) - primary[
                str(score_column)
            ].to_numpy(dtype=float)
            result = diebold_mariano(differential, hac_lag=int(dm["hac_lag"]))
            rows.append(
                {
                    "comparison_id": f"{model_id}_t{full_level}_minus_t{primary_level}:"
                    f"{score_column}",
                    "model_id": model_id,
                    "grouping_id": comparison["grouping_id"],
                    "score": score_column,
                    "observations": result.observations,
                    "mean_loss_difference": result.mean_difference,
                    "dm_statistic": result.statistic,
                    "p_value_two_sided": result.p_value_two_sided,
                    "hac_lag": result.hac_lag,
                }
            )
    if len(rows) != int(dm["family_size"]):
        raise AssertionError("full-vine DM family has the wrong size")
    adjusted = holm_adjust([row["p_value_two_sided"] for row in rows])
    alpha = float(dm["familywise_alpha"])
    for row, adjusted_p in zip(rows, adjusted, strict=True):
        difference = float(row["mean_loss_difference"])
        row["holm_p_value"] = float(adjusted_p)
        row["significant_after_holm"] = bool(adjusted_p < alpha)
        row["direction"] = (
            "full_vine_lower_loss"
            if difference < 0
            else "truncated_vine_lower_loss"
            if difference > 0
            else "equal_loss"
        )
        row["favours_full_vine_after_holm"] = bool(adjusted_p < alpha and difference < 0)
        row["favours_truncated_vine_after_holm"] = bool(adjusted_p < alpha and difference > 0)
    return rows


def _variant_summaries(
    scores: pd.DataFrame, calibration: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        variant = scores.loc[scores["variant_id"] == variant_id]
        row: dict[str, Any] = {
            "variant_id": variant_id,
            "model_id": str(variant["model_id"].iloc[0]),
            "grouping_id": str(variant["grouping_id"].iloc[0]),
            "truncation_level": int(variant["truncation_level"].iloc[0]),
            "observations": len(variant),
            "mean_quantile_loss_95": float(variant["quantile_loss_95"].mean()),
            "mean_quantile_loss_99": float(variant["quantile_loss_99"].mean()),
            "mean_fz0_975": float(variant["fz0_975"].mean()),
            "mean_copula_log_score": float(variant["copula_log_score"].mean()),
            "whole_vine_fallback_dates": int(variant["whole_vine_fallback"].sum()),
        }
        for confidence, exception_column in EXCEPTION_COLUMNS.items():
            suffix = str(confidence).replace("0.", "").replace(".", "")
            row[f"exceptions_{suffix}"] = int(variant[exception_column].sum())
            row[f"exception_rate_{suffix}"] = float(variant[exception_column].mean())
        row["adjusted_calibration_rejection_count"] = sum(
            int(item["kupiec_reject_after_holm"])
            + int(item["christoffersen_independence_reject_after_holm"])
            for item in calibration
            if item["variant_id"] == variant_id
        )
        rows.append(row)
    return rows


def _log_score_comparisons(scores: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, grouping_id in MODEL_GROUPINGS.items():
        model = scores.loc[scores["model_id"] == model_id]
        levels = sorted(model["truncation_level"].unique())
        primary = model.loc[model["truncation_level"] == levels[0]].set_index("date")
        full = model.loc[model["truncation_level"] == levels[1]].set_index("date")
        difference = full["copula_log_score"] - primary["copula_log_score"]
        rows.append(
            {
                "model_id": model_id,
                "grouping_id": grouping_id,
                "inference_role": "descriptive",
                "primary_mean_copula_log_score": float(primary["copula_log_score"].mean()),
                "full_mean_copula_log_score": float(full["copula_log_score"].mean()),
                "mean_full_minus_primary_log_score": float(difference.mean()),
                "direction": (
                    "full_vine_higher_log_score"
                    if difference.mean() > 0
                    else "truncated_vine_higher_log_score"
                    if difference.mean() < 0
                    else "equal_log_score"
                ),
            }
        )
    return rows


def build_robustness_outputs(
    primary_forecasts: pd.DataFrame,
    full_forecasts: pd.DataFrame,
    primary_refits: pd.DataFrame,
    full_refits: pd.DataFrame,
    robustness_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build matched full-vine robustness scores and statistical diagnostics."""

    protocol = validate_robustness_protocol(robustness_config, model_config)
    evaluation = protocol["evaluation"]
    refit_validation = validate_refit_identity(
        primary_refits,
        full_refits,
        primary_level=int(evaluation["primary_truncation_level"]),
        full_level=int(evaluation["full_truncation_level"]),
    )
    scores, forecast_validation = combine_robustness_forecasts(
        primary_forecasts, full_forecasts, evaluation
    )
    calibration = robustness_calibration(scores, protocol["calibration"])
    dm = robustness_dm(scores, protocol["dm"])
    full_supports = sum(bool(row["favours_full_vine_after_holm"]) for row in dm)
    truncated_supports = sum(bool(row["favours_truncated_vine_after_holm"]) for row in dm)
    evidence = (
        "full_vine_favoured"
        if full_supports and not truncated_supports
        else "truncated_vine_favoured"
        if truncated_supports and not full_supports
        else "mixed"
        if full_supports and truncated_supports
        else "no_adjusted_loss_difference"
    )
    audit = {
        "schema_version": 1,
        "gate_name": "full_vine_robustness_evaluation_v1",
        "analysis_role": "exploratory_robustness",
        "status": "pass",
        "issues": [],
        "refit_validation": refit_validation,
        "refit_complexity": _refit_complexity_summary(primary_refits, full_refits),
        "forecast_validation": forecast_validation,
        "variant_summaries": _variant_summaries(scores, calibration),
        "full_period_calibration": calibration,
        "diebold_mariano_comparisons": dm,
        "copula_log_score_comparisons": _log_score_comparisons(scores),
        "interpretation": {
            "adjusted_loss_evidence": evidence,
            "full_vine_favouring_adjusted_tests": full_supports,
            "truncated_vine_favouring_adjusted_tests": truncated_supports,
            "primary_hypotheses_changed": False,
            "primary_tree_3_confirmatory_results_remain_authoritative": True,
        },
    }
    return scores, audit


def _validate_upstream_output(
    audit: Mapping[str, Any],
    *,
    audit_name: str,
    analysis_role: str,
    truncation_level: int,
    output_key: str,
    output_path: Path,
    scope: str,
) -> None:
    if (
        audit.get("status") != "pass"
        or audit.get("reporting_scope") != scope
        or audit.get("analysis_role") != analysis_role
        or audit.get("truncation_level") != truncation_level
    ):
        raise RuntimeError(f"{audit_name} is not a passed, scope-matched vine audit")
    outputs = _mapping(audit.get("outputs"), f"{audit_name}.outputs")
    record = _mapping(outputs.get(output_key), f"{audit_name}.{output_key}")
    if record.get("sha256") != sha256_file(output_path):
        raise RuntimeError(f"{audit_name} does not bind the current {output_key}")


def _file_record(path: Path) -> dict[str, str]:
    return {"path": project_path(path), "sha256": sha256_file(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-refits",
        type=Path,
        default=PROJECT_ROOT / "data/processed/vine_copula_refits.parquet",
    )
    parser.add_argument(
        "--primary-forecasts",
        type=Path,
        default=PROJECT_ROOT / "data/processed/vine_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--primary-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/vine_copula_quality.json",
    )
    parser.add_argument(
        "--full-refits",
        type=Path,
        default=PROJECT_ROOT / "data/processed/full_vine_copula_refits.parquet",
    )
    parser.add_argument(
        "--full-forecasts",
        type=Path,
        default=PROJECT_ROOT / "data/processed/full_vine_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--full-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/full_vine_robustness_quality.json",
    )
    parser.add_argument(
        "--robustness-config",
        type=Path,
        default=PROJECT_ROOT / "config/full_vine_robustness.toml",
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
        "--daily-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/full_vine_robustness_daily.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/full_vine_robustness_evaluation.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    with args.robustness_config.open("rb") as handle:
        robustness_config = tomllib.load(handle)
    protocol = validate_robustness_protocol(robustness_config, model_config)
    evaluation = protocol["evaluation"]
    primary_level = int(evaluation["primary_truncation_level"])
    full_level = int(evaluation["full_truncation_level"])
    foundation = json.loads(args.foundation_status.read_text(encoding="utf-8"))
    require_current_hash_records(
        foundation,
        source_name=project_path(args.foundation_status),
    )
    scope = reporting_scope(foundation)
    primary_audit = json.loads(args.primary_audit.read_text(encoding="utf-8"))
    full_audit = json.loads(args.full_audit.read_text(encoding="utf-8"))
    require_current_hash_records(
        primary_audit,
        source_name=project_path(args.primary_audit),
    )
    require_current_hash_records(
        full_audit,
        source_name=project_path(args.full_audit),
    )
    _validate_upstream_output(
        primary_audit,
        audit_name="primary vine audit",
        analysis_role="primary",
        truncation_level=primary_level,
        output_key="vine_copula_refits",
        output_path=args.primary_refits,
        scope=scope,
    )
    _validate_upstream_output(
        primary_audit,
        audit_name="primary vine audit",
        analysis_role="primary",
        truncation_level=primary_level,
        output_key="vine_risk_forecasts",
        output_path=args.primary_forecasts,
        scope=scope,
    )
    _validate_upstream_output(
        full_audit,
        audit_name="full vine audit",
        analysis_role="robustness",
        truncation_level=full_level,
        output_key="full_vine_copula_refits",
        output_path=args.full_refits,
        scope=scope,
    )
    _validate_upstream_output(
        full_audit,
        audit_name="full vine audit",
        analysis_role="robustness",
        truncation_level=full_level,
        output_key="full_vine_risk_forecasts",
        output_path=args.full_forecasts,
        scope=scope,
    )
    scores, audit = build_robustness_outputs(
        pd.read_parquet(args.primary_forecasts),
        pd.read_parquet(args.full_forecasts),
        pd.read_parquet(args.primary_refits),
        pd.read_parquet(args.full_refits),
        robustness_config,
        model_config,
    )
    write_parquet_atomic(args.daily_output, scores)
    audit.update(
        {
            "reporting_scope": scope,
            "method": {
                "protocol": project_path(args.robustness_config),
                "loss_difference": "full vine minus truncated vine",
                "primary_scores": list(PRIMARY_SCORES),
                "dm_hac_lag": 7,
                "multiplicity": "Holm family-wise correction",
            },
            "inputs": {
                "primary_vine_refits": _file_record(args.primary_refits),
                "primary_vine_forecasts": _file_record(args.primary_forecasts),
                "primary_vine_audit": _file_record(args.primary_audit),
                "full_vine_refits": _file_record(args.full_refits),
                "full_vine_forecasts": _file_record(args.full_forecasts),
                "full_vine_audit": _file_record(args.full_audit),
                "robustness_config": _file_record(args.robustness_config),
                "model_config": _file_record(args.model_config),
                "foundation_status": _file_record(args.foundation_status),
            },
            "outputs": {
                "full_vine_robustness_daily": {
                    **_file_record(args.daily_output),
                    "rows": len(scores),
                }
            },
        }
    )
    write_json_atomic(args.audit_output, audit)
    interpretation = audit["interpretation"]
    print(
        f"full_vine_robustness={audit['status']} forecasts={len(scores)} "
        f"evidence={interpretation['adjusted_loss_evidence']}"
    )
    print(f"Audit: {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
