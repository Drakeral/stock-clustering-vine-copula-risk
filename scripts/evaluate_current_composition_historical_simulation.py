#!/usr/bin/env python3
"""Evaluate a current-composition backcast against the primary M0 benchmark."""

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
    from scripts.build_historical_simulation import (
        FORECAST_COLUMNS,
        WINDOW_COLUMNS,
        annual_active_sets,
        validate_historical_protocol,
        validate_portfolio_audit_binding,
    )
    from scripts.evaluate_risk_models import EXCEPTION_COLUMNS, PRIMARY_SCORES, score_forecasts
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
        rolling_historical_var_es,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from build_historical_simulation import (
        FORECAST_COLUMNS,
        WINDOW_COLUMNS,
        annual_active_sets,
        validate_historical_protocol,
        validate_portfolio_audit_binding,
    )
    from evaluate_risk_models import EXCEPTION_COLUMNS, PRIMARY_SCORES, score_forecasts
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
        rolling_historical_var_es,
    )


REFERENCE_MODEL = "M0"
ROBUSTNESS_MODEL = "M0_CC"
MODEL_ORDER = (REFERENCE_MODEL, ROBUSTNESS_MODEL)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a table")
    return cast(Mapping[str, Any], value)


def _require_settings(settings: Mapping[str, tuple[object, object]]) -> None:
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in settings.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(
            f"current-composition robustness protocol differs from the freeze: {mismatches}"
        )


def validate_robustness_protocol(
    robustness_config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    """Bind the exploratory comparison to the frozen core estimators."""

    validate_historical_protocol(model_config)
    if robustness_config.get("schema_version") != 1:
        raise ValueError("unsupported current-composition robustness schema")
    if (
        robustness_config.get("protocol_status")
        != "frozen_before_current_composition_historical_simulation_results"
    ):
        raise ValueError("current-composition protocol was not frozen before results")
    if robustness_config.get("analysis_role") != "exploratory_robustness":
        raise ValueError("current-composition analysis must remain exploratory")

    sections = {
        name: _mapping(robustness_config.get(name), name)
        for name in (
            "models",
            "portfolio",
            "forecast",
            "evaluation",
            "dm",
            "calibration",
            "interpretation",
        )
    }
    core_forecast = _mapping(model_config.get("forecast"), "core forecast")
    historical = _mapping(model_config.get("historical_simulation"), "core historical simulation")
    clustering = _mapping(model_config.get("clustering"), "core clustering")
    inference = _mapping(model_config.get("inference"), "core inference")
    models = sections["models"]
    portfolio = sections["portfolio"]
    forecast = sections["forecast"]
    evaluation = sections["evaluation"]
    dm = sections["dm"]
    calibration = sections["calibration"]
    interpretation = sections["interpretation"]
    _require_settings(
        {
            "models.reference_model_id": (models.get("reference_model_id"), REFERENCE_MODEL),
            "models.robustness_model_id": (models.get("robustness_model_id"), ROBUSTNESS_MODEL),
            "models.grouping_id": (models.get("grouping_id"), "none"),
            "portfolio.reference_training_backcast": (
                portfolio.get("reference_training_backcast"),
                "realised_historical_annual_active_sets",
            ),
            "portfolio.robustness_training_backcast": (
                portfolio.get("robustness_training_backcast"),
                "evaluation_year_active_set",
            ),
            "portfolio.weighting": (
                portfolio.get("weighting"),
                "daily_rebalanced_equal_security_weight",
            ),
            "portfolio.evaluation_return": (
                portfolio.get("evaluation_return"),
                "canonical_primary_portfolio_return",
            ),
            "forecast.window_calendar_years": (
                forecast.get("window_calendar_years"),
                core_forecast.get("training_window_calendar_years"),
            ),
            "forecast.minimum_training_observations": (
                forecast.get("minimum_training_observations"),
                core_forecast.get("minimum_training_observations"),
            ),
            "forecast.var_confidence_levels": (
                forecast.get("var_confidence_levels"),
                core_forecast.get("var_confidence_levels"),
            ),
            "forecast.es_confidence_level": (
                forecast.get("es_confidence_level"),
                core_forecast.get("es_confidence_level"),
            ),
            "forecast.quantile_method": (
                forecast.get("quantile_method"),
                historical.get("quantile_method"),
            ),
            "forecast.es_boundary_method": (
                forecast.get("es_boundary_method"),
                historical.get("es_boundary_method"),
            ),
            "evaluation.years": (
                (evaluation.get("evaluation_start_year"), evaluation.get("evaluation_end_year")),
                (clustering.get("evaluation_start_year"), clustering.get("evaluation_end_year")),
            ),
            "evaluation.primary_score_columns": (
                evaluation.get("primary_score_columns"),
                list(PRIMARY_SCORES),
            ),
            "dm.loss_difference": (
                dm.get("loss_difference"),
                "current_composition_minus_realised_history",
            ),
            "dm.hac_lag": (dm.get("hac_lag"), inference.get("dm_hac_lag")),
            "dm.family_size": (dm.get("family_size"), 3),
            "calibration.family_size": (calibration.get("family_size"), 8),
            "interpretation.role": (
                interpretation.get("role"),
                "robustness_not_new_confirmatory_hypothesis",
            ),
            "interpretation.core_H1_H2_H3_revised": (
                interpretation.get("core_H1_H2_H3_revised"),
                False,
            ),
        }
    )
    tolerance = evaluation.get("realised_loss_identity_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, int | float):
        raise ValueError("realised-loss tolerance must be numeric")
    if not np.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise ValueError("realised-loss tolerance must be positive and finite")
    if (
        float(dm.get("familywise_alpha", -1)) != 0.05
        or float(calibration.get("familywise_alpha", -1)) != 0.05
    ):
        raise ValueError("robustness family-wise alpha must remain 0.05")
    if calibration.get("confidence_levels") != [0.95, 0.99]:
        raise ValueError("robustness calibration levels must remain 95% and 99%")
    return sections


def _normalized_stock_panel(stock_simple_returns: pd.DataFrame) -> pd.DataFrame:
    panel = stock_simple_returns.copy()
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index, errors="raise"), name="date")
    if panel.empty or panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("stock return panel must have unique, increasing dates")
    if panel.columns.has_duplicates or not all(isinstance(column, str) for column in panel.columns):
        raise ValueError("stock return columns must be unique strings")
    return panel


def _normalized_primary_forecasts(primary_forecasts: pd.DataFrame) -> pd.DataFrame:
    if set(primary_forecasts.columns) != set(FORECAST_COLUMNS):
        raise ValueError("primary M0 forecast schema differs from the frozen schema")
    primary = primary_forecasts.loc[:, FORECAST_COLUMNS].copy()
    primary["date"] = pd.to_datetime(primary["date"], errors="raise")
    if set(primary["model_id"]) != {REFERENCE_MODEL}:
        raise ValueError("primary forecast input must contain only M0")
    if primary["date"].isna().any() or primary.duplicated(["model_id", "date"]).any():
        raise ValueError("primary M0 forecasts have invalid or duplicate dates")
    return primary.sort_values("date").reset_index(drop=True)


def _validate_historical_forecast_values(
    frame: pd.DataFrame, *, model_id: str, tolerance: float
) -> None:
    numeric_columns = (
        "var_95",
        "var_975",
        "var_99",
        "es_975",
        "realised_simple_return",
        "realised_loss",
    )
    if not np.isfinite(frame.loc[:, numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{model_id} forecasts contain non-finite risk values")
    if bool(
        (
            (frame["var_95"] > frame["var_975"])
            | (frame["var_975"] > frame["var_99"])
            | (frame["es_975"] < frame["var_975"])
            | (frame["es_975"] <= 0)
        ).any()
    ):
        raise ValueError(f"{model_id} forecasts have invalid VaR/ES ordering")
    if float(np.max(np.abs(frame["realised_loss"] + frame["realised_simple_return"]))) > tolerance:
        raise ValueError(f"{model_id} realised loss has the wrong sign")
    if set(frame["grouping_id"]) != {"none"}:
        raise ValueError(f"{model_id} must not use a grouping")
    if (
        frame["seed_components"].notna().any()
        or frame["copula_log_score"].notna().any()
        or bool((frame["margin_fallback_count"] != 0).any())
        or bool(frame["whole_vine_fallback"].any())
        or set(frame["forecast_status"]) != {"ok"}
    ):
        raise ValueError(f"{model_id} contains non-historical-simulation diagnostics")


def _annual_backcast_returns(
    panel: pd.DataFrame,
    tickers: Sequence[str],
    requested_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> pd.Series:
    backcast = panel.loc[
        (panel.index >= requested_start) & (panel.index <= evaluation_end), list(tickers)
    ]
    if backcast.empty:
        raise ValueError("current-composition backcast is empty")
    values = backcast.to_numpy(dtype=float)
    complete = np.isfinite(values).all(axis=1)
    allowed_initial = np.isnan(values).all(axis=1) & (backcast.index == panel.index.min())
    invalid = ~(complete | allowed_initial)
    if bool(invalid.any()):
        dates = [date.date().isoformat() for date in backcast.index[invalid]]
        raise ValueError(f"current-composition active returns are incomplete on: {dates[:10]}")
    returns = backcast.loc[complete].mean(axis=1).rename("simple_return")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("current-composition backcast contains non-finite returns")
    return returns


def build_current_composition_forecasts(
    stock_simple_returns: pd.DataFrame,
    active_schedule: Mapping[str, Any],
    primary_forecasts: pd.DataFrame,
    model_config: Mapping[str, Any],
    robustness_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build M0_CC forecasts with each evaluation year's active set backcast."""

    sections = validate_robustness_protocol(robustness_config, model_config)
    panel = _normalized_stock_panel(stock_simple_returns)
    primary = _normalized_primary_forecasts(primary_forecasts)
    active_by_year = annual_active_sets(active_schedule, panel.columns.tolist())
    forecast_config = sections["forecast"]
    evaluation = sections["evaluation"]
    start_year = int(evaluation["evaluation_start_year"])
    end_year = int(evaluation["evaluation_end_year"])
    expected_years = set(range(start_year, end_year + 1))
    if set(primary["date"].dt.year) != expected_years:
        raise ValueError("primary M0 forecasts do not cover the robustness years")
    if not expected_years.issubset(active_by_year):
        raise ValueError("active-universe schedule does not cover the robustness years")

    calendar_years = int(forecast_config["window_calendar_years"])
    minimum = int(forecast_config["minimum_training_observations"])
    confidence_levels = [float(value) for value in forecast_config["var_confidence_levels"]]
    es_confidence = float(forecast_config["es_confidence_level"])
    tolerance = float(evaluation["realised_loss_identity_tolerance"])
    _validate_historical_forecast_values(
        primary,
        model_id=REFERENCE_MODEL,
        tolerance=tolerance,
    )
    forecast_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    annual_diagnostics: list[dict[str, Any]] = []
    maximum_identity_error = 0.0

    for year in sorted(expected_years):
        reference = primary.loc[primary["date"].dt.year.eq(year)].sort_values("date")
        if reference.empty:
            raise ValueError(f"primary M0 has no forecasts for {year}")
        tickers = active_by_year[year]
        requested_start = pd.Timestamp(reference["date"].min()) - pd.DateOffset(
            years=calendar_years
        )
        annual_returns = _annual_backcast_returns(
            panel,
            tickers,
            requested_start,
            pd.Timestamp(reference["date"].max()),
        )
        evaluation_returns = annual_returns.reindex(pd.DatetimeIndex(reference["date"]))
        if evaluation_returns.isna().any():
            raise ValueError(f"current-composition evaluation dates are incomplete for {year}")
        reference_returns = reference["realised_simple_return"].to_numpy(dtype=float)
        identity_error = float(
            np.max(np.abs(evaluation_returns.to_numpy(dtype=float) - reference_returns))
        )
        maximum_identity_error = max(maximum_identity_error, identity_error)
        if identity_error > tolerance:
            raise ValueError(f"current-composition realised return differs from M0 in {year}")
        losses = -annual_returns
        annual_training_counts: list[int] = []
        for row in reference.itertuples(index=False):
            date = pd.Timestamp(row.date)
            window_start = date - pd.DateOffset(years=calendar_years)
            sample = losses.loc[(losses.index >= window_start) & (losses.index < date)]
            risks = {
                confidence: rolling_historical_var_es(
                    losses,
                    date,
                    confidence,
                    calendar_years=calendar_years,
                    minimum_observations=minimum,
                )
                for confidence in confidence_levels
            }
            annual_training_counts.append(len(sample))
            refit_id = f"{ROBUSTNESS_MODEL}:{date.date().isoformat()}:rolling_{calendar_years}y"
            realised_return = float(row.realised_simple_return)
            forecast_rows.append(
                {
                    "date": date,
                    "model_id": ROBUSTNESS_MODEL,
                    "grouping_id": "none",
                    "refit_id": refit_id,
                    "var_95": risks[0.95].var,
                    "var_975": risks[0.975].var,
                    "var_99": risks[0.99].var,
                    "es_975": risks[es_confidence].es,
                    "realised_simple_return": realised_return,
                    "realised_loss": -realised_return,
                    "seed_components": None,
                    "margin_fallback_count": 0,
                    "whole_vine_fallback": False,
                    "copula_log_score": None,
                    "forecast_status": "ok",
                }
            )
            window_rows.append(
                {
                    "refit_id": refit_id,
                    "forecast_date": date,
                    "requested_training_start": window_start,
                    "training_start": sample.index.min(),
                    "training_end": sample.index.max(),
                    "training_observations": len(sample),
                    "active_security_count": len(tickers),
                }
            )
        annual_diagnostics.append(
            {
                "year": year,
                "active_security_count": len(tickers),
                "forecast_count": len(reference),
                "minimum_training_observations": min(annual_training_counts),
                "maximum_training_observations": max(annual_training_counts),
                "maximum_realised_return_identity_error": identity_error,
            }
        )

    forecasts = pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS).sort_values("date")
    windows = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS).sort_values("forecast_date")
    issues = _construction_issues(forecasts, windows, len(primary), minimum, tolerance)
    audit = {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "forecast_count": len(forecasts),
        "window_count": len(windows),
        "evaluation_date_count": int(forecasts["date"].nunique()),
        "minimum_training_observations": int(windows["training_observations"].min()),
        "maximum_training_observations": int(windows["training_observations"].max()),
        "maximum_realised_return_identity_error": maximum_identity_error,
        "annual_diagnostics": annual_diagnostics,
    }
    return forecasts.reset_index(drop=True), windows.reset_index(drop=True), audit


def _construction_issues(
    forecasts: pd.DataFrame,
    windows: pd.DataFrame,
    expected_count: int,
    minimum: int,
    tolerance: float,
) -> list[str]:
    issues: list[str] = []
    if len(forecasts) != expected_count or forecasts["date"].nunique() != expected_count:
        issues.append("incomplete_current_composition_coverage")
    if forecasts.duplicated(["date", "model_id"]).any() or forecasts["refit_id"].duplicated().any():
        issues.append("duplicate_current_composition_forecast")
    if windows["refit_id"].duplicated().any() or set(windows["refit_id"]) != set(
        forecasts["refit_id"]
    ):
        issues.append("invalid_current_composition_window_binding")
    numeric = forecasts[
        ["var_95", "var_975", "var_99", "es_975", "realised_simple_return", "realised_loss"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        issues.append("nonfinite_current_composition_forecast")
    if bool(
        (
            (forecasts["var_95"] > forecasts["var_975"])
            | (forecasts["var_975"] > forecasts["var_99"])
        ).any()
    ):
        issues.append("nonmonotone_current_composition_var")
    if bool((forecasts["es_975"] < forecasts["var_975"]).any()):
        issues.append("current_composition_es_below_var")
    if (
        float(np.max(np.abs(forecasts["realised_loss"] + forecasts["realised_simple_return"])))
        > tolerance
    ):
        issues.append("current_composition_loss_identity_failed")
    if int(windows["training_observations"].min()) < minimum:
        issues.append("insufficient_current_composition_training")
    if bool((windows["training_end"] >= windows["forecast_date"]).any()):
        issues.append("current_composition_training_lookahead")
    return issues


def combine_historical_forecasts(
    primary_forecasts: pd.DataFrame,
    robustness_forecasts: pd.DataFrame,
    *,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate matched dates and realised losses before scoring M0 and M0_CC."""

    primary = _normalized_primary_forecasts(primary_forecasts)
    if set(robustness_forecasts.columns) != set(FORECAST_COLUMNS):
        raise ValueError("M0_CC forecast schema differs from the frozen schema")
    robustness = robustness_forecasts.loc[:, FORECAST_COLUMNS].copy()
    robustness["date"] = pd.to_datetime(robustness["date"], errors="raise")
    if set(robustness["model_id"]) != {ROBUSTNESS_MODEL}:
        raise ValueError("robustness forecasts must contain only M0_CC")
    if robustness["date"].isna().any() or robustness.duplicated(["model_id", "date"]).any():
        raise ValueError("M0_CC forecasts have invalid or duplicate dates")
    robustness = robustness.sort_values("date").reset_index(drop=True)
    _validate_historical_forecast_values(
        primary,
        model_id=REFERENCE_MODEL,
        tolerance=tolerance,
    )
    _validate_historical_forecast_values(
        robustness,
        model_id=ROBUSTNESS_MODEL,
        tolerance=tolerance,
    )
    if not pd.DatetimeIndex(primary["date"]).equals(pd.DatetimeIndex(robustness["date"])):
        raise ValueError("M0 and M0_CC forecast dates are not matched")
    realised_error = float(
        np.max(
            np.abs(
                primary["realised_loss"].to_numpy(dtype=float)
                - robustness["realised_loss"].to_numpy(dtype=float)
            )
        )
    )
    if realised_error > tolerance:
        raise ValueError("M0 and M0_CC realised losses differ")
    risk_columns = ("var_95", "var_975", "var_99", "es_975")
    forecast_differences = {
        column: np.abs(
            robustness[column].to_numpy(dtype=float) - primary[column].to_numpy(dtype=float)
        )
        for column in risk_columns
    }
    combined = pd.concat([primary, robustness], ignore_index=True)
    order = {model_id: position for position, model_id in enumerate(MODEL_ORDER)}
    combined["_order"] = combined["model_id"].map(order)
    combined = (
        combined.sort_values(["date", "_order"]).drop(columns="_order").reset_index(drop=True)
    )
    return combined, {
        "forecast_count": len(combined),
        "evaluation_date_count": len(primary),
        "model_date_counts": {
            model: int((combined["model_id"] == model).sum()) for model in MODEL_ORDER
        },
        "maximum_realised_loss_identity_error": realised_error,
        "realised_loss_identity_tolerance": tolerance,
        "forecast_difference_date_count": int(
            np.logical_or.reduce([values > 0 for values in forecast_differences.values()]).sum()
        ),
        "maximum_absolute_forecast_differences": {
            column: float(values.max()) for column, values in forecast_differences.items()
        },
    }


def calibration_results(
    scores: pd.DataFrame, calibration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Run the separate eight-test full-period calibration family."""

    results: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        model = scores.loc[scores["model_id"].eq(model_id)].sort_values("date")
        for raw_confidence in calibration["confidence_levels"]:
            confidence = float(raw_confidence)
            indicators = model[EXCEPTION_COLUMNS[confidence]].to_numpy(dtype=bool)
            coverage = kupiec_unconditional_coverage(indicators, confidence)
            independence = christoffersen_independence(indicators)
            results.append(
                {
                    "model_id": model_id,
                    "confidence": confidence,
                    "observations": coverage.observations,
                    "exceptions": coverage.exceptions,
                    "exception_rate": coverage.exception_rate,
                    "kupiec_p_value": coverage.p_value,
                    "christoffersen_independence_p_value": independence.p_value,
                    "conditional_coverage_p_value": christoffersen_conditional_coverage_p_value(
                        coverage, independence
                    ),
                }
            )
    raw = [
        value
        for row in results
        for value in (row["kupiec_p_value"], row["christoffersen_independence_p_value"])
    ]
    if len(raw) != int(calibration["family_size"]):
        raise AssertionError("current-composition calibration family has the wrong size")
    adjusted = iter(holm_adjust(raw).tolist())
    alpha = float(calibration["familywise_alpha"])
    for row in results:
        row["kupiec_holm_p_value"] = next(adjusted)
        row["kupiec_reject_after_holm"] = bool(row["kupiec_holm_p_value"] < alpha)
        row["christoffersen_independence_holm_p_value"] = next(adjusted)
        row["christoffersen_independence_reject_after_holm"] = bool(
            row["christoffersen_independence_holm_p_value"] < alpha
        )
    return results


def dm_results(scores: pd.DataFrame, dm: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run M0_CC-minus-M0 DM comparisons for the three primary scores."""

    primary = scores.loc[scores["model_id"].eq(REFERENCE_MODEL)].set_index("date")
    robustness = scores.loc[scores["model_id"].eq(ROBUSTNESS_MODEL)].set_index("date")
    if not primary.index.equals(robustness.index):
        raise ValueError("M0 and M0_CC score dates are not matched")
    results: list[dict[str, Any]] = []
    for score_column in PRIMARY_SCORES:
        differential = robustness[score_column].to_numpy() - primary[score_column].to_numpy()
        result = diebold_mariano(differential, hac_lag=int(dm["hac_lag"]))
        results.append(
            {
                "comparison_id": f"{ROBUSTNESS_MODEL}_minus_{REFERENCE_MODEL}:{score_column}",
                "score": score_column,
                "observations": result.observations,
                "mean_loss_difference": result.mean_difference,
                "dm_statistic": result.statistic,
                "p_value_two_sided": result.p_value_two_sided,
                "hac_lag": result.hac_lag,
            }
        )
    if len(results) != int(dm["family_size"]):
        raise AssertionError("current-composition DM family has the wrong size")
    adjusted = holm_adjust([row["p_value_two_sided"] for row in results])
    alpha = float(dm["familywise_alpha"])
    for row, adjusted_p in zip(results, adjusted, strict=True):
        difference = float(row["mean_loss_difference"])
        row["holm_p_value"] = float(adjusted_p)
        row["significant_after_holm"] = bool(adjusted_p < alpha)
        row["direction"] = (
            "current_composition_lower_loss"
            if difference < 0
            else "realised_history_lower_loss"
            if difference > 0
            else "equal_loss"
        )
        row["favours_current_composition_after_holm"] = bool(adjusted_p < alpha and difference < 0)
        row["favours_realised_history_after_holm"] = bool(adjusted_p < alpha and difference > 0)
    return results


def evaluate_forecasts(
    primary_forecasts: pd.DataFrame,
    robustness_forecasts: pd.DataFrame,
    robustness_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score and compare the matched historical-simulation variants."""

    evaluation = _mapping(robustness_config["evaluation"], "evaluation")
    combined, validation = combine_historical_forecasts(
        primary_forecasts,
        robustness_forecasts,
        tolerance=float(evaluation["realised_loss_identity_tolerance"]),
    )
    scores = score_forecasts(combined)
    calibration = calibration_results(
        scores, _mapping(robustness_config["calibration"], "calibration")
    )
    comparisons = dm_results(scores, _mapping(robustness_config["dm"], "dm"))
    summaries = []
    for model_id in MODEL_ORDER:
        model = scores.loc[scores["model_id"].eq(model_id)]
        summaries.append(
            {
                "model_id": model_id,
                "observations": len(model),
                "mean_quantile_loss_95": float(model["quantile_loss_95"].mean()),
                "mean_quantile_loss_99": float(model["quantile_loss_99"].mean()),
                "mean_fz0_975": float(model["fz0_975"].mean()),
                "exceptions_95": int(model["exception_95"].sum()),
                "exceptions_975": int(model["exception_975"].sum()),
                "exceptions_99": int(model["exception_99"].sum()),
                "adjusted_calibration_rejection_count": sum(
                    int(row["kupiec_reject_after_holm"])
                    + int(row["christoffersen_independence_reject_after_holm"])
                    for row in calibration
                    if row["model_id"] == model_id
                ),
            }
        )
    current_favoured = sum(row["favours_current_composition_after_holm"] for row in comparisons)
    history_favoured = sum(row["favours_realised_history_after_holm"] for row in comparisons)
    if current_favoured == len(comparisons):
        evidence = "current_composition_favoured"
    elif history_favoured == len(comparisons):
        evidence = "realised_history_favoured"
    elif current_favoured and not history_favoured:
        evidence = "partial_current_composition_favoured"
    elif history_favoured and not current_favoured:
        evidence = "partial_realised_history_favoured"
    elif current_favoured and history_favoured:
        evidence = "mixed"
    else:
        evidence = "no_holm_adjusted_difference"
    return scores, {
        "forecast_validation": validation,
        "model_summaries": summaries,
        "full_period_calibration": calibration,
        "diebold_mariano_comparisons": comparisons,
        "interpretation": {
            "analysis_role": "exploratory_robustness",
            "tests_favouring_current_composition_after_holm": current_favoured,
            "tests_favouring_realised_history_after_holm": history_favoured,
            "evidence": evidence,
            "core_H1_H2_H3_revised": False,
            "cross_portfolio_ranking_performed": False,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _require_primary_binding(
    audit: Mapping[str, Any], forecasts_path: Path, *, audit_name: str
) -> None:
    require_current_hash_records(audit, source_name=audit_name)
    if audit.get("status") != "pass":
        raise RuntimeError("primary historical-simulation gate is not passed")
    record = _mapping(
        _mapping(audit.get("outputs"), "outputs").get("historical_simulation_risk_forecasts"),
        "primary forecasts",
    )
    if record.get("sha256") != sha256_file(forecasts_path):
        raise RuntimeError("primary historical-simulation audit does not bind the forecasts")


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
        "--portfolio-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/portfolio_arithmetic.json",
    )
    parser.add_argument(
        "--primary-forecasts",
        type=Path,
        default=PROJECT_ROOT / "data/processed/historical_simulation_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--primary-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/historical_simulation_quality.json",
    )
    parser.add_argument(
        "--model-config", type=Path, default=PROJECT_ROOT / "config/model_config.toml"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/current_composition_hs_robustness.toml",
    )
    parser.add_argument(
        "--foundation-status",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_gate_status.json",
    )
    parser.add_argument(
        "--forecasts-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/current_composition_hs_robustness/forecasts.parquet",
    )
    parser.add_argument(
        "--windows-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/current_composition_hs_robustness/windows.parquet",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/current_composition_hs_robustness/daily_scores.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_composition_hs_robustness.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    with args.config.open("rb") as handle:
        robustness_config = tomllib.load(handle)
    validate_robustness_protocol(robustness_config, model_config)
    foundation = _load_json(args.foundation_status)
    require_current_hash_records(foundation, source_name=project_path(args.foundation_status))
    scope = reporting_scope(foundation)
    portfolio_audit = _load_json(args.portfolio_audit)
    require_current_hash_records(portfolio_audit, source_name=project_path(args.portfolio_audit))
    validate_portfolio_audit_binding(
        portfolio_audit,
        returns_path=args.returns,
        active_universe_path=args.active_universe,
    )
    primary_audit = _load_json(args.primary_audit)
    _require_primary_binding(
        primary_audit,
        args.primary_forecasts,
        audit_name=project_path(args.primary_audit),
    )
    primary = pd.read_parquet(args.primary_forecasts)
    forecasts, windows, construction = build_current_composition_forecasts(
        pd.read_parquet(args.returns),
        _load_json(args.active_universe),
        primary,
        model_config,
        robustness_config,
    )
    if construction["status"] != "pass":
        raise RuntimeError(f"current-composition construction failed: {construction['issues']}")
    scores, evaluation = evaluate_forecasts(primary, forecasts, robustness_config)
    write_parquet_atomic(args.forecasts_output, forecasts)
    write_parquet_atomic(args.windows_output, windows)
    write_parquet_atomic(args.scores_output, scores)
    audit = {
        "schema_version": 1,
        "gate_name": "current_composition_historical_simulation_robustness_v1",
        "status": "pass",
        "reporting_scope": scope,
        "analysis_role": robustness_config["analysis_role"],
        "method": {
            "reference_model": REFERENCE_MODEL,
            "robustness_model": ROBUSTNESS_MODEL,
            **dict(robustness_config["portfolio"]),
            **dict(robustness_config["forecast"]),
        },
        "construction_quality": construction,
        **evaluation,
        "inputs": {
            "stock_simple_returns": {
                "path": project_path(args.returns),
                "sha256": sha256_file(args.returns),
            },
            "active_universe": {
                "path": project_path(args.active_universe),
                "sha256": sha256_file(args.active_universe),
            },
            "portfolio_arithmetic": {
                "path": project_path(args.portfolio_audit),
                "sha256": sha256_file(args.portfolio_audit),
            },
            "primary_forecasts": {
                "path": project_path(args.primary_forecasts),
                "sha256": sha256_file(args.primary_forecasts),
            },
            "primary_historical_audit": {
                "path": project_path(args.primary_audit),
                "sha256": sha256_file(args.primary_audit),
            },
            "model_config": {
                "path": project_path(args.model_config),
                "sha256": sha256_file(args.model_config),
            },
            "robustness_config": {
                "path": project_path(args.config),
                "sha256": sha256_file(args.config),
            },
            "foundation_status": {
                "path": project_path(args.foundation_status),
                "sha256": sha256_file(args.foundation_status),
            },
        },
        "outputs": {
            "current_composition_forecasts": {
                "path": project_path(args.forecasts_output),
                "sha256": sha256_file(args.forecasts_output),
                "rows": len(forecasts),
            },
            "current_composition_windows": {
                "path": project_path(args.windows_output),
                "sha256": sha256_file(args.windows_output),
                "rows": len(windows),
            },
            "daily_scores": {
                "path": project_path(args.scores_output),
                "sha256": sha256_file(args.scores_output),
                "rows": len(scores),
            },
        },
    }
    write_json_atomic(args.audit_output, audit)
    print(
        f"current_composition_hs_robustness=pass forecasts={len(forecasts)} "
        f"evidence={evaluation['interpretation']['evidence']}"
    )
    print(f"Audit: {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
