#!/usr/bin/env python3
"""Fit and evaluate the frozen annually rebalanced group-balanced portfolios."""

from __future__ import annotations

import argparse
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
    from scripts.build_gaussian_copula import build_gaussian_outputs
    from scripts.build_group_balanced_portfolios import (
        ANALYSIS_ROLE,
        PortfolioSpec,
        load_json,
        validate_group_balanced_protocol,
    )
    from scripts.build_historical_simulation import FORECAST_COLUMNS, WINDOW_COLUMNS
    from scripts.build_marginal_models import build_marginal_outputs
    from scripts.build_vine_copula import build_vine_outputs
    from scripts.evaluate_risk_models import (
        EXCEPTION_COLUMNS,
        PRIMARY_SCORES,
        score_forecasts,
    )
    from scripts.ml_risk_common import artifact_record, require_same_seed_manifest
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
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
    from build_gaussian_copula import build_gaussian_outputs
    from build_group_balanced_portfolios import (
        ANALYSIS_ROLE,
        PortfolioSpec,
        load_json,
        validate_group_balanced_protocol,
    )
    from build_historical_simulation import FORECAST_COLUMNS, WINDOW_COLUMNS
    from build_marginal_models import build_marginal_outputs
    from build_vine_copula import build_vine_outputs
    from evaluate_risk_models import (
        EXCEPTION_COLUMNS,
        PRIMARY_SCORES,
        score_forecasts,
    )
    from ml_risk_common import artifact_record, require_same_seed_manifest
    from pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
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


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a table")
    return cast(Mapping[str, Any], value)


def _model_maps(
    specs: Sequence[PortfolioSpec],
) -> tuple[dict[str, PortfolioSpec], dict[str, str]]:
    by_model: dict[str, PortfolioSpec] = {}
    dependence: dict[str, str] = {}
    for spec in specs:
        for model_id, kind in (
            (spec.historical_model_id, "historical_simulation"),
            (spec.gaussian_model_id, "gaussian"),
            (spec.vine_model_id, "vine_tree_3"),
        ):
            if model_id in by_model:
                raise ValueError(f"duplicate group-balanced model ID: {model_id}")
            by_model[model_id] = spec
            dependence[model_id] = kind
    return by_model, dependence


def validate_risk_protocol(
    robustness_config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> tuple[PortfolioSpec, ...]:
    """Bind robustness inference choices to the frozen core configuration."""

    specs = validate_group_balanced_protocol(robustness_config)
    evaluation = _mapping(robustness_config.get("evaluation"), "evaluation")
    inference = _mapping(model_config.get("inference"), "model inference")
    portfolio = _mapping(model_config.get("portfolio"), "model portfolio")
    vine = _mapping(model_config.get("vine"), "model vine")
    expected = {
        "DM HAC lag": (evaluation["dm_hac_lag"], inference["dm_hac_lag"]),
        "familywise alpha": (
            evaluation["familywise_alpha"],
            inference["confirmatory_familywise_alpha"],
        ),
        "vine truncation": (
            robustness_config["models"]["vine_truncation_tree"],
            vine["primary_truncation_tree"],
        ),
        "secondary rebalancing": (portfolio["secondary_rebalancing"], "annual_buy_and_hold"),
    }
    mismatches = {
        name: {"observed": observed, "expected": required}
        for name, (observed, required) in expected.items()
        if observed != required
    }
    if mismatches:
        raise ValueError(f"group-balanced protocol differs from the core freeze: {mismatches}")
    return specs


def _historical_rows_for_portfolio(
    frame: pd.DataFrame,
    spec: PortfolioSpec,
    *,
    calendar_years: int,
    minimum: int,
    confidence_levels: Sequence[float],
    es_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forecast_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    selected = frame.loc[frame["portfolio_id"].eq(spec.portfolio_id)]
    for year, annual in selected.groupby("year", sort=True):
        annual = annual.sort_values("date")
        if annual.duplicated("date").any():
            raise ValueError(f"overlapping portfolio sample roles for {spec.portfolio_id}/{year}")
        losses = -annual.set_index("date")["simple_return"]
        evaluation = annual.loc[annual["sample_role"].eq("evaluation")]
        if evaluation.empty:
            raise ValueError(f"missing portfolio evaluation rows for {spec.portfolio_id}/{year}")
        for row in evaluation.itertuples(index=False):
            date = pd.Timestamp(row.date)
            requested_start = date - pd.DateOffset(years=calendar_years)
            sample = losses.loc[(losses.index >= requested_start) & (losses.index < date)]
            if len(sample) < minimum:
                raise ValueError(
                    f"{spec.portfolio_id}/{date.date()} has {len(sample)} observations"
                )
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
            realised = float(row.simple_return)
            refit_id = f"{spec.historical_model_id}:{date.date().isoformat()}:rolling_3y"
            forecast_rows.append(
                {
                    "date": date,
                    "model_id": spec.historical_model_id,
                    "grouping_id": spec.model_grouping_id,
                    "refit_id": refit_id,
                    "var_95": risks[0.95].var,
                    "var_975": risks[0.975].var,
                    "var_99": risks[0.99].var,
                    "es_975": risks[es_confidence].es,
                    "realised_simple_return": realised,
                    "realised_loss": -realised,
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
                    "requested_training_start": requested_start,
                    "training_start": sample.index.min(),
                    "training_end": sample.index.max(),
                    "training_observations": len(sample),
                    "active_security_count": int(row.active_security_count),
                }
            )
    return forecast_rows, window_rows


def _historical_quality_issues(
    forecasts: pd.DataFrame,
    windows: pd.DataFrame,
    expected_counts: Mapping[str, int],
    minimum: int,
) -> list[str]:
    issues: list[str] = []
    numeric = forecasts[["var_95", "var_975", "var_99", "es_975", "realised_loss"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        issues.append("nonfinite_historical_forecast")
    if bool(
        (
            (forecasts["var_95"] > forecasts["var_975"])
            | (forecasts["var_975"] > forecasts["var_99"])
        ).any()
    ):
        issues.append("nonmonotone_historical_var")
    if bool((forecasts["es_975"] < forecasts["var_975"]).any()):
        issues.append("historical_es_below_var")
    if int(windows["training_observations"].min()) < minimum:
        issues.append("insufficient_historical_training")
    observed_counts = {
        str(model_id): int(rows["date"].nunique())
        for model_id, rows in forecasts.groupby("model_id")
    }
    if observed_counts != expected_counts:
        issues.append("incomplete_historical_coverage")
    return issues


def build_historical_outputs(
    portfolio_returns: pd.DataFrame,
    specs: Sequence[PortfolioSpec],
    model_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build portfolio-specific rolling historical-simulation forecasts."""

    forecast = _mapping(model_config.get("forecast"), "forecast")
    historical = _mapping(model_config.get("historical_simulation"), "historical simulation")
    if (
        historical.get("quantile_method") != "inverted_empirical_cdf"
        or historical.get("es_boundary_method") != "fractional_order_statistic"
    ):
        raise ValueError("historical simulation differs from the frozen protocol")
    calendar_years = int(forecast["training_window_calendar_years"])
    minimum = int(forecast["minimum_training_observations"])
    confidence_levels = [float(value) for value in forecast["var_confidence_levels"]]
    es_confidence = float(forecast["es_confidence_level"])
    frame = portfolio_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    forecast_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for spec in specs:
        portfolio_forecasts, portfolio_windows = _historical_rows_for_portfolio(
            frame,
            spec,
            calendar_years=calendar_years,
            minimum=minimum,
            confidence_levels=confidence_levels,
            es_confidence=es_confidence,
        )
        forecast_rows.extend(portfolio_forecasts)
        window_rows.extend(portfolio_windows)
    forecasts = pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS).sort_values(
        ["date", "model_id"]
    )
    windows = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS).sort_values(
        ["forecast_date", "refit_id"]
    )
    model_counts = {
        str(model_id): int(rows["date"].nunique())
        for model_id, rows in forecasts.groupby("model_id")
    }
    expected_counts = {
        spec.historical_model_id: int(
            frame.loc[
                frame["portfolio_id"].eq(spec.portfolio_id) & frame["sample_role"].eq("evaluation"),
                "date",
            ].nunique()
        )
        for spec in specs
    }
    issues = _historical_quality_issues(forecasts, windows, expected_counts, minimum)
    audit = {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "forecast_count": len(forecasts),
        "window_count": len(windows),
        "model_date_counts": model_counts,
        "minimum_training_observations": int(windows["training_observations"].min()),
    }
    return forecasts.reset_index(drop=True), windows.reset_index(drop=True), audit


def _normalize_forecast_frame(source: pd.DataFrame) -> pd.DataFrame:
    if set(source.columns) != set(FORECAST_COLUMNS):
        raise ValueError("group-balanced forecast schema differs from the frozen schema")
    frame = source.loc[:, FORECAST_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _validate_forecast_values(combined: pd.DataFrame, tolerance: float) -> None:
    numeric = combined[
        [
            "var_95",
            "var_975",
            "var_99",
            "es_975",
            "realised_simple_return",
            "realised_loss",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("group-balanced forecasts contain non-finite risk values")
    nonmonotone = (combined["var_95"] > combined["var_975"]) | (
        combined["var_975"] > combined["var_99"]
    )
    if bool(nonmonotone.any()) or bool((combined["es_975"] < combined["var_975"]).any()):
        raise ValueError("group-balanced forecasts contain invalid VaR/ES ordering")
    sign_error = np.abs(combined["realised_loss"] + combined["realised_simple_return"])
    if float(sign_error.max()) > tolerance:
        raise ValueError("group-balanced realised loss has the wrong sign")


def _portfolio_forecast_identity(
    combined: pd.DataFrame,
    spec: PortfolioSpec,
    tolerance: float,
) -> tuple[dict[str, int], float]:
    model_ids = [
        spec.historical_model_id,
        spec.gaussian_model_id,
        spec.vine_model_id,
    ]
    selected = combined.loc[combined["model_id"].isin(model_ids)]
    realised = selected.pivot(index="date", columns="model_id", values="realised_loss")
    if set(realised.columns) != set(model_ids) or realised.isna().any().any():
        raise ValueError(f"incomplete realised loss panel for {spec.portfolio_id}")
    reference = realised[spec.historical_model_id]
    error = float(realised.sub(reference, axis="index").abs().to_numpy().max())
    if error > tolerance:
        raise ValueError(f"realised portfolio loss differs for {spec.portfolio_id}")
    counts = {model_id: int((selected["model_id"] == model_id).sum()) for model_id in model_ids}
    if len(set(counts.values())) != 1:
        raise ValueError(f"forecast coverage differs for {spec.portfolio_id}")
    return counts, error


def combine_forecasts(
    historical: pd.DataFrame,
    gaussian: pd.DataFrame,
    vine: pd.DataFrame,
    specs: Sequence[PortfolioSpec],
    *,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate all six forecasts with realised-loss identity within portfolio."""

    by_model, _ = _model_maps(specs)
    expected_models = set(by_model)
    frames = [_normalize_forecast_frame(source) for source in (historical, gaussian, vine)]
    records = [record for frame in frames for record in frame.to_dict(orient="records")]
    combined = pd.DataFrame.from_records(records, columns=FORECAST_COLUMNS)
    if set(combined["model_id"]) != expected_models:
        raise ValueError("group-balanced forecasts do not contain the frozen model set")
    if combined.duplicated(["date", "model_id"]).any():
        raise ValueError("group-balanced forecasts contain duplicate model dates")
    _validate_forecast_values(combined, tolerance)
    maximum_identity = 0.0
    portfolio_counts: dict[str, dict[str, int]] = {}
    for spec in specs:
        counts, error = _portfolio_forecast_identity(combined, spec, tolerance)
        maximum_identity = max(maximum_identity, error)
        portfolio_counts[spec.portfolio_id] = counts
    order = {model_id: index for index, model_id in enumerate(by_model)}
    combined["_order"] = combined["model_id"].map(order)
    combined = combined.sort_values(["date", "_order"]).drop(columns="_order")
    return combined.reset_index(drop=True), {
        "forecast_count": len(combined),
        "evaluation_date_count": int(combined["date"].nunique()),
        "portfolio_model_date_counts": portfolio_counts,
        "maximum_within_portfolio_realised_loss_identity_error": maximum_identity,
        "identity_tolerance": tolerance,
    }


def calibration_results(
    scores: pd.DataFrame,
    specs: Sequence[PortfolioSpec],
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run one frozen 24-test Holm family across all robustness models."""

    by_model, _ = _model_maps(specs)
    results: list[dict[str, Any]] = []
    for model_id, spec in by_model.items():
        model = scores.loc[scores["model_id"].eq(model_id)].sort_values("date")
        for raw_confidence in evaluation["calibration_confidence_levels"]:
            confidence = float(raw_confidence)
            indicators = model[EXCEPTION_COLUMNS[confidence]].to_numpy(dtype=bool)
            coverage = kupiec_unconditional_coverage(indicators, confidence)
            independence = christoffersen_independence(indicators)
            results.append(
                {
                    "model_id": model_id,
                    "portfolio_id": spec.portfolio_id,
                    "confidence": confidence,
                    "observations": coverage.observations,
                    "exceptions": coverage.exceptions,
                    "exception_rate": coverage.exception_rate,
                    "kupiec_likelihood_ratio": coverage.likelihood_ratio,
                    "kupiec_p_value": coverage.p_value,
                    "christoffersen_independence_likelihood_ratio": independence.likelihood_ratio,
                    "christoffersen_independence_p_value": independence.p_value,
                    "conditional_coverage_p_value": christoffersen_conditional_coverage_p_value(
                        coverage, independence
                    ),
                }
            )
    raw_p_values = [
        value
        for row in results
        for value in (row["kupiec_p_value"], row["christoffersen_independence_p_value"])
    ]
    if len(raw_p_values) != int(evaluation["calibration_family_size"]):
        raise AssertionError("group-balanced calibration family has the wrong size")
    adjusted = iter(holm_adjust(raw_p_values).tolist())
    alpha = float(evaluation["familywise_alpha"])
    for row in results:
        row["kupiec_holm_p_value"] = next(adjusted)
        row["kupiec_reject_after_holm"] = bool(row["kupiec_holm_p_value"] < alpha)
        row["christoffersen_independence_holm_p_value"] = next(adjusted)
        row["christoffersen_independence_reject_after_holm"] = bool(
            row["christoffersen_independence_holm_p_value"] < alpha
        )
    return results


def dm_results(
    scores: pd.DataFrame,
    specs: Sequence[PortfolioSpec],
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare vine minus Gaussian loss only within the same portfolio target."""

    results: list[dict[str, Any]] = []
    for spec in specs:
        vine = scores.loc[scores["model_id"].eq(spec.vine_model_id)].set_index("date")
        gaussian = scores.loc[scores["model_id"].eq(spec.gaussian_model_id)].set_index("date")
        if not vine.index.equals(gaussian.index):
            raise ValueError(f"DM dates differ for {spec.portfolio_id}")
        for score_column in evaluation["loss_scores"]:
            result = diebold_mariano(
                vine[score_column].to_numpy() - gaussian[score_column].to_numpy(),
                hac_lag=int(evaluation["dm_hac_lag"]),
            )
            results.append(
                {
                    "comparison_id": (
                        f"{spec.vine_model_id}_minus_{spec.gaussian_model_id}:{score_column}"
                    ),
                    "portfolio_id": spec.portfolio_id,
                    "vine_model": spec.vine_model_id,
                    "gaussian_model": spec.gaussian_model_id,
                    "score": score_column,
                    "observations": result.observations,
                    "mean_loss_difference": result.mean_difference,
                    "dm_statistic": result.statistic,
                    "p_value_two_sided": result.p_value_two_sided,
                    "hac_lag": result.hac_lag,
                }
            )
    if len(results) != int(evaluation["dm_family_size"]):
        raise AssertionError("group-balanced DM family has the wrong size")
    adjusted = holm_adjust([row["p_value_two_sided"] for row in results])
    alpha = float(evaluation["familywise_alpha"])
    for row, adjusted_p in zip(results, adjusted, strict=True):
        difference = float(row["mean_loss_difference"])
        row["holm_p_value"] = float(adjusted_p)
        row["significant_after_holm"] = bool(adjusted_p < alpha)
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
    scores: pd.DataFrame,
    specs: Sequence[PortfolioSpec],
    calibration: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank historical, Gaussian, and vine models separately within each portfolio."""

    by_model, dependence = _model_maps(specs)
    rows: list[dict[str, Any]] = []
    for model_id, spec in by_model.items():
        model = scores.loc[scores["model_id"].eq(model_id)]
        row: dict[str, Any] = {
            "model_id": model_id,
            "portfolio_id": spec.portfolio_id,
            "grouping_id": spec.model_grouping_id,
            "dependence": dependence[model_id],
            "observations": len(model),
            "mean_quantile_loss_95": float(model["quantile_loss_95"].mean()),
            "mean_quantile_loss_99": float(model["quantile_loss_99"].mean()),
            "mean_fz0_975": float(model["fz0_975"].mean()),
            "mean_copula_log_score": (
                None
                if dependence[model_id] == "historical_simulation"
                else float(model["copula_log_score"].mean())
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
    result = pd.DataFrame(rows)
    rank_columns: list[str] = []
    for score in PRIMARY_SCORES:
        rank_column = f"rank_{score}"
        result[rank_column] = result.groupby("portfolio_id")[f"mean_{score}"].rank(
            method="average", ascending=True
        )
        rank_columns.append(rank_column)
    result["average_primary_score_rank"] = result[rank_columns].mean(axis=1)
    result["within_portfolio_overall_rank"] = result.groupby("portfolio_id")[
        "average_primary_score_rank"
    ].rank(method="average", ascending=True)
    return cast(list[dict[str, Any]], result.to_dict(orient="records"))


def evaluate_forecasts(
    historical: pd.DataFrame,
    gaussian: pd.DataFrame,
    vine: pd.DataFrame,
    specs: Sequence[PortfolioSpec],
    robustness_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    vine_audits: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score and evaluate the two distinct robustness portfolio targets."""

    tolerance = float(model_config["portfolio"]["reconstruction_tolerance"])
    combined, validation = combine_forecasts(historical, gaussian, vine, specs, tolerance=tolerance)
    scores = score_forecasts(combined)
    evaluation = _mapping(robustness_config.get("evaluation"), "evaluation")
    calibration = calibration_results(scores, specs, evaluation)
    dm = dm_results(scores, specs, evaluation)
    summaries = model_summaries(scores, specs, calibration)
    vine_favouring = sum(bool(row["favours_vine_after_holm"]) for row in dm)
    gaussian_favouring = sum(bool(row["favours_gaussian_after_holm"]) for row in dm)
    audit = {
        "forecast_validation": validation,
        "model_summaries": summaries,
        "full_period_calibration": calibration,
        "diebold_mariano_comparisons": dm,
        "portfolio_conclusions": [
            {
                "portfolio_id": spec.portfolio_id,
                "best_average_score_rank_model": min(
                    (row for row in summaries if row["portfolio_id"] == spec.portfolio_id),
                    key=lambda row: (row["average_primary_score_rank"], row["model_id"]),
                )["model_id"],
                "vine_quality_eligible": bool(
                    vine_audits[spec.portfolio_id]["eligible_to_be_declared_best"]
                ),
                "adjusted_dm_tests_favouring_vine": sum(
                    bool(row["favours_vine_after_holm"])
                    for row in dm
                    if row["portfolio_id"] == spec.portfolio_id
                ),
                "adjusted_dm_tests_favouring_gaussian": sum(
                    bool(row["favours_gaussian_after_holm"])
                    for row in dm
                    if row["portfolio_id"] == spec.portfolio_id
                ),
            }
            for spec in specs
        ],
        "interpretation": {
            "analysis_role": ANALYSIS_ROLE,
            "tests_favouring_vine_after_holm": vine_favouring,
            "tests_favouring_gaussian_after_holm": gaussian_favouring,
            "evidence": (
                "vine_favoured"
                if vine_favouring and not gaussian_favouring
                else "gaussian_favoured"
                if gaussian_favouring and not vine_favouring
                else "mixed"
                if vine_favouring and gaussian_favouring
                else "no_holm_adjusted_difference"
            ),
            "core_H1_H2_H3_revised": False,
            "cross_portfolio_ranking_performed": False,
        },
    }
    return scores, audit


def _filter_grouping(frame: pd.DataFrame, grouping_id: str) -> pd.DataFrame:
    selected = frame.loc[frame["grouping_id"].eq(grouping_id)].copy()
    if selected.empty:
        raise ValueError(f"no rows for grouping {grouping_id}")
    return selected


def _compact(audit: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: audit[field] for field in fields if field in audit}


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
        "--group-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/group_balanced_group_returns.parquet",
    )
    parser.add_argument(
        "--portfolio-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/group_balanced_portfolio_returns.parquet",
    )
    parser.add_argument(
        "--construction-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/group_balanced_portfolio_construction.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/group_balanced_robustness.toml",
    )
    parser.add_argument(
        "--model-config", type=Path, default=PROJECT_ROOT / "config/model_config.toml"
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
        "--foundation-status",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_gate_status.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "data/processed/group_balanced_robustness",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/group_balanced_robustness.json",
    )
    parser.add_argument("--progress-every", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    with args.config.open("rb") as handle:
        robustness_config = tomllib.load(handle)
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    specs = validate_risk_protocol(robustness_config, model_config)
    foundation = load_json(args.foundation_status)
    require_current_hash_records(foundation, source_name=project_path(args.foundation_status))
    scope = reporting_scope(foundation)
    construction_audit = load_json(args.construction_audit)
    _require_binding(
        construction_audit,
        "outputs",
        "group_returns",
        args.group_returns,
        project_path(args.construction_audit),
    )
    _require_binding(
        construction_audit,
        "outputs",
        "portfolio_returns",
        args.portfolio_returns,
        project_path(args.construction_audit),
    )
    if (
        construction_audit.get("reporting_scope") != scope
        or construction_audit.get("analysis_role") != ANALYSIS_ROLE
    ):
        raise RuntimeError("construction audit scope or role differs")
    primary_gaussian_audit = load_json(args.primary_gaussian_audit)
    _require_binding(
        primary_gaussian_audit,
        "outputs",
        "simulation_seed_manifest",
        args.primary_seed_manifest,
        project_path(args.primary_gaussian_audit),
    )
    primary_seed_manifest = load_json(args.primary_seed_manifest)
    group_returns = pd.read_parquet(args.group_returns)
    portfolio_returns = pd.read_parquet(args.portfolio_returns)

    historical, historical_windows, historical_audit = build_historical_outputs(
        portfolio_returns, specs, model_config
    )
    marginal_refits, daily_margins, training_pits, marginal_audit = build_marginal_outputs(
        group_returns,
        model_config,
        universe_variant="security_primary",
        progress_every=args.progress_every,
        portfolio_weight_rule="annual_equal_group_buy_and_hold",
    )
    if historical_audit["status"] != "pass" or marginal_audit["status"] != "pass":
        raise RuntimeError("group-balanced historical or marginal quality failed")

    gaussian_refit_frames: list[pd.DataFrame] = []
    gaussian_forecast_frames: list[pd.DataFrame] = []
    gaussian_audits: dict[str, Mapping[str, Any]] = {}
    for spec in specs:
        grouping = spec.model_grouping_id
        refits, forecasts, seed_manifest, audit = build_gaussian_outputs(
            _filter_grouping(training_pits, grouping),
            _filter_grouping(marginal_refits, grouping),
            _filter_grouping(daily_margins, grouping),
            _filter_grouping(group_returns, grouping),
            model_config,
            universe_variant="security_primary",
            progress_every=args.progress_every,
            model_by_grouping={grouping: (spec.gaussian_model_id, grouping)},
        )
        require_same_seed_manifest(seed_manifest, primary_seed_manifest)
        if audit["status"] != "pass":
            raise RuntimeError(f"Gaussian quality failed for {spec.portfolio_id}")
        gaussian_refit_frames.append(refits)
        gaussian_forecast_frames.append(forecasts)
        gaussian_audits[spec.portfolio_id] = audit
    gaussian_refits = pd.concat(gaussian_refit_frames, ignore_index=True).sort_values(
        ["year", "month", "model_id"]
    )
    gaussian_forecasts = pd.concat(gaussian_forecast_frames, ignore_index=True).sort_values(
        ["date", "model_id"]
    )

    vine_refit_frames: list[pd.DataFrame] = []
    vine_forecast_frames: list[pd.DataFrame] = []
    vine_audits: dict[str, Mapping[str, Any]] = {}
    for spec in specs:
        grouping = spec.model_grouping_id
        refits, forecasts, audit = build_vine_outputs(
            _filter_grouping(training_pits, grouping),
            _filter_grouping(marginal_refits, grouping),
            _filter_grouping(daily_margins, grouping),
            _filter_grouping(group_returns, grouping),
            gaussian_refits.loc[gaussian_refits["model_id"].eq(spec.gaussian_model_id)].copy(),
            primary_seed_manifest,
            model_config,
            universe_variant="security_primary",
            truncation_level=3,
            progress_every=args.progress_every,
            model_by_grouping={grouping: (spec.vine_model_id, grouping, spec.gaussian_model_id)},
            analysis_role=ANALYSIS_ROLE,
        )
        if audit["status"] != "pass":
            raise RuntimeError(f"vine quality failed for {spec.portfolio_id}")
        vine_refit_frames.append(refits)
        vine_forecast_frames.append(forecasts)
        vine_audits[spec.portfolio_id] = audit
    vine_refits = pd.concat(vine_refit_frames, ignore_index=True).sort_values(
        ["year", "month", "model_id"]
    )
    vine_forecasts = pd.concat(vine_forecast_frames, ignore_index=True).sort_values(
        ["date", "model_id"]
    )
    scores, evaluation_audit = evaluate_forecasts(
        historical,
        gaussian_forecasts,
        vine_forecasts,
        specs,
        robustness_config,
        model_config,
        vine_audits,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "historical_forecasts": historical,
        "historical_windows": historical_windows,
        "marginal_refits": marginal_refits,
        "marginal_daily": daily_margins,
        "training_pits": training_pits,
        "gaussian_refits": gaussian_refits,
        "gaussian_forecasts": gaussian_forecasts,
        "vine_refits": vine_refits,
        "vine_forecasts": vine_forecasts,
        "daily_scores": scores,
    }
    output_paths = {name: args.output_directory / f"{name}.parquet" for name in output_frames}
    for name, frame in output_frames.items():
        write_parquet_atomic(output_paths[name], frame)
    audit = {
        "schema_version": 1,
        "gate_name": "group_balanced_robustness_v1",
        "analysis_role": ANALYSIS_ROLE,
        "status": "pass",
        "reporting_scope": scope,
        "issues": [],
        "method": {
            **dict(robustness_config["models"]),
            "portfolio_weight_rule": "annual_equal_group_buy_and_hold",
            "inference": dict(robustness_config["evaluation"]),
        },
        "quality": {
            "historical": historical_audit,
            "marginal": _compact(
                marginal_audit,
                (
                    "status",
                    "refit_count",
                    "selected_method_counts",
                    "fallback_fit_count",
                    "ewma_fit_count",
                    "ewma_fit_fraction",
                    "portfolio_weight_rule",
                    "issues",
                ),
            ),
            "gaussian": {
                portfolio_id: _compact(
                    stage_audit,
                    (
                        "status",
                        "model_ids",
                        "refit_count",
                        "repaired_refit_count",
                        "maximum_condition_number",
                        "maximum_realised_portfolio_identity_error",
                        "issues",
                    ),
                )
                for portfolio_id, stage_audit in gaussian_audits.items()
            },
            "vine": {
                portfolio_id: _compact(
                    stage_audit,
                    (
                        "status",
                        "model_ids",
                        "refit_count",
                        "failed_pair_count",
                        "whole_vine_fallback_refit_count",
                        "whole_vine_fallback_date_fraction",
                        "eligible_to_be_declared_best",
                        "issues",
                    ),
                )
                for portfolio_id, stage_audit in vine_audits.items()
            },
        },
        "evaluation": evaluation_audit,
        "inputs": {
            "group_returns": artifact_record(args.group_returns),
            "portfolio_returns": artifact_record(args.portfolio_returns),
            "construction_audit": artifact_record(args.construction_audit),
            "group_balanced_config": artifact_record(args.config),
            "model_config": artifact_record(args.model_config),
            "primary_seed_manifest": artifact_record(args.primary_seed_manifest),
            "primary_gaussian_audit": artifact_record(args.primary_gaussian_audit),
            "foundation_status": artifact_record(args.foundation_status),
        },
        "outputs": {
            name: artifact_record(path, rows=len(output_frames[name]))
            for name, path in output_paths.items()
        },
    }
    write_json_atomic(args.audit_output, audit)
    print(
        f"group_balanced_robustness=pass scores={len(scores)} "
        f"marginal_refits={len(marginal_refits)} gaussian_refits={len(gaussian_refits)} "
        f"vine_refits={len(vine_refits)} evidence={evaluation_audit['interpretation']['evidence']}"
    )
    print(f"Audit: {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
