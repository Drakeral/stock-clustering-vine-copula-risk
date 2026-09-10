#!/usr/bin/env python3
"""Build the M0 rolling historical-simulation portfolio-risk benchmark."""

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
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        sha256_file,
        write_json_atomic,
        write_parquet_atomic,
    )
    from scripts.research_methods import rolling_historical_var_es
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        sha256_file,
        write_json_atomic,
        write_parquet_atomic,
    )
    from research_methods import rolling_historical_var_es


FORECAST_COLUMNS = [
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
]
WINDOW_COLUMNS = [
    "refit_id",
    "forecast_date",
    "requested_training_start",
    "training_start",
    "training_end",
    "training_observations",
    "active_security_count",
]
ACTIVE_UNIVERSE_RULE = (
    "Freeze the initial eligible universe and remove terminal securities at the next annual "
    "rebalance; introduce no later index additions."
)


def _positive_integer(section: Mapping[str, Any], key: str, section_name: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{section_name}.{key} must be a positive integer")
    return value


def validate_historical_protocol(
    model_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate the frozen M0 window, estimators, portfolio, and evaluation span."""

    forecast = model_config.get("forecast")
    historical = model_config.get("historical_simulation")
    portfolio = model_config.get("portfolio")
    clustering = model_config.get("clustering")
    if not all(
        isinstance(section, Mapping) for section in (forecast, historical, portfolio, clustering)
    ):
        raise TypeError("model configuration is missing an M0 protocol table")
    forecast = cast(Mapping[str, Any], forecast)
    historical = cast(Mapping[str, Any], historical)
    portfolio = cast(Mapping[str, Any], portfolio)
    clustering = cast(Mapping[str, Any], clustering)

    expected = {
        "historical_simulation.quantile_method": (
            historical.get("quantile_method"),
            "inverted_empirical_cdf",
        ),
        "historical_simulation.es_boundary_method": (
            historical.get("es_boundary_method"),
            "fractional_order_statistic",
        ),
        "portfolio.primary_return_scale": (portfolio.get("primary_return_scale"), "simple"),
        "portfolio.primary_rebalancing": (portfolio.get("primary_rebalancing"), "daily"),
        "portfolio.membership_rebalancing": (
            portfolio.get("membership_rebalancing"),
            "annual",
        ),
        "portfolio.realised_loss_definition": (
            portfolio.get("realised_loss_definition"),
            "negative_simple_portfolio_return",
        ),
    }
    mismatches = {
        key: {"observed": observed, "expected": required}
        for key, (observed, required) in expected.items()
        if observed != required
    }
    if mismatches:
        raise ValueError(f"unsupported historical-simulation protocol: {mismatches}")

    confidence_levels = [float(value) for value in forecast.get("var_confidence_levels", [])]
    if confidence_levels != [0.95, 0.975, 0.99]:
        raise ValueError("forecast confidence levels differ from the frozen protocol")
    if float(forecast.get("es_confidence_level", float("nan"))) != 0.975:
        raise ValueError("forecast ES confidence level must be 0.975")
    if _positive_integer(forecast, "horizon_trading_days", "forecast") != 1:
        raise ValueError("only one-day historical-simulation forecasts are supported")
    if _positive_integer(forecast, "training_window_calendar_years", "forecast") != 3:
        raise ValueError("historical simulation requires a three-calendar-year window")
    if _positive_integer(forecast, "minimum_training_observations", "forecast") != 700:
        raise ValueError("historical simulation requires at least 700 observations")
    start_year = _positive_integer(clustering, "evaluation_start_year", "clustering")
    end_year = _positive_integer(clustering, "evaluation_end_year", "clustering")
    if start_year != 2020 or end_year != 2025:
        raise ValueError("M0 evaluation span must remain 2020 through 2025")
    return forecast, historical, portfolio, clustering


def annual_active_sets(
    active_schedule: Mapping[str, Any],
    available_tickers: Sequence[str],
) -> dict[int, tuple[str, ...]]:
    """Validate and normalize annual active security sets."""

    if active_schedule.get("schema_version") != 1:
        raise ValueError("unsupported active-universe schema version")
    if active_schedule.get("rule") != ACTIVE_UNIVERSE_RULE:
        raise ValueError("active-universe schedule uses an unsupported lifecycle rule")
    rows = active_schedule.get("years")
    if not isinstance(rows, list) or not rows:
        raise ValueError("active-universe schedule must contain annual rows")
    available = {str(ticker) for ticker in available_tickers}
    result: dict[int, tuple[str, ...]] = {}
    previous_active: set[str] | None = None
    previous_year: int | None = None
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("active-universe row must be an object")
        year = raw.get("year")
        tickers = raw.get("active_tickers")
        count = raw.get("security_count")
        rebalance_date = raw.get("rebalance_date")
        if isinstance(year, bool) or not isinstance(year, int):
            raise ValueError("active-universe year must be an integer")
        if year in result:
            raise ValueError(f"duplicate active-universe year: {year}")
        if previous_year is not None and year != previous_year + 1:
            raise ValueError("active-universe years must be consecutive and increasing")
        if not isinstance(tickers, list) or not tickers:
            raise ValueError(f"active-universe year {year} has no tickers")
        normalized = tuple(str(ticker) for ticker in tickers)
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"active-universe year {year} contains duplicate tickers")
        if isinstance(count, bool) or not isinstance(count, int) or count != len(normalized):
            raise ValueError(f"active-universe year {year} has an inconsistent security count")
        missing = sorted(set(normalized) - available)
        if missing:
            raise ValueError(f"active-universe year {year} is missing panel columns: {missing}")
        current_active = set(normalized)
        if previous_active is not None and not current_active.issubset(previous_active):
            additions = sorted(current_active - previous_active)
            raise ValueError(f"active-universe year {year} introduces later additions: {additions}")
        try:
            rebalance = pd.Timestamp(rebalance_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"active-universe year {year} has an invalid rebalance date") from exc
        if rebalance.year != year:
            raise ValueError(f"active-universe year {year} has an out-of-year rebalance date")
        result[year] = normalized
        previous_active = current_active
        previous_year = year
    return result


def daily_portfolio_simple_returns(
    stock_simple_returns: pd.DataFrame,
    active_schedule: Mapping[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Construct the canonical daily-rebalanced portfolio and annual counts."""

    if stock_simple_returns.empty:
        raise ValueError("stock simple-return panel must not be empty")
    panel = stock_simple_returns.copy()
    try:
        panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index), name="date")
    except (TypeError, ValueError) as exc:
        raise ValueError("stock simple-return panel has an invalid date index") from exc
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("stock simple-return dates must be unique and increasing")
    if panel.columns.has_duplicates:
        raise ValueError("stock simple-return panel has duplicate ticker columns")
    active_by_year = annual_active_sets(active_schedule, panel.columns.astype(str).tolist())
    if set(panel.index.year) != set(active_by_year):
        raise ValueError("stock return years and active-universe years do not agree")

    portfolio_parts: list[pd.Series] = []
    count_parts: list[pd.Series] = []
    earliest_date = panel.index.min()
    schedule_rows = cast(list[Mapping[str, Any]], active_schedule["years"])
    rebalance_by_year = {
        int(row["year"]): pd.Timestamp(row["rebalance_date"]) for row in schedule_rows
    }
    for year in sorted(active_by_year):
        tickers = list(active_by_year[year])
        annual = panel.loc[panel.index.year == year, tickers]
        if annual.empty or rebalance_by_year[year] != annual.index.min():
            raise ValueError(f"active-universe year {year} does not begin on its first session")
        values = annual.to_numpy(dtype=float)
        complete = np.isfinite(values).all(axis=1)
        allowed_initial = np.isnan(values).all(axis=1) & (annual.index == earliest_date)
        if bool((~(complete | allowed_initial)).any()):
            dates = [
                date.date().isoformat() for date in annual.index[~(complete | allowed_initial)]
            ]
            raise ValueError(f"active security returns are incomplete on: {dates[:10]}")
        defined = annual.loc[complete]
        portfolio_parts.append(defined.mean(axis=1))
        count_parts.append(pd.Series(len(tickers), index=defined.index, dtype="int64"))
    portfolio = pd.concat(portfolio_parts).sort_index().rename("simple_return")
    counts = pd.concat(count_parts).sort_index().rename("active_security_count")
    if portfolio.index.has_duplicates or not portfolio.index.is_monotonic_increasing:
        raise AssertionError("constructed portfolio dates are not unique and increasing")
    if not np.isfinite(portfolio.to_numpy(dtype=float)).all():
        raise ValueError("constructed portfolio contains non-finite returns")
    return portfolio, counts


def build_historical_simulation_outputs(
    stock_simple_returns: pd.DataFrame,
    active_schedule: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build M0 forecasts, daily window records, and computational audit fields."""

    forecast, _, _, clustering = validate_historical_protocol(model_config)
    portfolio, active_counts = daily_portfolio_simple_returns(stock_simple_returns, active_schedule)
    calendar_years = int(forecast["training_window_calendar_years"])
    minimum_observations = int(forecast["minimum_training_observations"])
    confidence_levels = [float(value) for value in forecast["var_confidence_levels"]]
    es_confidence = float(forecast["es_confidence_level"])
    start_year = int(clustering["evaluation_start_year"])
    end_year = int(clustering["evaluation_end_year"])
    losses = -portfolio
    evaluation_dates = portfolio.index[
        (portfolio.index.year >= start_year) & (portfolio.index.year <= end_year)
    ]
    if evaluation_dates.empty:
        raise ValueError("portfolio panel contains no M0 evaluation dates")

    forecast_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for date in evaluation_dates:
        requested_start = date - pd.DateOffset(years=calendar_years)
        sample = losses.loc[(losses.index >= requested_start) & (losses.index < date)]
        risks = {
            confidence: rolling_historical_var_es(
                losses,
                date,
                confidence,
                calendar_years=calendar_years,
                minimum_observations=minimum_observations,
            )
            for confidence in confidence_levels
        }
        observation_counts = {risk.observations for risk in risks.values()}
        if observation_counts != {len(sample)}:
            raise AssertionError("historical-risk observation counts are inconsistent")
        refit_id = f"M0:{date.date().isoformat()}:rolling_{calendar_years}y"
        realised_return = float(portfolio.loc[date])
        forecast_rows.append(
            {
                "date": date,
                "model_id": "M0",
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
                "requested_training_start": requested_start,
                "training_start": sample.index.min(),
                "training_end": sample.index.max(),
                "training_observations": len(sample),
                "active_security_count": int(active_counts.loc[date]),
            }
        )

    forecasts = pd.DataFrame(forecast_rows, columns=FORECAST_COLUMNS)
    windows = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS)
    issues: list[str] = []
    if forecasts.duplicated(["date", "model_id"]).any() or forecasts["refit_id"].duplicated().any():
        issues.append("duplicate_historical_forecast")
    if windows["refit_id"].duplicated().any() or set(windows["refit_id"]) != set(
        forecasts["refit_id"]
    ):
        issues.append("invalid_historical_window_binding")
    numeric = forecasts[
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
    maximum_loss_identity_error = float(
        np.max(np.abs(forecasts["realised_loss"] + forecasts["realised_simple_return"]))
    )
    if maximum_loss_identity_error > 1e-15:
        issues.append("historical_realised_loss_identity_failed")
    if int(windows["training_observations"].min()) < minimum_observations:
        issues.append("historical_minimum_training_observations_failed")
    if bool((windows["training_end"] >= windows["forecast_date"]).any()):
        issues.append("historical_training_lookahead")
    expected_dates = portfolio.loc[
        (portfolio.index.year >= start_year) & (portfolio.index.year <= end_year)
    ].index
    if not forecasts["date"].equals(pd.Series(expected_dates, name="date")):
        issues.append("incomplete_historical_forecast_coverage")
    annual_count_variants = windows.groupby(windows["forecast_date"].dt.year)[
        "active_security_count"
    ].nunique()
    if bool((annual_count_variants != 1).any()):
        issues.append("within_year_active_security_count_changed")
    annual_forecast_counts = {
        str(year): int(count)
        for year, count in forecasts.groupby(forecasts["date"].dt.year).size().items()
    }
    annual_active_counts = {
        str(year): int(rows["active_security_count"].iloc[0])
        for year, rows in windows.groupby(windows["forecast_date"].dt.year)
    }

    audit = {
        "schema_version": 1,
        "gate_name": "historical_simulation_quality_v1",
        "status": "pass" if not issues else "fail",
        "model_id": "M0",
        "grouping_id": "none",
        "forecast_count": len(forecasts),
        "window_count": len(windows),
        "evaluation_date_count": int(forecasts["date"].nunique()),
        "annual_forecast_counts": annual_forecast_counts,
        "annual_active_security_counts": annual_active_counts,
        "first_forecast_date": forecasts["date"].min().date().isoformat(),
        "last_forecast_date": forecasts["date"].max().date().isoformat(),
        "minimum_training_observations": int(windows["training_observations"].min()),
        "maximum_training_observations": int(windows["training_observations"].max()),
        "required_minimum_training_observations": minimum_observations,
        "maximum_realised_loss_identity_error": maximum_loss_identity_error,
        "issues": issues,
    }
    return forecasts, windows, audit


def validate_portfolio_audit_binding(
    audit: Mapping[str, Any],
    *,
    returns_path: Path,
    active_universe_path: Path,
) -> None:
    """Require a passed arithmetic audit bound to both canonical M0 inputs."""

    if audit.get("status") != "pass":
        raise RuntimeError("portfolio-arithmetic gate is not passed")
    if audit.get("portfolio_definition") != "daily_rebalanced_equal_weight":
        raise RuntimeError("portfolio-arithmetic audit has an incompatible portfolio definition")
    expected = {
        "simple_return_panel": returns_path,
        "active_universe": active_universe_path,
    }
    for name, path in expected.items():
        if audit.get("inputs", {}).get(name, {}).get("sha256") != sha256_file(path):
            raise RuntimeError(f"{name} differs from the passed portfolio-arithmetic audit")


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
        "--model-config", type=Path, default=PROJECT_ROOT / "config/model_config.toml"
    )
    parser.add_argument(
        "--foundation-status",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_gate_status.json",
    )
    parser.add_argument(
        "--forecasts-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/historical_simulation_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--windows-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/historical_simulation_windows.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/historical_simulation_quality.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    validate_historical_protocol(model_config)
    foundation = json.loads(args.foundation_status.read_text(encoding="utf-8"))
    scope = reporting_scope(foundation)
    portfolio_audit = json.loads(args.portfolio_audit.read_text(encoding="utf-8"))
    validate_portfolio_audit_binding(
        portfolio_audit,
        returns_path=args.returns,
        active_universe_path=args.active_universe,
    )
    active_schedule = json.loads(args.active_universe.read_text(encoding="utf-8"))
    forecasts, windows, audit = build_historical_simulation_outputs(
        pd.read_parquet(args.returns),
        active_schedule,
        model_config,
    )
    write_parquet_atomic(args.forecasts_output, forecasts)
    write_parquet_atomic(args.windows_output, windows)
    audit.update(
        {
            "reporting_scope": scope,
            "method": {
                "portfolio_definition": "daily_rebalanced_equal_weight",
                "realised_loss_definition": "negative_simple_portfolio_return",
                "window_calendar_years": model_config["forecast"]["training_window_calendar_years"],
                "window_interval": "left_closed_right_open",
                "minimum_training_observations": model_config["forecast"][
                    "minimum_training_observations"
                ],
                "var_confidence_levels": model_config["forecast"]["var_confidence_levels"],
                "es_confidence_level": model_config["forecast"]["es_confidence_level"],
                "quantile_method": model_config["historical_simulation"]["quantile_method"],
                "es_boundary_method": model_config["historical_simulation"]["es_boundary_method"],
            },
            "inputs": {
                "portfolio_constituent_simple_returns": {
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
                "model_config": {
                    "path": project_path(args.model_config),
                    "sha256": sha256_file(args.model_config),
                },
                "foundation_status": {
                    "path": project_path(args.foundation_status),
                    "sha256": sha256_file(args.foundation_status),
                },
            },
            "outputs": {
                "historical_simulation_risk_forecasts": {
                    "path": project_path(args.forecasts_output),
                    "sha256": sha256_file(args.forecasts_output),
                    "rows": len(forecasts),
                },
                "historical_simulation_windows": {
                    "path": project_path(args.windows_output),
                    "sha256": sha256_file(args.windows_output),
                    "rows": len(windows),
                },
            },
        }
    )
    write_json_atomic(args.audit_output, audit)
    print(
        f"historical_simulation_quality={audit['status']} "
        f"forecasts={len(forecasts)} min_training={audit['minimum_training_observations']}"
    )
    print(f"Audit: {args.audit_output}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
