#!/usr/bin/env python3
"""Validate the frozen daily-rebalanced portfolio arithmetic.

The active set and therefore the group-size weights are annual.  In particular,
terminal securities that are removed at an annual rebalance must not remain in
the following year's group counts merely because their columns still exist in
the wide return panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.pipeline_io import write_json_atomic
    from scripts.research_methods import group_simple_returns
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import write_json_atomic
    from research_methods import group_simple_returns


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _issue(code: str, **details: object) -> dict[str, object]:
    return {"code": code, **details}


def _validated_return_panel(stock_simple_returns: pd.DataFrame) -> pd.DataFrame:
    """Return a date-indexed copy after enforcing panel-level invariants."""

    if stock_simple_returns.empty:
        raise ValueError("simple-return panel must not be empty")
    panel = stock_simple_returns.copy()
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index), name=panel.index.name or "date")
    if panel.index.has_duplicates:
        raise ValueError("simple-return panel has duplicate dates")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("simple-return panel dates must be sorted")
    if panel.columns.has_duplicates:
        raise ValueError("simple-return panel has duplicate security columns")
    return panel


def _schedule_by_year(active_schedule: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    """Index validated annual active-set rows by year."""

    schedule_rows = active_schedule.get("years")
    if not isinstance(schedule_rows, list) or not schedule_rows:
        raise ValueError("active schedule must contain a non-empty years list")
    indexed: dict[int, Mapping[str, object]] = {}
    for row in schedule_rows:
        if not isinstance(row, Mapping) or "year" not in row:
            raise ValueError("each active-schedule row must contain a year")
        year = int(row["year"])
        if year in indexed:
            raise ValueError(f"active schedule contains duplicate year {year}")
        indexed[year] = row
    return indexed


def _active_tickers_for_year(
    row: Mapping[str, object],
    year: int,
    panel: pd.DataFrame,
    labels: Mapping[str, str],
    issues: list[dict[str, object]],
) -> list[str] | None:
    """Validate one annual active set and return tickers when it is usable."""

    active_value = row.get("active_tickers")
    if not isinstance(active_value, list) or not active_value:
        issues.append(_issue("empty_annual_active_set", year=year))
        return None
    active = [str(ticker) for ticker in active_value]
    if len(active) != len(set(active)):
        issues.append(_issue("duplicate_annual_active_security", year=year))
        return None
    declared_count = row.get("security_count")
    if declared_count is None or int(declared_count) != len(active):
        issues.append(
            _issue(
                "annual_security_count_mismatch",
                year=year,
                declared=declared_count,
                observed=len(active),
            )
        )
    missing_columns = sorted(set(active) - set(panel.columns))
    missing_labels = sorted(set(active) - set(labels))
    if missing_columns:
        issues.append(
            _issue("active_security_missing_from_panel", year=year, tickers=missing_columns)
        )
    if missing_labels:
        issues.append(
            _issue("active_security_missing_group_label", year=year, tickers=missing_labels)
        )
    return None if missing_columns or missing_labels else active


def _complete_annual_returns(
    annual: pd.DataFrame,
    year: int,
    earliest_date: pd.Timestamp,
    issues: list[dict[str, object]],
) -> tuple[pd.DataFrame, list[str]]:
    """Select complete return rows and record invalid active-period observations."""

    values = annual.to_numpy(dtype=float)
    complete_rows = np.isfinite(values).all(axis=1)
    allowed_undefined = np.isnan(values).all(axis=1) & (annual.index == earliest_date)
    bad_rows = ~(complete_rows | allowed_undefined)
    if bool(bad_rows.any()):
        bad_dates = [date.date().isoformat() for date in annual.index[bad_rows]]
        issues.append(
            _issue("nonfinite_active_return", year=year, dates=bad_dates, count=len(bad_dates))
        )
    undefined_dates = [date.date().isoformat() for date in annual.index[allowed_undefined]]
    return annual.loc[complete_rows], undefined_dates


def _reconstruction_metrics(
    checked: pd.DataFrame,
    active: list[str],
    labels: Mapping[str, str],
    year: int,
    tolerance: float,
    issues: list[dict[str, object]],
) -> tuple[dict[str, int], float | None, list[float]]:
    """Reconstruct the equal-weight portfolio from groups and measure its error."""

    grouped = group_simple_returns(checked, {ticker: labels[ticker] for ticker in active})
    direct = checked.mean(axis=1)
    weights = grouped.group_sizes / len(active)
    reconstructed = grouped.simple_returns.mul(weights, axis="columns").sum(axis=1)
    errors = (direct - reconstructed).abs()
    maximum_error = float(errors.max()) if len(errors) else None
    if maximum_error is not None and maximum_error > tolerance:
        issues.append(
            _issue(
                "portfolio_reconstruction_tolerance_exceeded",
                year=year,
                date=errors.idxmax().date().isoformat(),
                maximum_absolute_error=maximum_error,
                tolerance=tolerance,
            )
        )
    group_sizes = {key: int(value) for key, value in grouped.group_sizes.items()}
    return group_sizes, maximum_error, [float(value) for value in errors]


def validate_annual_reconstruction(
    stock_simple_returns: pd.DataFrame,
    active_schedule: Mapping[str, object],
    labels: Mapping[str, str],
    *,
    grouping_id: str = "gics_sector",
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Validate direct versus group-size-weighted returns for each annual active set.

    The one all-missing row at the very start of the panel is permitted because
    no prior close exists from which to define a return.  Any other non-finite
    value in an active security is a failure.  Inactive columns are deliberately
    ignored after their annual removal.
    """

    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    panel = _validated_return_panel(stock_simple_returns)
    schedule_by_year = _schedule_by_year(active_schedule)

    panel_years = sorted(int(year) for year in panel.index.year.unique())
    issues: list[dict[str, object]] = []
    if set(panel_years) != set(schedule_by_year):
        issues.append(
            _issue(
                "schedule_panel_year_mismatch",
                panel_years=panel_years,
                schedule_years=sorted(schedule_by_year),
            )
        )

    earliest_date = panel.index.min()
    annual_results: list[dict[str, object]] = []
    all_errors: list[float] = []
    undefined_dates: list[str] = []

    for year in panel_years:
        row = schedule_by_year.get(year)
        if row is None:
            continue
        active = _active_tickers_for_year(row, year, panel, labels, issues)
        if active is None:
            continue

        annual = panel.loc[panel.index.year == year, active]
        checked, year_undefined = _complete_annual_returns(
            annual,
            year,
            earliest_date,
            issues,
        )
        undefined_dates.extend(year_undefined)
        group_sizes, maximum_error, errors = _reconstruction_metrics(
            checked,
            active,
            labels,
            year,
            tolerance,
            issues,
        )
        all_errors.extend(errors)

        annual_results.append(
            {
                "year": year,
                "rebalance_date": row.get("rebalance_date"),
                "session_count": len(annual),
                "checked_return_dates": len(checked),
                "undefined_initial_dates": year_undefined,
                "active_security_count": len(active),
                "group_count": len(group_sizes),
                "group_sizes": group_sizes,
                "maximum_absolute_error": maximum_error,
                "status": "pass"
                if not any(issue.get("year") == year for issue in issues)
                else "fail",
            }
        )

    maximum_error = max(all_errors) if all_errors else None
    status = "pass" if not issues and maximum_error is not None else "fail"
    return {
        "schema_version": 1,
        "gate_name": "portfolio_arithmetic_v1",
        "status": status,
        "grouping_id": grouping_id,
        "portfolio_definition": "daily_rebalanced_equal_weight",
        "aggregation_rule": "annual active group sizes divided by annual active security count",
        "tolerance": tolerance,
        "session_count": len(panel),
        "checked_return_dates": sum(result["checked_return_dates"] for result in annual_results),
        "undefined_initial_dates": undefined_dates,
        "maximum_absolute_error": maximum_error,
        "annual_results": annual_results,
        "issues": issues,
    }


def _gics_labels(universe: Mapping[str, object]) -> dict[str, str]:
    constituents = universe.get("constituents")
    if not isinstance(constituents, list):
        raise ValueError("final universe must contain a constituents list")
    labels: dict[str, str] = {}
    for row in constituents:
        if row.get("initial_universe_status") != "included":
            continue
        ticker = str(row["ticker"])
        if ticker in labels:
            raise ValueError(f"duplicate included ticker in final universe: {ticker}")
        labels[ticker] = str(row["gics_sector"])
    return labels


def main() -> int:
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/portfolio_arithmetic.json",
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    panel = pd.read_parquet(args.returns)
    with args.active_universe.open() as handle:
        schedule = json.load(handle)
    with args.universe.open() as handle:
        universe = json.load(handle)
    report = validate_annual_reconstruction(
        panel,
        schedule,
        _gics_labels(universe),
        tolerance=args.tolerance,
    )
    report["inputs"] = {
        "simple_return_panel": {
            "path": _manifest_path(args.returns),
            "sha256": _sha256(args.returns),
        },
        "active_universe": {
            "path": _manifest_path(args.active_universe),
            "sha256": _sha256(args.active_universe),
        },
        "final_universe": {
            "path": _manifest_path(args.universe),
            "sha256": _sha256(args.universe),
        },
    }
    write_json_atomic(args.output, report)
    print(
        f"portfolio_arithmetic={report['status']} "
        f"checked_dates={report['checked_return_dates']} "
        f"max_error={report['maximum_absolute_error']}"
    )
    print(f"Audit: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
