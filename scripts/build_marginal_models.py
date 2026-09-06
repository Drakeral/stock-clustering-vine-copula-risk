#!/usr/bin/env python3
"""Fit frozen monthly marginal models and produce daily filtered forecasts.

The primary run covers the security-level GICS and hierarchical groupings. Each
month is fitted from the left-closed, right-open three-calendar-year window. The
selected parameters remain fixed within the month while conditional states are
updated after every observed return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch.univariate import ARX, GARCH, ConstantMean, StudentsT

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.research_methods import (
        fallback_fraction_passes,
        margin_fit_rejection_reasons,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from research_methods import fallback_fraction_passes, margin_fit_rejection_reasons


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MonthlyTask:
    year: int
    month: int
    universe_variant: str
    grouping_id: str
    group_id: str
    group_size: int
    portfolio_weight: float
    refit_date: pd.Timestamp
    requested_training_start: pd.Timestamp
    training: pd.Series
    evaluation: pd.Series

    @property
    def refit_id(self) -> str:
        return (
            f"{self.year}-{self.month:02d}:{self.universe_variant}:"
            f"{self.grouping_id}:{self.group_id}"
        )


@dataclass(frozen=True)
class MarginState:
    selected_method: str
    fallback_level: int
    mean_constant: float
    phi: float
    omega: float
    alpha: float
    beta: float
    student_t_df: float | None
    next_variance: float
    previous_return: float
    training_standardized_residuals: pd.Series
    empirical_innovations: np.ndarray | None
    convergence_flag: int | None
    loglikelihood: float | None
    aic: float | None
    bic: float | None


ArchFitFunction = Callable[
    [pd.Series, str, str, np.ndarray | None, int, Mapping[str, Any]],
    tuple[MarginState | None, dict[str, Any]],
]


class PersistenceBoundedGARCH(GARCH):
    """GARCH(1,1) with the frozen strict persistence limit in estimation."""

    def __init__(self, persistence_limit: float) -> None:
        if not np.isfinite(persistence_limit) or not 0 < persistence_limit < 1:
            raise ValueError("persistence_limit must lie in (0, 1)")
        super().__init__(p=1, o=0, q=1)
        self._persistence_limit = float(np.nextafter(persistence_limit, 0.0))

    def constraints(self) -> tuple[np.ndarray, np.ndarray]:
        matrix, boundary = super().constraints()
        boundary[-1] = -self._persistence_limit
        return matrix, boundary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _finite_config_float(
    section: Mapping[str, Any], key: str, *, section_name: str = "marginal"
) -> float:
    """Read one required finite floating-point configuration value."""

    try:
        raw_value = section[key]
        if isinstance(raw_value, bool):
            raise TypeError
        value = float(raw_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid {section_name} configuration value: {key}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{section_name} configuration value must be finite: {key}")
    return value


def _positive_config_integer(
    section: Mapping[str, Any], key: str, *, section_name: str = "marginal"
) -> int:
    """Read one required positive integer configuration value."""

    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{section_name} configuration value must be a positive integer: {key}")
    return value


def _reporting_scope(gate: Mapping[str, Any]) -> str:
    foundation = gate.get("foundation_v2", {})
    readiness = gate.get("gates", {}).get("modelling_readiness", {})
    if foundation.get("status") not in {"pass", "provisional_pass"}:
        raise RuntimeError("foundation_v2 is not ready for modelling")
    if readiness.get("status") not in {"pass", "pass_provisional"}:
        raise RuntimeError("modelling-readiness gate is not passed")
    scope = foundation.get("reporting_scope")
    if scope not in {"confirmatory", "provisional_research_results"}:
        raise RuntimeError(f"unsupported reporting scope: {scope}")
    if readiness.get("status") == "pass_provisional" and scope != "provisional_research_results":
        raise RuntimeError("provisional readiness requires provisional reporting scope")
    return str(scope)


def _validate_protocol(model_config: Mapping[str, Any]) -> Mapping[str, Any]:
    marginal = model_config.get("marginal")
    if not isinstance(marginal, Mapping):
        raise TypeError("model configuration must contain a marginal table")
    expected = {
        "primary": "AR(1)-GARCH(1,1)-Student-t",
        "fallback_order": [
            "retry_ar_garch_t",
            "constant_mean_garch_t",
            "ewma_empirical",
        ],
        "ewma_mean": "constant_training_mean",
        "ewma_initial_variance": "sample_variance_ddof_1",
        "ewma_innovation_distribution": "frozen_training_empirical_midrank",
        "drop_ar_for_insignificant_p_value": False,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": marginal.get(key)}
        for key, expected_value in expected.items()
        if marginal.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"unsupported marginal protocol: {mismatches}")
    _positive_config_integer(marginal, "initial_optimizer_max_iterations")
    _positive_config_integer(marginal, "retry_optimizer_max_iterations")
    scale = _finite_config_float(marginal, "estimation_return_scale")
    tolerance = _finite_config_float(marginal, "optimizer_tolerance")
    phi_clip = _finite_config_float(marginal, "retry_start_phi_clip")
    retry_alpha = _finite_config_float(marginal, "retry_start_alpha")
    retry_beta = _finite_config_float(marginal, "retry_start_beta")
    retry_df = _finite_config_float(marginal, "retry_start_student_t_df")
    ar_limit = _finite_config_float(marginal, "ar_absolute_limit")
    persistence_limit = _finite_config_float(marginal, "garch_persistence_limit")
    df_minimum = _finite_config_float(marginal, "student_t_df_minimum")
    ewma_lambda = _finite_config_float(marginal, "ewma_lambda")
    lower = _finite_config_float(marginal, "pit_clip_lower")
    upper = _finite_config_float(marginal, "pit_clip_upper")
    maximum_ewma = _finite_config_float(marginal, "maximum_ewma_fit_fraction")
    invalid: list[str] = []
    if scale <= 0:
        invalid.append("estimation_return_scale must be positive")
    if tolerance <= 0:
        invalid.append("optimizer_tolerance must be positive")
    if not 0 < ar_limit < 1:
        invalid.append("ar_absolute_limit must lie in (0, 1)")
    if not 0 < phi_clip < ar_limit:
        invalid.append("retry_start_phi_clip must lie in (0, ar_absolute_limit)")
    if not 0 < persistence_limit < 1:
        invalid.append("garch_persistence_limit must lie in (0, 1)")
    if retry_alpha < 0 or retry_beta < 0 or retry_alpha + retry_beta >= persistence_limit:
        invalid.append("retry GARCH starts must be nonnegative and below the persistence limit")
    if df_minimum <= 2:
        invalid.append("student_t_df_minimum must exceed 2")
    if retry_df <= df_minimum:
        invalid.append("retry_start_student_t_df must exceed student_t_df_minimum")
    if not 0 < ewma_lambda < 1:
        invalid.append("ewma_lambda must lie in (0, 1)")
    if not 0 < lower < upper < 1:
        invalid.append("PIT clipping bounds must satisfy 0 < lower < upper < 1")
    if not 0 <= maximum_ewma <= 1:
        invalid.append("maximum_ewma_fit_fraction must lie in [0, 1]")
    if invalid:
        raise ValueError("invalid marginal configuration: " + "; ".join(invalid))
    return marginal


def _task_protocol(model_config: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Validate and return the frozen task-window configuration."""

    forecast = model_config.get("forecast")
    clustering = model_config.get("clustering")
    if not isinstance(forecast, Mapping) or not isinstance(clustering, Mapping):
        raise TypeError("model configuration must contain forecast and clustering tables")
    calendar_years = _positive_config_integer(
        forecast, "training_window_calendar_years", section_name="forecast"
    )
    minimum = _positive_config_integer(
        forecast, "minimum_training_observations", section_name="forecast"
    )
    start_year = _positive_config_integer(
        clustering, "evaluation_start_year", section_name="clustering"
    )
    end_year = _positive_config_integer(
        clustering, "evaluation_end_year", section_name="clustering"
    )
    if minimum < 3:
        raise ValueError("minimum_training_observations must be at least three")
    if start_year > end_year:
        raise ValueError("evaluation_start_year must not exceed evaluation_end_year")
    return calendar_years, minimum, start_year, end_year


def _validate_grouping_binding(
    clustering_audit: Mapping[str, Any], group_returns_path: Path
) -> None:
    if clustering_audit.get("status") != "pass":
        raise RuntimeError("annual grouping audit is not passed")
    recorded = clustering_audit.get("outputs", {}).get("annual_group_returns", {})
    if recorded.get("sha256") != _sha256(group_returns_path):
        raise RuntimeError("group-return panel differs from the passed clustering audit")


def _validate_group_metadata(frame: pd.DataFrame) -> None:
    """Validate annual group sizes and their portfolio-weight identity."""

    group_key = ["year", "universe_variant", "grouping_id", "group_id"]
    variation = frame.groupby(group_key, sort=False)[["group_size", "portfolio_weight"]].nunique(
        dropna=False
    )
    if bool((variation > 1).any().any()):
        raise ValueError("group size or portfolio weight changes within an annual group")
    metadata = frame[group_key + ["group_size", "portfolio_weight"]].drop_duplicates()
    sizes = metadata["group_size"].to_numpy(dtype=float)
    weights = metadata["portfolio_weight"].to_numpy(dtype=float)
    if bool((sizes <= 0).any()) or not np.equal(sizes, np.floor(sizes)).all():
        raise ValueError("group sizes must be positive integers")
    if bool(((weights <= 0) | (weights > 1)).any()):
        raise ValueError("portfolio weights must lie in (0, 1]")
    for identifiers, annual_groups in metadata.groupby(
        ["year", "universe_variant", "grouping_id"], sort=False
    ):
        expected = annual_groups["group_size"] / annual_groups["group_size"].sum()
        if not np.allclose(
            annual_groups["portfolio_weight"].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"portfolio weights do not equal normalized group sizes: {identifiers}"
            )


def prepare_monthly_tasks(
    group_returns: pd.DataFrame,
    model_config: Mapping[str, Any],
    *,
    universe_variant: str = "security_primary",
) -> list[MonthlyTask]:
    """Validate the long panel and build leakage-free group-month tasks."""

    required = [
        "date",
        "year",
        "universe_variant",
        "sample_role",
        "grouping_id",
        "group_id",
        "group_size",
        "portfolio_weight",
        "log_return",
    ]
    missing = sorted(set(required) - set(group_returns.columns))
    if missing:
        raise ValueError(f"group-return panel is missing columns: {missing}")
    frame = group_returns.loc[
        group_returns["universe_variant"] == universe_variant, required
    ].copy()
    if frame.empty:
        raise ValueError(f"group-return panel has no rows for {universe_variant}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("group-return panel contains an invalid date")
    if not set(frame["sample_role"]).issubset({"training", "evaluation"}):
        raise ValueError("group-return panel contains an invalid sample_role")
    key = ["date", "year", "universe_variant", "sample_role", "grouping_id", "group_id"]
    if frame.duplicated(key).any():
        raise ValueError("group-return panel has duplicate logical rows")
    if not np.isfinite(
        frame[["group_size", "portfolio_weight", "log_return"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("group-return panel contains non-finite required values")
    _validate_group_metadata(frame)

    calendar_years, minimum, start_year, end_year = _task_protocol(model_config)
    tasks: list[MonthlyTask] = []
    for (year, grouping_id, group_id), group in frame.groupby(
        ["year", "grouping_id", "group_id"], sort=True
    ):
        year = int(year)
        if not start_year <= year <= end_year:
            continue
        group = group.sort_values("date")
        duplicate_dates = group.loc[group["date"].duplicated(keep=False), "date"]
        if not duplicate_dates.empty:
            dates = sorted(
                pd.Timestamp(timestamp).date().isoformat() for timestamp in duplicate_dates.unique()
            )
            raise ValueError(
                f"overlapping sample roles for {year}/{grouping_id}/{group_id}: {dates[:5]}"
            )
        group_size = int(group["group_size"].iloc[0])
        portfolio_weight = float(group["portfolio_weight"].iloc[0])
        evaluation = group.loc[group["sample_role"] == "evaluation"]
        if evaluation.empty or not (evaluation["date"].dt.year == year).all():
            raise ValueError(f"invalid evaluation rows for {year}/{grouping_id}/{group_id}")
        for period, month_rows in evaluation.groupby(evaluation["date"].dt.to_period("M")):
            month_rows = month_rows.sort_values("date")
            refit_date = pd.Timestamp(month_rows["date"].iloc[0])
            requested_start = refit_date - pd.DateOffset(years=calendar_years)
            training = group.loc[
                (group["date"] >= requested_start) & (group["date"] < refit_date),
                ["date", "log_return"],
            ]
            if len(training) < minimum:
                raise ValueError(
                    f"{year}/{period.month}/{grouping_id}/{group_id} has "
                    f"{len(training)} training observations; requires {minimum}"
                )
            training_series = training.set_index("date")["log_return"].sort_index()
            evaluation_series = month_rows.set_index("date")["log_return"].sort_index()
            if training_series.index.max() >= refit_date:
                raise AssertionError("training window includes the refit date")
            tasks.append(
                MonthlyTask(
                    year=year,
                    month=int(period.month),
                    universe_variant=universe_variant,
                    grouping_id=str(grouping_id),
                    group_id=str(group_id),
                    group_size=group_size,
                    portfolio_weight=portfolio_weight,
                    refit_date=refit_date,
                    requested_training_start=requested_start,
                    training=training_series,
                    evaluation=evaluation_series,
                )
            )
    tasks.sort(key=lambda item: (item.year, item.month, item.grouping_id, item.group_id))
    if not tasks:
        raise ValueError("no monthly marginal tasks were created")
    return tasks


def _ar_starting_values(sample: pd.Series, marginal: Mapping[str, Any]) -> np.ndarray:
    values = sample.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(values) - 1), values[:-1]])
    intercept, phi = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    phi = float(
        np.clip(phi, -float(marginal["retry_start_phi_clip"]), marginal["retry_start_phi_clip"])
    )
    residuals = values[1:] - (intercept + phi * values[:-1])
    variance = max(float(np.var(residuals, ddof=1)), 1e-8)
    return np.asarray(
        [
            intercept,
            phi,
            0.05 * variance,
            float(marginal["retry_start_alpha"]),
            float(marginal["retry_start_beta"]),
            float(marginal["retry_start_student_t_df"]),
        ]
    )


def _constant_starting_values(sample: pd.Series, marginal: Mapping[str, Any]) -> np.ndarray:
    values = sample.to_numpy(dtype=float)
    mean = float(np.mean(values))
    variance = max(float(np.var(values - mean, ddof=1)), 1e-8)
    return np.asarray(
        [
            mean,
            0.05 * variance,
            float(marginal["retry_start_alpha"]),
            float(marginal["retry_start_beta"]),
            float(marginal["retry_start_student_t_df"]),
        ]
    )


def _parameter_value(parameters: pd.Series, name: str) -> float:
    return float(parameters[name]) if name in parameters else float("nan")


def _fit_arch_attempt(
    sample: pd.Series,
    attempt_id: str,
    mean_model: str,
    starting_values: np.ndarray | None,
    maximum_iterations: int,
    marginal: Mapping[str, Any],
) -> tuple[MarginState | None, dict[str, Any]]:
    """Fit one ARCH-library attempt and return explicit rejection details."""

    attempt: dict[str, Any] = {
        "attempt_id": attempt_id,
        "mean_model": mean_model,
        "status": "failed",
        "rejection_reasons": [],
        "exception_type": None,
        "exception_message": None,
        "warning_count": 0,
        "parameters": None,
    }
    try:
        volatility = PersistenceBoundedGARCH(float(marginal["garch_persistence_limit"]))
        model_options = {
            "volatility": volatility,
            "distribution": StudentsT(),
            "rescale": False,
        }
        model = (
            ARX(sample, lags=1, **model_options)
            if mean_model == "AR"
            else ConstantMean(sample, **model_options)
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = model.fit(
                disp="off",
                show_warning=False,
                update_freq=0,
                starting_values=starting_values,
                tol=float(marginal["optimizer_tolerance"]),
                options={"maxiter": int(maximum_iterations)},
            )
        attempt["warning_count"] = len(captured)
        parameters = result.params
        mean_constant = _parameter_value(parameters, "Const")
        if not np.isfinite(mean_constant):
            mean_constant = _parameter_value(parameters, "mu")
        ar_names = [
            name
            for name in parameters.index
            if name.endswith("[1]") and name not in {"alpha[1]", "beta[1]"}
        ]
        phi = float(parameters[ar_names[0]]) if mean_model == "AR" and ar_names else 0.0
        omega = _parameter_value(parameters, "omega")
        alpha = _parameter_value(parameters, "alpha[1]")
        beta = _parameter_value(parameters, "beta[1]")
        student_t_df = _parameter_value(parameters, "nu")
        residuals = result.resid.dropna()
        volatilities = result.conditional_volatility.dropna()
        standardized_residuals = result.std_resid.dropna().astype(float)
        if residuals.empty or volatilities.empty or standardized_residuals.empty:
            raise ValueError("fit produced no filtered residual or volatility state")
        if not np.isfinite(standardized_residuals.to_numpy()).all():
            raise ValueError("fit produced non-finite standardized residuals")
        last_residual = float(residuals.iloc[-1])
        last_variance = float(volatilities.iloc[-1] ** 2)
        forecast_variance = omega + alpha * last_residual**2 + beta * last_variance
        reasons = margin_fit_rejection_reasons(
            phi=phi,
            omega=omega,
            alpha=alpha,
            beta=beta,
            student_t_df=student_t_df,
            forecast_variance=forecast_variance,
            ar_absolute_limit=float(marginal["ar_absolute_limit"]),
            persistence_limit=float(marginal["garch_persistence_limit"]),
            student_t_df_minimum=float(marginal["student_t_df_minimum"]),
        )
        convergence_flag = int(result.convergence_flag)
        if convergence_flag != 0:
            reasons.insert(0, f"optimizer_convergence_flag_{convergence_flag}")
        attempt.update(
            {
                "status": "accepted" if not reasons else "rejected",
                "rejection_reasons": reasons,
                "convergence_flag": convergence_flag,
                "parameters": {key: float(value) for key, value in parameters.items()},
                "loglikelihood": float(result.loglikelihood),
                "aic": float(result.aic),
                "bic": float(result.bic),
            }
        )
        if reasons:
            return None, attempt
        return (
            MarginState(
                selected_method=(
                    "constant_mean_garch_t"
                    if mean_model == "Constant"
                    else "ar_garch_t_retry"
                    if attempt_id == "retry_ar_garch_t"
                    else "ar_garch_t"
                ),
                fallback_level=(
                    2 if mean_model == "Constant" else 1 if attempt_id.startswith("retry") else 0
                ),
                mean_constant=mean_constant,
                phi=phi,
                omega=omega,
                alpha=alpha,
                beta=beta,
                student_t_df=student_t_df,
                next_variance=forecast_variance,
                previous_return=float(sample.iloc[-1]),
                training_standardized_residuals=standardized_residuals,
                empirical_innovations=None,
                convergence_flag=convergence_flag,
                loglikelihood=float(result.loglikelihood),
                aic=float(result.aic),
                bic=float(result.bic),
            ),
            attempt,
        )
    except Exception as exc:  # The fallback contract requires logging numerical failures.
        attempt["exception_type"] = type(exc).__name__
        attempt["exception_message"] = str(exc)[:500]
        attempt["rejection_reasons"] = ["fit_exception"]
        return None, attempt


def _ewma_state(sample: pd.Series, marginal: Mapping[str, Any]) -> MarginState:
    values = sample.to_numpy(dtype=float)
    mean = float(np.mean(values))
    residuals = values - mean
    variance = float(np.var(residuals, ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("EWMA requires a finite positive training variance")
    smoothing = float(marginal["ewma_lambda"])
    innovations: list[float] = []
    for residual in residuals:
        innovations.append(float(residual / np.sqrt(variance)))
        variance = smoothing * variance + (1.0 - smoothing) * residual**2
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("EWMA produced an invalid forecast variance")
    return MarginState(
        selected_method="ewma_empirical",
        fallback_level=3,
        mean_constant=mean,
        phi=0.0,
        omega=0.0,
        alpha=1.0 - smoothing,
        beta=smoothing,
        student_t_df=None,
        next_variance=float(variance),
        previous_return=float(values[-1]),
        training_standardized_residuals=pd.Series(innovations, index=sample.index, dtype=float),
        empirical_innovations=np.asarray(innovations),
        convergence_flag=None,
        loglikelihood=None,
        aic=None,
        bic=None,
    )


def select_margin_model(
    training_log_returns: pd.Series,
    marginal: Mapping[str, Any],
    *,
    arch_fitter: ArchFitFunction | None = None,
) -> tuple[MarginState, list[dict[str, Any]]]:
    """Apply the exact initial/retry/constant/EWMA selection sequence."""

    scale = float(marginal["estimation_return_scale"])
    sample = (training_log_returns.astype(float) * scale).rename("return")
    if not np.isfinite(sample.to_numpy()).all() or len(sample) < 3:
        raise ValueError("marginal training sample must be finite with at least three rows")
    fitter = arch_fitter or _fit_arch_attempt
    attempts: list[dict[str, Any]] = []
    specifications = [
        (
            "initial_ar_garch_t",
            "AR",
            None,
            int(marginal["initial_optimizer_max_iterations"]),
        ),
        (
            "retry_ar_garch_t",
            "AR",
            _ar_starting_values(sample, marginal),
            int(marginal["retry_optimizer_max_iterations"]),
        ),
        (
            "constant_mean_garch_t",
            "Constant",
            _constant_starting_values(sample, marginal),
            int(marginal["retry_optimizer_max_iterations"]),
        ),
    ]
    for attempt_id, mean_model, starting_values, maximum_iterations in specifications:
        state, attempt = fitter(
            sample,
            attempt_id,
            mean_model,
            starting_values,
            maximum_iterations,
            marginal,
        )
        attempts.append(attempt)
        if state is not None:
            return state, attempts
    state = _ewma_state(sample, marginal)
    attempts.append(
        {
            "attempt_id": "ewma_empirical",
            "mean_model": "Constant",
            "status": "accepted",
            "rejection_reasons": [],
            "exception_type": None,
            "exception_message": None,
            "warning_count": 0,
            "parameters": {
                "mean": state.mean_constant,
                "lambda": state.beta,
                "initial_variance": "sample_variance_ddof_1",
            },
        }
    )
    return state, attempts


def _empirical_midrank_cdf(reference: np.ndarray, value: float) -> float:
    less = int(np.sum(reference < value))
    equal = int(np.sum(reference == value))
    return (less + 0.5 * equal) / len(reference)


def training_pit_frame(
    task: MonthlyTask,
    state: MarginState,
    marginal: Mapping[str, Any],
) -> pd.DataFrame:
    """Transform a fitted margin's in-sample residuals into training PITs."""

    standardized = state.training_standardized_residuals.sort_index()
    if standardized.empty or standardized.index.has_duplicates:
        raise ValueError(f"invalid training residual index for {task.refit_id}")
    if not standardized.index.isin(task.training.index).all():
        raise ValueError(f"training residual date falls outside {task.refit_id}")
    if standardized.index.max() >= task.refit_date:
        raise ValueError(f"training residual reaches the refit date for {task.refit_id}")
    values = standardized.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite training residual for {task.refit_id}")
    if state.student_t_df is not None:
        raw_pits = np.asarray(StudentsT().cdf(values, [state.student_t_df]), dtype=float)
    else:
        if state.empirical_innovations is None:
            raise AssertionError("EWMA state has no empirical innovation reference")
        raw_pits = np.asarray(
            [_empirical_midrank_cdf(state.empirical_innovations, value) for value in values],
            dtype=float,
        )
    lower = float(marginal["pit_clip_lower"])
    upper = float(marginal["pit_clip_upper"])
    pits = np.clip(raw_pits, lower, upper)
    training_returns = task.training.reindex(standardized.index)
    if training_returns.isna().any():
        raise AssertionError("training return alignment unexpectedly produced a missing value")
    return pd.DataFrame(
        {
            "training_date": standardized.index,
            "year": task.year,
            "month": task.month,
            "refit_date": task.refit_date,
            "universe_variant": task.universe_variant,
            "grouping_id": task.grouping_id,
            "group_id": task.group_id,
            "copula_refit_id": (
                f"{task.year}-{task.month:02d}:{task.universe_variant}:{task.grouping_id}"
            ),
            "margin_refit_id": task.refit_id,
            "selected_method": state.selected_method,
            "fallback_level": state.fallback_level,
            "training_log_return": training_returns.to_numpy(dtype=float),
            "standardized_residual": values,
            "pit": pits,
            "pit_was_clipped": pits != raw_pits,
        }
    )


def align_training_pits(
    training_pits: pd.DataFrame,
    refits: pd.DataFrame,
    *,
    expected_dimension: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep only dates observed for every group in each monthly copula block."""

    block_key = ["year", "month", "universe_variant", "grouping_id"]
    date_key = block_key + ["training_date"]
    issues: list[str] = []
    refit_dimensions = refits.groupby(block_key, sort=False)["group_id"].nunique()
    if bool((refit_dimensions != expected_dimension).any()):
        issues.append("unexpected_monthly_copula_dimension")
    observed_dimensions = training_pits.groupby(date_key, sort=False)["group_id"].transform(
        "nunique"
    )
    expected_by_row = training_pits.groupby(block_key, sort=False)["group_id"].transform("nunique")
    aligned = training_pits.loc[observed_dimensions == expected_by_row].copy()
    if aligned.empty:
        issues.append("no_complete_training_pit_dates")
        return aligned, issues
    aligned_dimensions = aligned.groupby(date_key, sort=False)["group_id"].nunique()
    if bool((aligned_dimensions != expected_dimension).any()):
        issues.append("incomplete_training_pit_matrix")
    aligned_groups = aligned.groupby(block_key, sort=False)["group_id"].nunique()
    if len(aligned_groups) != len(refit_dimensions):
        issues.append("missing_monthly_training_pit_block")
    if aligned.duplicated(date_key + ["group_id"]).any():
        issues.append("duplicate_training_pit_record")
    if bool((aligned["training_date"] >= aligned["refit_date"]).any()):
        issues.append("training_pit_lookahead")
    return aligned.sort_values(date_key + ["group_id"]).reset_index(drop=True), issues


def filter_month(
    task: MonthlyTask,
    state: MarginState,
    marginal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Produce daily one-step forecasts and PITs with fixed monthly parameters."""

    scale = float(marginal["estimation_return_scale"])
    lower = float(marginal["pit_clip_lower"])
    upper = float(marginal["pit_clip_upper"])
    variance = float(state.next_variance)
    previous_return = float(state.previous_return)
    student = StudentsT() if state.student_t_df is not None else None
    rows: list[dict[str, Any]] = []
    for date, realised_decimal in task.evaluation.items():
        realised = float(realised_decimal) * scale
        mean = state.mean_constant + state.phi * previous_return
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError(f"invalid filtered variance for {task.refit_id} on {date}")
        volatility = float(np.sqrt(variance))
        residual = realised - mean
        standardized = residual / volatility
        if student is not None:
            pit_raw = float(student.cdf(np.asarray([standardized]), [state.student_t_df])[0])
        else:
            if state.empirical_innovations is None:
                raise AssertionError("EWMA state has no empirical innovation reference")
            pit_raw = _empirical_midrank_cdf(state.empirical_innovations, standardized)
        pit = float(np.clip(pit_raw, lower, upper))
        rows.append(
            {
                "date": pd.Timestamp(date),
                "year": task.year,
                "month": task.month,
                "universe_variant": task.universe_variant,
                "grouping_id": task.grouping_id,
                "group_id": task.group_id,
                "group_size": task.group_size,
                "portfolio_weight": task.portfolio_weight,
                "refit_id": task.refit_id,
                "selected_method": state.selected_method,
                "fallback_level": state.fallback_level,
                "conditional_mean_log_return": mean / scale,
                "conditional_volatility_log_return": volatility / scale,
                "conditional_variance_log_return": variance / scale**2,
                "realised_log_return": float(realised_decimal),
                "standardized_residual": standardized,
                "pit": pit,
                "pit_was_clipped": bool(pit != pit_raw),
                "forecast_status": "ok" if state.fallback_level == 0 else "fallback",
            }
        )
        previous_return = realised
        variance = state.omega + state.alpha * residual**2 + state.beta * variance
    return rows


def run_monthly_task(
    task: MonthlyTask,
    marginal: Mapping[str, Any],
    *,
    arch_fitter: ArchFitFunction | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    state, attempts = select_margin_model(task.training, marginal, arch_fitter=arch_fitter)
    daily = filter_month(task, state, marginal)
    training_pits = training_pit_frame(task, state, marginal)
    record = {
        "refit_id": task.refit_id,
        "refit_date": task.refit_date,
        "year": task.year,
        "month": task.month,
        "universe_variant": task.universe_variant,
        "grouping_id": task.grouping_id,
        "group_id": task.group_id,
        "group_size": task.group_size,
        "portfolio_weight": task.portfolio_weight,
        "requested_training_start": task.requested_training_start,
        "first_training_date": task.training.index.min(),
        "last_training_date": task.training.index.max(),
        "training_observations": len(task.training),
        "evaluation_observations": len(task.evaluation),
        "selected_method": state.selected_method,
        "fallback_level": state.fallback_level,
        "fallback_used": state.fallback_level > 0,
        "ewma_used": state.selected_method == "ewma_empirical",
        "mean_constant_scaled": state.mean_constant,
        "phi": state.phi,
        "omega_scaled": state.omega,
        "alpha": state.alpha,
        "beta": state.beta,
        "student_t_df": state.student_t_df,
        "first_forecast_variance_scaled": state.next_variance,
        "convergence_flag": state.convergence_flag,
        "loglikelihood": state.loglikelihood,
        "aic": state.aic,
        "bic": state.bic,
        "attempt_count": len(attempts),
        "attempt_log_json": json.dumps(attempts, sort_keys=True, separators=(",", ":")),
        "fit_status": "ok" if state.fallback_level == 0 else "fallback",
    }
    return record, daily, training_pits


def build_marginal_outputs(
    group_returns: pd.DataFrame,
    model_config: Mapping[str, Any],
    *,
    universe_variant: str = "security_primary",
    progress_every: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    marginal = _validate_protocol(model_config)
    vine = model_config.get("vine")
    if not isinstance(vine, Mapping):
        raise TypeError("model configuration must contain a vine table")
    expected_dimension = _positive_config_integer(vine, "dimension", section_name="vine")
    tasks = prepare_monthly_tasks(group_returns, model_config, universe_variant=universe_variant)
    refits: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    training_frames: list[pd.DataFrame] = []
    for index, task in enumerate(tasks, start=1):
        refit, daily, training_pits = run_monthly_task(task, marginal)
        refits.append(refit)
        daily_rows.extend(daily)
        training_frames.append(training_pits)
        if progress_every and index % progress_every == 0:
            print(f"marginal_progress={index}/{len(tasks)}")
    refit_frame = pd.DataFrame(refits).sort_values(["year", "month", "grouping_id", "group_id"])
    daily_frame = pd.DataFrame(daily_rows).sort_values(["date", "grouping_id", "group_id"])
    raw_training_pits = pd.concat(training_frames, ignore_index=True)
    training_pit_panel, training_pit_issues = align_training_pits(
        raw_training_pits,
        refit_frame,
        expected_dimension=expected_dimension,
    )
    refit_key = ["year", "month", "universe_variant", "grouping_id", "group_id"]
    daily_key = ["date", "universe_variant", "grouping_id", "group_id"]
    issues: list[str] = list(training_pit_issues)
    if refit_frame.duplicated(refit_key).any():
        issues.append("duplicate_group_month_refit")
    if daily_frame.duplicated(daily_key).any():
        issues.append("duplicate_daily_margin_forecast")
    if int(refit_frame["evaluation_observations"].sum()) != len(daily_frame):
        issues.append("incomplete_monthly_evaluation_coverage")
    month_counts = refit_frame.groupby(
        ["year", "universe_variant", "grouping_id", "group_id"], sort=False
    )["month"].nunique()
    if bool((month_counts != 12).any()):
        issues.append("incomplete_annual_refit_schedule")
    expected_group_counts = (
        refit_frame.groupby(["year", "grouping_id"], sort=False)["group_id"].nunique().to_dict()
    )
    observed_group_counts = daily_frame.groupby(["year", "date", "grouping_id"], sort=False)[
        "group_id"
    ].nunique()
    if any(
        count != expected_group_counts[(year, grouping_id)]
        for (year, _date, grouping_id), count in observed_group_counts.items()
    ):
        issues.append("incomplete_daily_group_coverage")
    numeric_daily = daily_frame[
        [
            "conditional_mean_log_return",
            "conditional_volatility_log_return",
            "conditional_variance_log_return",
            "realised_log_return",
            "standardized_residual",
            "pit",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric_daily).all():
        issues.append("nonfinite_daily_margin_output")
    lower = float(marginal["pit_clip_lower"])
    upper = float(marginal["pit_clip_upper"])
    if bool(((daily_frame["pit"] < lower) | (daily_frame["pit"] > upper)).any()):
        issues.append("pit_outside_frozen_bounds")
    numeric_training = training_pit_panel[
        ["training_log_return", "standardized_residual", "pit"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric_training).all():
        issues.append("nonfinite_training_pit_output")
    if bool(((training_pit_panel["pit"] < lower) | (training_pit_panel["pit"] > upper)).any()):
        issues.append("training_pit_outside_frozen_bounds")
    block_key = ["year", "month", "universe_variant", "grouping_id"]
    aligned_date_counts = training_pit_panel.groupby(block_key, sort=False)[
        "training_date"
    ].nunique()
    minimum_aligned = int(aligned_date_counts.min()) if not aligned_date_counts.empty else 0
    maximum_aligned = int(aligned_date_counts.max()) if not aligned_date_counts.empty else 0
    minimum_required = int(model_config["forecast"]["minimum_training_observations"])
    if minimum_aligned < minimum_required:
        issues.append("insufficient_aligned_training_pit_dates")
    ewma_count = int(refit_frame["ewma_used"].sum())
    maximum_ewma = float(marginal["maximum_ewma_fit_fraction"])
    if not fallback_fraction_passes(ewma_count, len(refit_frame), maximum=maximum_ewma):
        issues.append("ewma_fit_fraction_exceeds_limit")
    selected_counts = {
        str(key): int(value) for key, value in refit_frame["selected_method"].value_counts().items()
    }
    audit = {
        "schema_version": 2,
        "gate_name": "marginal_model_quality_v2",
        "status": "pass" if not issues else "fail",
        "universe_variant": universe_variant,
        "refit_count": len(refit_frame),
        "daily_forecast_count": len(daily_frame),
        "raw_training_pit_record_count": len(raw_training_pits),
        "training_pit_record_count": len(training_pit_panel),
        "alignment_dropped_record_count": len(raw_training_pits) - len(training_pit_panel),
        "copula_training_block_count": len(aligned_date_counts),
        "copula_dimension": expected_dimension,
        "minimum_aligned_training_dates": minimum_aligned,
        "maximum_aligned_training_dates": maximum_aligned,
        "minimum_required_aligned_training_dates": minimum_required,
        "selected_method_counts": selected_counts,
        "fallback_fit_count": int(refit_frame["fallback_used"].sum()),
        "ewma_fit_count": ewma_count,
        "ewma_fit_fraction": ewma_count / len(refit_frame),
        "maximum_ewma_fit_fraction": maximum_ewma,
        "pit_clipped_count": int(daily_frame["pit_was_clipped"].sum()),
        "pit_minimum": float(daily_frame["pit"].min()),
        "pit_maximum": float(daily_frame["pit"].max()),
        "training_pit_clipped_count": int(training_pit_panel["pit_was_clipped"].sum()),
        "training_pit_minimum": float(training_pit_panel["pit"].min()),
        "training_pit_maximum": float(training_pit_panel["pit"].max()),
        "issues": issues,
    }
    return (
        refit_frame.reset_index(drop=True),
        daily_frame.reset_index(drop=True),
        training_pit_panel,
        audit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/annual_group_returns.parquet",
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
        "--foundation-status",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_gate_status.json",
    )
    parser.add_argument("--universe-variant", default="security_primary")
    parser.add_argument(
        "--refits-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/marginal_refits.parquet",
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/marginal_daily_forecasts.parquet",
    )
    parser.add_argument(
        "--training-pits-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/monthly_copula_training_pits.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/marginal_model_quality.json",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    marginal = _validate_protocol(model_config)
    gate = json.loads(args.foundation_status.read_text(encoding="utf-8"))
    scope = _reporting_scope(gate)
    clustering_audit = json.loads(args.clustering_audit.read_text(encoding="utf-8"))
    _validate_grouping_binding(clustering_audit, args.group_returns)
    group_returns = pd.read_parquet(args.group_returns)
    refits, daily, training_pits, audit = build_marginal_outputs(
        group_returns,
        model_config,
        universe_variant=args.universe_variant,
        progress_every=args.progress_every,
    )
    _write_parquet_atomic(args.refits_output, refits)
    _write_parquet_atomic(args.daily_output, daily)
    _write_parquet_atomic(args.training_pits_output, training_pits)
    audit.update(
        {
            "reporting_scope": scope,
            "method": {
                "primary": marginal["primary"],
                "fallback_order": marginal["fallback_order"],
                "estimation_return_scale": marginal["estimation_return_scale"],
                "ewma_lambda": marginal["ewma_lambda"],
                "pit_clip_lower": marginal["pit_clip_lower"],
                "pit_clip_upper": marginal["pit_clip_upper"],
            },
            "inputs": {
                "annual_group_returns": {
                    "path": _project_path(args.group_returns),
                    "sha256": _sha256(args.group_returns),
                },
                "clustering_audit": {
                    "path": _project_path(args.clustering_audit),
                    "sha256": _sha256(args.clustering_audit),
                },
                "model_config": {
                    "path": _project_path(args.model_config),
                    "sha256": _sha256(args.model_config),
                },
                "foundation_status": {
                    "path": _project_path(args.foundation_status),
                    "sha256": _sha256(args.foundation_status),
                },
            },
            "outputs": {
                "marginal_refits": {
                    "path": _project_path(args.refits_output),
                    "sha256": _sha256(args.refits_output),
                    "rows": len(refits),
                },
                "marginal_daily_forecasts": {
                    "path": _project_path(args.daily_output),
                    "sha256": _sha256(args.daily_output),
                    "rows": len(daily),
                },
                "monthly_copula_training_pits": {
                    "path": _project_path(args.training_pits_output),
                    "sha256": _sha256(args.training_pits_output),
                    "rows": len(training_pits),
                },
            },
        }
    )
    _write_json_atomic(args.audit_output, audit)
    print(
        f"marginal_model_quality={audit['status']} refits={len(refits)} "
        f"daily={len(daily)} training_pits={len(training_pits)} "
        f"ewma_fraction={audit['ewma_fit_fraction']:.6f}"
    )
    print(f"Audit: {args.audit_output}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
