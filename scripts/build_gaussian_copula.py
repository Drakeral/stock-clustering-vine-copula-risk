#!/usr/bin/env python3
"""Fit monthly Gaussian copulas and produce M1/M3 portfolio-risk forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri, stdtrit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.research_methods import common_uniforms
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from research_methods import common_uniforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_BY_GROUPING = {
    "gics_sector": ("M1", "gics"),
    "hierarchical_cluster": ("M3", "hierarchical"),
}


@dataclass(frozen=True)
class GaussianCopulaFit:
    correlation: np.ndarray
    raw_min_eigenvalue: float
    repaired_min_eigenvalue: float
    maximum_absolute_adjustment: float
    condition_number: float
    log_determinant: float
    correlation_repaired: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


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


def _finite_float(section: Mapping[str, Any], key: str, section_name: str) -> float:
    try:
        raw = section[key]
        if isinstance(raw, bool):
            raise TypeError
        value = float(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid {section_name} value: {key}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{section_name} value must be finite: {key}")
    return value


def _positive_integer(section: Mapping[str, Any], key: str, section_name: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{section_name} value must be a positive integer: {key}")
    return value


def _validate_protocol(
    model_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    gaussian = model_config.get("gaussian")
    simulation = model_config.get("simulation")
    forecast = model_config.get("forecast")
    marginal = model_config.get("marginal")
    portfolio = model_config.get("portfolio")
    vine = model_config.get("vine")
    sections = [gaussian, simulation, forecast, marginal, portfolio, vine]
    if not all(isinstance(section, Mapping) for section in sections):
        raise TypeError("model configuration is missing a required modelling table")
    assert isinstance(gaussian, Mapping)
    assert isinstance(simulation, Mapping)
    assert isinstance(forecast, Mapping)
    assert isinstance(marginal, Mapping)
    assert isinstance(portfolio, Mapping)
    assert isinstance(vine, Mapping)

    expected_gaussian = {
        "estimator": "pearson_correlation_of_normal_scores",
        "repair_method": "symmetric_eigenvalue_floor_then_unit_diagonal_rescale",
        "factorization": "cholesky",
        "log_score": "gaussian_copula_log_density",
    }
    expected_simulation = {
        "bit_generator": "PCG64DXSM",
        "seed_components": ["base_seed", "year", "month"],
        "common_random_numbers": True,
        "student_t_standardization": "unit_variance_sqrt_df_minus_2_over_df",
        "ewma_inverse_cdf": "inverted_empirical_cdf",
    }
    expected_portfolio = {
        "primary_return_scale": "simple",
        "primary_rebalancing": "daily",
        "group_weighting": "group_size",
        "realised_loss_definition": "negative_simple_portfolio_return",
    }
    mismatches: dict[str, Any] = {}
    for key, expected in expected_gaussian.items():
        if gaussian.get(key) != expected:
            mismatches[f"gaussian.{key}"] = {"expected": expected, "observed": gaussian.get(key)}
    for key, expected in expected_simulation.items():
        if simulation.get(key) != expected:
            mismatches[f"simulation.{key}"] = {
                "expected": expected,
                "observed": simulation.get(key),
            }
    for key, expected in expected_portfolio.items():
        if portfolio.get(key) != expected:
            mismatches[f"portfolio.{key}"] = {
                "expected": expected,
                "observed": portfolio.get(key),
            }
    if mismatches:
        raise ValueError(f"unsupported Gaussian risk protocol: {mismatches}")

    dimension = _positive_integer(simulation, "dimension", "simulation")
    draws = _positive_integer(simulation, "draws_per_month", "simulation")
    base_seed = _positive_integer(simulation, "base_seed", "simulation")
    vine_dimension = _positive_integer(vine, "dimension", "vine")
    if dimension != vine_dimension:
        raise ValueError("simulation and vine dimensions must agree")
    if draws < 100:
        raise ValueError("simulation.draws_per_month must be at least 100")
    if base_seed != 5110:
        raise ValueError("simulation.base_seed must remain 5110")
    confidence_levels = [float(value) for value in forecast.get("var_confidence_levels", [])]
    if confidence_levels != [0.95, 0.975, 0.99]:
        raise ValueError("forecast.var_confidence_levels differs from the frozen protocol")
    if float(forecast.get("es_confidence_level", float("nan"))) != 0.975:
        raise ValueError("forecast.es_confidence_level must be 0.975")
    if _positive_integer(forecast, "horizon_trading_days", "forecast") != 1:
        raise ValueError("only one-day forecasts are supported")
    lower = _finite_float(marginal, "pit_clip_lower", "marginal")
    upper = _finite_float(marginal, "pit_clip_upper", "marginal")
    if not 0 < lower < upper < 1:
        raise ValueError("marginal PIT bounds must lie strictly inside (0, 1)")
    eigenvalue_floor = _finite_float(gaussian, "eigenvalue_floor", "gaussian")
    repair_tolerance = _finite_float(gaussian, "repair_tolerance", "gaussian")
    maximum_condition = _finite_float(gaussian, "maximum_condition_number", "gaussian")
    if eigenvalue_floor <= 0 or repair_tolerance < 0 or maximum_condition <= 1:
        raise ValueError("invalid Gaussian numerical-stability thresholds")
    return gaussian, simulation, forecast, marginal


def fit_gaussian_copula(
    pits: np.ndarray,
    *,
    eigenvalue_floor: float = 1e-8,
    repair_tolerance: float = 1e-12,
) -> GaussianCopulaFit:
    """Estimate and, if necessary, deterministically repair a copula correlation."""

    values = np.asarray(pits, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("PIT matrix must have at least three rows and two columns")
    if not np.isfinite(values).all() or bool(((values <= 0) | (values >= 1)).any()):
        raise ValueError("PIT matrix must be finite and lie strictly inside (0, 1)")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be positive")
    if not np.isfinite(repair_tolerance) or repair_tolerance < 0:
        raise ValueError("repair_tolerance must be nonnegative")

    normal_scores = ndtri(values)
    raw = np.asarray(np.corrcoef(normal_scores, rowvar=False), dtype=float)
    if raw.shape != (values.shape[1], values.shape[1]) or not np.isfinite(raw).all():
        raise ValueError("normal-score correlation is non-finite or has the wrong shape")
    raw = 0.5 * (raw + raw.T)
    np.fill_diagonal(raw, 1.0)
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(raw)
    raw_minimum = float(raw_eigenvalues.min())
    repaired = raw.copy()
    if raw_minimum < eigenvalue_floor:
        floored = np.maximum(raw_eigenvalues, eigenvalue_floor)
        covariance = (raw_eigenvectors * floored) @ raw_eigenvectors.T
        standard_deviations = np.sqrt(np.diag(covariance))
        if not np.isfinite(standard_deviations).all() or bool((standard_deviations <= 0).any()):
            raise ValueError("spectral repair produced an invalid diagonal")
        repaired = covariance / np.outer(standard_deviations, standard_deviations)
        repaired = 0.5 * (repaired + repaired.T)
        np.fill_diagonal(repaired, 1.0)
    repaired_eigenvalues = np.linalg.eigvalsh(repaired)
    repaired_minimum = float(repaired_eigenvalues.min())
    if repaired_minimum <= 0:
        raise ValueError("Gaussian correlation repair did not produce positive definiteness")
    sign, log_determinant = np.linalg.slogdet(repaired)
    if sign <= 0 or not np.isfinite(log_determinant):
        raise ValueError("Gaussian correlation has an invalid determinant")
    np.linalg.cholesky(repaired)
    adjustment = float(np.max(np.abs(repaired - raw)))
    return GaussianCopulaFit(
        correlation=repaired,
        raw_min_eigenvalue=raw_minimum,
        repaired_min_eigenvalue=repaired_minimum,
        maximum_absolute_adjustment=adjustment,
        condition_number=float(np.linalg.cond(repaired)),
        log_determinant=float(log_determinant),
        correlation_repaired=adjustment > repair_tolerance,
    )


def gaussian_dependence_uniforms(
    independent_uniforms: np.ndarray,
    correlation: np.ndarray,
    *,
    lower: float = 1e-6,
    upper: float = 1 - 1e-6,
) -> np.ndarray:
    """Map independent uniforms through a Gaussian copula using Cholesky."""

    uniforms = np.asarray(independent_uniforms, dtype=float)
    matrix = np.asarray(correlation, dtype=float)
    if uniforms.ndim != 2 or matrix.shape != (uniforms.shape[1], uniforms.shape[1]):
        raise ValueError("uniform matrix and correlation dimensions do not agree")
    if not np.isfinite(uniforms).all() or bool(((uniforms < 0) | (uniforms > 1)).any()):
        raise ValueError("independent uniforms must be finite and lie in [0, 1]")
    if not 0 < lower < upper < 1:
        raise ValueError("copula bounds must lie strictly inside (0, 1)")
    factor = np.linalg.cholesky(matrix)
    independent_scores = ndtri(np.clip(uniforms, lower, upper))
    dependent_scores = independent_scores @ factor.T
    return np.clip(ndtr(dependent_scores), lower, upper)


def gaussian_copula_log_density(pits: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    """Return Gaussian copula log densities for one or more PIT vectors."""

    values = np.asarray(pits, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    matrix = np.asarray(correlation, dtype=float)
    if values.ndim != 2 or matrix.shape != (values.shape[1], values.shape[1]):
        raise ValueError("PIT and correlation dimensions do not agree")
    if not np.isfinite(values).all() or bool(((values <= 0) | (values >= 1)).any()):
        raise ValueError("PIT values must be finite and lie strictly inside (0, 1)")
    sign, log_determinant = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("correlation determinant must be positive")
    scores = ndtri(values)
    solved = np.linalg.solve(matrix, scores.T).T
    quadratic = np.sum(scores * solved, axis=1) - np.sum(scores * scores, axis=1)
    return -0.5 * float(log_determinant) - 0.5 * quadratic


def _ewma_empirical_innovations(
    group_returns: pd.DataFrame,
    refit: pd.Series,
    *,
    scale: float,
    smoothing: float,
) -> np.ndarray:
    """Reconstruct the complete frozen EWMA innovation reference sample."""

    dates = pd.to_datetime(group_returns["date"], errors="coerce")
    start = pd.Timestamp(refit["requested_training_start"])
    end = pd.Timestamp(refit["refit_date"])
    selected = group_returns.loc[
        (group_returns["year"] == int(refit["year"]))
        & (group_returns["universe_variant"] == refit["universe_variant"])
        & (group_returns["grouping_id"] == refit["grouping_id"])
        & (group_returns["group_id"] == refit["group_id"])
        & (dates >= start)
        & (dates < end)
    ].copy()
    selected["date"] = dates.loc[selected.index]
    selected = selected.sort_values("date")
    if len(selected) != int(refit["training_observations"]):
        raise ValueError(f"cannot reconstruct EWMA training sample for {refit['refit_id']}")
    values = selected["log_return"].to_numpy(dtype=float) * scale
    mean = float(refit["mean_constant_scaled"])
    if not np.isclose(mean, values.mean(), rtol=0.0, atol=1e-10):
        raise ValueError(f"EWMA mean does not match its training sample for {refit['refit_id']}")
    residuals = values - mean
    variance = float(np.var(residuals, ddof=1))
    innovations = np.empty(len(residuals), dtype=float)
    for index, residual in enumerate(residuals):
        innovations[index] = residual / np.sqrt(variance)
        variance = smoothing * variance + (1.0 - smoothing) * residual**2
    if not np.isclose(
        variance,
        float(refit["first_forecast_variance_scaled"]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(f"EWMA variance does not reproduce for {refit['refit_id']}")
    return innovations


def marginal_innovation_draws(
    copula_uniforms: np.ndarray,
    group_order: Sequence[str],
    monthly_refits: pd.DataFrame,
    group_returns: pd.DataFrame,
    marginal: Mapping[str, Any],
) -> np.ndarray:
    """Invert every fitted Student-t or empirical innovation distribution."""

    uniforms = np.asarray(copula_uniforms, dtype=float)
    if uniforms.shape[1] != len(group_order):
        raise ValueError("copula uniforms and group order have different dimensions")
    indexed = monthly_refits.set_index("group_id", verify_integrity=True)
    if set(indexed.index) != set(group_order):
        raise ValueError("monthly marginal refits do not match the copula group order")
    scale = float(marginal["estimation_return_scale"])
    smoothing = float(marginal["ewma_lambda"])
    result = np.empty_like(uniforms)
    for column, group_id in enumerate(group_order):
        refit = indexed.loc[group_id]
        degrees_of_freedom = refit["student_t_df"]
        if pd.notna(degrees_of_freedom):
            degrees_of_freedom = float(degrees_of_freedom)
            result[:, column] = stdtrit(degrees_of_freedom, uniforms[:, column]) * np.sqrt(
                (degrees_of_freedom - 2.0) / degrees_of_freedom
            )
        else:
            reference = _ewma_empirical_innovations(
                group_returns,
                refit,
                scale=scale,
                smoothing=smoothing,
            )
            result[:, column] = np.quantile(
                reference,
                uniforms[:, column],
                method="inverted_cdf",
            )
    if not np.isfinite(result).all():
        raise ValueError("marginal inversion produced a non-finite innovation")
    return result


def empirical_risk_levels(
    losses: np.ndarray,
    confidence_levels: Sequence[float],
) -> dict[float, tuple[float, float]]:
    """Compute inverted-CDF VaR and fractionally weighted ES with one partition."""

    values = np.asarray(losses, dtype=float).reshape(-1)
    levels = [float(level) for level in confidence_levels]
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("loss sample must be nonempty and finite")
    if not levels or any(not 0 < level < 1 for level in levels):
        raise ValueError("confidence levels must lie in (0, 1)")
    orders = {level: int(np.ceil(len(values) * level)) for level in levels}
    partitioned = np.partition(values.copy(), [order - 1 for order in orders.values()])
    risks: dict[float, tuple[float, float]] = {}
    for level, order in orders.items():
        value_at_risk = float(partitioned[order - 1])
        tail_mass = len(values) * (1.0 - level)
        boundary_weight = order - len(values) * level
        expected_shortfall = float(
            (partitioned[order:].sum() + boundary_weight * value_at_risk) / tail_mass
        )
        risks[level] = value_at_risk, expected_shortfall
    return risks


def _validate_input_frames(
    training_pits: pd.DataFrame,
    marginal_refits: pd.DataFrame,
    daily_margins: pd.DataFrame,
    group_returns: pd.DataFrame,
) -> None:
    requirements = {
        "training PIT": {
            "training_date",
            "year",
            "month",
            "refit_date",
            "universe_variant",
            "grouping_id",
            "group_id",
            "copula_refit_id",
            "pit",
        },
        "marginal refit": {
            "refit_id",
            "refit_date",
            "year",
            "month",
            "universe_variant",
            "grouping_id",
            "group_id",
            "requested_training_start",
            "training_observations",
            "selected_method",
            "fallback_level",
            "mean_constant_scaled",
            "student_t_df",
            "first_forecast_variance_scaled",
        },
        "daily margin": {
            "date",
            "year",
            "month",
            "universe_variant",
            "grouping_id",
            "group_id",
            "portfolio_weight",
            "conditional_mean_log_return",
            "conditional_volatility_log_return",
            "realised_log_return",
            "pit",
        },
        "group return": {
            "date",
            "year",
            "universe_variant",
            "grouping_id",
            "group_id",
            "log_return",
        },
    }
    frames = {
        "training PIT": training_pits,
        "marginal refit": marginal_refits,
        "daily margin": daily_margins,
        "group return": group_returns,
    }
    for name, required in requirements.items():
        missing = sorted(required - set(frames[name].columns))
        if missing:
            raise ValueError(f"{name} frame is missing columns: {missing}")


def build_gaussian_outputs(
    training_pits: pd.DataFrame,
    marginal_refits: pd.DataFrame,
    daily_margins: pd.DataFrame,
    group_returns: pd.DataFrame,
    model_config: Mapping[str, Any],
    *,
    universe_variant: str = "security_primary",
    progress_every: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Build all monthly Gaussian fits, daily forecasts, seeds, and quality checks."""

    gaussian, simulation, forecast, marginal = _validate_protocol(model_config)
    _validate_input_frames(training_pits, marginal_refits, daily_margins, group_returns)
    dimension = int(simulation["dimension"])
    draws = int(simulation["draws_per_month"])
    base_seed = int(simulation["base_seed"])
    lower = float(marginal["pit_clip_lower"])
    upper = float(marginal["pit_clip_upper"])
    confidence_levels = [float(value) for value in forecast["var_confidence_levels"]]
    es_confidence = float(forecast["es_confidence_level"])
    eigenvalue_floor = float(gaussian["eigenvalue_floor"])
    repair_tolerance = float(gaussian["repair_tolerance"])
    maximum_condition = float(gaussian["maximum_condition_number"])

    training = training_pits.loc[training_pits["universe_variant"] == universe_variant].copy()
    refits = marginal_refits.loc[marginal_refits["universe_variant"] == universe_variant].copy()
    daily = daily_margins.loc[daily_margins["universe_variant"] == universe_variant].copy()
    returns = group_returns.loc[group_returns["universe_variant"] == universe_variant].copy()
    for frame, columns in [
        (training, ["training_date", "refit_date"]),
        (refits, ["refit_date", "requested_training_start"]),
        (daily, ["date"]),
        (returns, ["date"]),
    ]:
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
            if frame[column].isna().any():
                raise ValueError(f"invalid date in {column}")
    if training.empty or refits.empty or daily.empty:
        raise ValueError(f"no modelling rows for universe variant {universe_variant}")
    if not set(training["grouping_id"]).issubset(MODEL_BY_GROUPING):
        raise ValueError("training PIT panel contains an unsupported grouping")
    training_key = [
        "year",
        "month",
        "universe_variant",
        "grouping_id",
        "training_date",
        "group_id",
    ]
    if training.duplicated(training_key).any():
        raise ValueError("training PIT panel contains duplicate logical rows")
    if bool((training["training_date"] >= training["refit_date"]).any()):
        raise ValueError("training PIT panel contains look-ahead observations")

    fit_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    seed_records: dict[tuple[int, int], dict[str, Any]] = {}
    issues: list[str] = []
    block_columns = ["year", "month", "universe_variant", "grouping_id"]
    grouped = training.groupby(block_columns, sort=True)
    for block_index, (identifiers, block) in enumerate(grouped, start=1):
        year, month, _, input_grouping = identifiers
        year, month = int(year), int(month)
        model_id, output_grouping = MODEL_BY_GROUPING[str(input_grouping)]
        group_order = sorted(str(value) for value in block["group_id"].unique())
        if len(group_order) != dimension:
            raise ValueError(f"{identifiers} has {len(group_order)} groups; expected {dimension}")
        pivot = block.pivot(index="training_date", columns="group_id", values="pit")
        pivot = pivot.reindex(columns=group_order).sort_index()
        if pivot.isna().any().any() or pivot.shape[1] != dimension:
            raise ValueError(f"{identifiers} does not form a complete PIT matrix")
        refit_dates = block["refit_date"].drop_duplicates()
        copula_ids = block["copula_refit_id"].drop_duplicates()
        if len(refit_dates) != 1 or len(copula_ids) != 1:
            raise ValueError(f"{identifiers} has inconsistent refit metadata")
        refit_date = pd.Timestamp(refit_dates.iloc[0])
        copula_refit_id = str(copula_ids.iloc[0])
        fit = fit_gaussian_copula(
            pivot.to_numpy(dtype=float),
            eigenvalue_floor=eigenvalue_floor,
            repair_tolerance=repair_tolerance,
        )
        if fit.condition_number > maximum_condition:
            issues.append(f"condition_number_exceeded:{copula_refit_id}")

        independent = common_uniforms(
            year,
            month,
            draws=draws,
            dimension=dimension,
            base_seed=base_seed,
        )
        uniform_hash = _array_sha256(independent)
        seed_key = (year, month)
        seed_record = {
            "year": year,
            "month": month,
            "seed_components": [base_seed, year, month],
            "draws": draws,
            "dimension": dimension,
            "base_uniform_sha256": uniform_hash,
        }
        previous_seed = seed_records.setdefault(seed_key, seed_record)
        if previous_seed != seed_record:
            raise AssertionError("common-random-number identity differs across groupings")
        dependent = gaussian_dependence_uniforms(
            independent,
            fit.correlation,
            lower=lower,
            upper=upper,
        )
        monthly_refits = refits.loc[
            (refits["year"] == year)
            & (refits["month"] == month)
            & (refits["grouping_id"] == input_grouping)
        ].copy()
        innovations = marginal_innovation_draws(
            dependent,
            group_order,
            monthly_refits,
            returns,
            marginal,
        )
        fallback_count = int((monthly_refits["fallback_level"] > 0).sum())
        fit_rows.append(
            {
                "copula_refit_id": copula_refit_id,
                "refit_date": refit_date,
                "year": year,
                "month": month,
                "universe_variant": universe_variant,
                "model_id": model_id,
                "grouping_id": output_grouping,
                "training_start": pivot.index.min(),
                "training_end": pivot.index.max(),
                "training_observations": len(pivot),
                "dimension": dimension,
                "group_order_json": json.dumps(group_order, separators=(",", ":")),
                "estimator": gaussian["estimator"],
                "correlation_matrix_json": json.dumps(
                    fit.correlation.tolist(), separators=(",", ":")
                ),
                "raw_min_eigenvalue": fit.raw_min_eigenvalue,
                "repaired_min_eigenvalue": fit.repaired_min_eigenvalue,
                "maximum_absolute_adjustment": fit.maximum_absolute_adjustment,
                "condition_number": fit.condition_number,
                "log_determinant": fit.log_determinant,
                "correlation_repaired": fit.correlation_repaired,
                "factorization": gaussian["factorization"],
                "seed_components": [base_seed, year, month],
                "base_uniform_sha256": uniform_hash,
                "fit_status": "repaired" if fit.correlation_repaired else "ok",
            }
        )

        daily_block = daily.loc[
            (daily["year"] == year)
            & (daily["month"] == month)
            & (daily["grouping_id"] == input_grouping)
        ].copy()
        if daily_block.empty:
            raise ValueError(f"no daily margins for {copula_refit_id}")
        for date, day in daily_block.groupby("date", sort=True):
            ordered = day.set_index("group_id", verify_integrity=True).reindex(group_order)
            if ordered.isna().any().any():
                raise ValueError(f"incomplete daily margin vector for {date}/{copula_refit_id}")
            means = ordered["conditional_mean_log_return"].to_numpy(dtype=float)
            volatilities = ordered["conditional_volatility_log_return"].to_numpy(dtype=float)
            weights = ordered["portfolio_weight"].to_numpy(dtype=float)
            realised_log = ordered["realised_log_return"].to_numpy(dtype=float)
            if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
                raise ValueError(f"portfolio weights do not sum to one for {date}/{model_id}")
            simulated_log = means + innovations * volatilities
            simulated_losses = -(np.expm1(simulated_log) @ weights)
            risks = empirical_risk_levels(simulated_losses, confidence_levels)
            realised_simple = float(np.expm1(realised_log) @ weights)
            observed_pits = ordered["pit"].to_numpy(dtype=float)
            log_score = float(gaussian_copula_log_density(observed_pits, fit.correlation)[0])
            forecast_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "model_id": model_id,
                    "grouping_id": output_grouping,
                    "refit_id": copula_refit_id,
                    "var_95": risks[0.95][0],
                    "var_975": risks[0.975][0],
                    "var_99": risks[0.99][0],
                    "es_975": risks[es_confidence][1],
                    "realised_simple_return": realised_simple,
                    "realised_loss": -realised_simple,
                    "seed_components": [base_seed, year, month],
                    "margin_fallback_count": fallback_count,
                    "whole_vine_fallback": False,
                    "copula_log_score": log_score,
                    "forecast_status": "fallback" if fallback_count else "ok",
                }
            )
        if progress_every and block_index % progress_every == 0:
            print(f"gaussian_progress={block_index}/{len(grouped)}")

    fit_frame = pd.DataFrame(fit_rows).sort_values(["year", "month", "model_id"])
    forecast_frame = pd.DataFrame(forecast_rows).sort_values(["date", "model_id"])
    forecast_key = ["date", "model_id"]
    if fit_frame.duplicated(["year", "month", "model_id"]).any():
        issues.append("duplicate_gaussian_refit")
    if forecast_frame.duplicated(forecast_key).any():
        issues.append("duplicate_gaussian_forecast")
    numeric = forecast_frame[
        [
            "var_95",
            "var_975",
            "var_99",
            "es_975",
            "realised_simple_return",
            "realised_loss",
            "copula_log_score",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        issues.append("nonfinite_gaussian_forecast")
    if bool(
        (
            (forecast_frame["var_95"] > forecast_frame["var_975"])
            | (forecast_frame["var_975"] > forecast_frame["var_99"])
        ).any()
    ):
        issues.append("nonmonotone_var_forecast")
    if bool((forecast_frame["es_975"] < forecast_frame["var_975"]).any()):
        issues.append("es_below_var")
    date_counts = {
        str(key): int(value)
        for key, value in forecast_frame.groupby("model_id")["date"].nunique().items()
    }
    expected_models = {"M1", "M3"}
    if set(date_counts) != expected_models or len(set(date_counts.values())) != 1:
        issues.append("incomplete_matched_model_coverage")
    realised_wide = forecast_frame.pivot(
        index="date", columns="model_id", values="realised_simple_return"
    )
    if realised_wide.isna().any().any() or not expected_models.issubset(realised_wide.columns):
        issues.append("incomplete_realised_portfolio_identity")
        maximum_identity_error = float("inf")
    else:
        maximum_identity_error = float(np.max(np.abs(realised_wide["M1"] - realised_wide["M3"])))
        if maximum_identity_error > float(model_config["portfolio"]["reconstruction_tolerance"]):
            issues.append("grouping_portfolio_identity_failed")
    repaired_count = int(fit_frame["correlation_repaired"].sum())
    seed_manifest = {
        "schema_version": 1,
        "manifest_type": "monthly_common_random_numbers",
        "bit_generator": simulation["bit_generator"],
        "base_seed": base_seed,
        "draws_per_month": draws,
        "dimension": dimension,
        "seed_sequence": "SeedSequence([base_seed, year, month])",
        "records": [seed_records[key] for key in sorted(seed_records)],
    }
    audit = {
        "schema_version": 1,
        "gate_name": "gaussian_copula_quality_v1",
        "status": "pass" if not issues else "fail",
        "universe_variant": universe_variant,
        "model_ids": sorted(expected_models),
        "refit_count": len(fit_frame),
        "forecast_count": len(forecast_frame),
        "evaluation_date_count": int(forecast_frame["date"].nunique()),
        "model_date_counts": date_counts,
        "dimension": dimension,
        "draws_per_month": draws,
        "common_random_number_month_count": len(seed_records),
        "repaired_refit_count": repaired_count,
        "maximum_absolute_correlation_adjustment": float(
            fit_frame["maximum_absolute_adjustment"].max()
        ),
        "minimum_repaired_eigenvalue": float(fit_frame["repaired_min_eigenvalue"].min()),
        "maximum_condition_number": float(fit_frame["condition_number"].max()),
        "maximum_allowed_condition_number": maximum_condition,
        "maximum_realised_portfolio_identity_error": maximum_identity_error,
        "portfolio_identity_tolerance": float(
            model_config["portfolio"]["reconstruction_tolerance"]
        ),
        "issues": issues,
    }
    return (
        fit_frame.reset_index(drop=True),
        forecast_frame.reset_index(drop=True),
        seed_manifest,
        audit,
    )


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
    return str(scope)


def _validate_marginal_binding(
    audit: Mapping[str, Any],
    *,
    training_pits: Path,
    marginal_refits: Path,
    daily_margins: Path,
    group_returns: Path,
    model_config: Path,
) -> None:
    if audit.get("status") != "pass":
        raise RuntimeError("marginal-model quality gate is not passed")
    expected_outputs = {
        "monthly_copula_training_pits": training_pits,
        "marginal_refits": marginal_refits,
        "marginal_daily_forecasts": daily_margins,
    }
    for name, path in expected_outputs.items():
        if audit.get("outputs", {}).get(name, {}).get("sha256") != _sha256(path):
            raise RuntimeError(f"{name} differs from the passed marginal audit")
    expected_inputs = {
        "annual_group_returns": group_returns,
        "model_config": model_config,
    }
    for name, path in expected_inputs.items():
        if audit.get("inputs", {}).get(name, {}).get("sha256") != _sha256(path):
            raise RuntimeError(f"{name} differs from the passed marginal audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-pits",
        type=Path,
        default=PROJECT_ROOT / "data/processed/monthly_copula_training_pits.parquet",
    )
    parser.add_argument(
        "--marginal-refits",
        type=Path,
        default=PROJECT_ROOT / "data/processed/marginal_refits.parquet",
    )
    parser.add_argument(
        "--daily-margins",
        type=Path,
        default=PROJECT_ROOT / "data/processed/marginal_daily_forecasts.parquet",
    )
    parser.add_argument(
        "--group-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/annual_group_returns.parquet",
    )
    parser.add_argument(
        "--marginal-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/marginal_model_quality.json",
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
        "--refits-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/gaussian_copula_refits.parquet",
    )
    parser.add_argument(
        "--forecasts-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/gaussian_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--seed-manifest-output",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/simulation_seed_manifest.json",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/gaussian_copula_quality.json",
    )
    parser.add_argument("--universe-variant", default="security_primary")
    parser.add_argument("--progress-every", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    _validate_protocol(model_config)
    gate = json.loads(args.foundation_status.read_text(encoding="utf-8"))
    scope = _reporting_scope(gate)
    marginal_audit = json.loads(args.marginal_audit.read_text(encoding="utf-8"))
    _validate_marginal_binding(
        marginal_audit,
        training_pits=args.training_pits,
        marginal_refits=args.marginal_refits,
        daily_margins=args.daily_margins,
        group_returns=args.group_returns,
        model_config=args.model_config,
    )
    training = pd.read_parquet(args.training_pits)
    refits = pd.read_parquet(args.marginal_refits)
    daily = pd.read_parquet(args.daily_margins)
    returns = pd.read_parquet(args.group_returns)
    gaussian_refits, forecasts, seed_manifest, audit = build_gaussian_outputs(
        training,
        refits,
        daily,
        returns,
        model_config,
        universe_variant=args.universe_variant,
        progress_every=args.progress_every,
    )
    _write_parquet_atomic(args.refits_output, gaussian_refits)
    _write_parquet_atomic(args.forecasts_output, forecasts)
    _write_json_atomic(args.seed_manifest_output, seed_manifest)
    audit.update(
        {
            "reporting_scope": scope,
            "method": {
                "estimator": model_config["gaussian"]["estimator"],
                "repair_method": model_config["gaussian"]["repair_method"],
                "eigenvalue_floor": model_config["gaussian"]["eigenvalue_floor"],
                "factorization": model_config["gaussian"]["factorization"],
                "simulation_bit_generator": model_config["simulation"]["bit_generator"],
            },
            "inputs": {
                "monthly_copula_training_pits": {
                    "path": _project_path(args.training_pits),
                    "sha256": _sha256(args.training_pits),
                },
                "marginal_refits": {
                    "path": _project_path(args.marginal_refits),
                    "sha256": _sha256(args.marginal_refits),
                },
                "marginal_daily_forecasts": {
                    "path": _project_path(args.daily_margins),
                    "sha256": _sha256(args.daily_margins),
                },
                "annual_group_returns": {
                    "path": _project_path(args.group_returns),
                    "sha256": _sha256(args.group_returns),
                },
                "marginal_model_quality": {
                    "path": _project_path(args.marginal_audit),
                    "sha256": _sha256(args.marginal_audit),
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
                "gaussian_copula_refits": {
                    "path": _project_path(args.refits_output),
                    "sha256": _sha256(args.refits_output),
                    "rows": len(gaussian_refits),
                },
                "gaussian_risk_forecasts": {
                    "path": _project_path(args.forecasts_output),
                    "sha256": _sha256(args.forecasts_output),
                    "rows": len(forecasts),
                },
                "simulation_seed_manifest": {
                    "path": _project_path(args.seed_manifest_output),
                    "sha256": _sha256(args.seed_manifest_output),
                    "records": len(seed_manifest["records"]),
                },
            },
        }
    )
    _write_json_atomic(args.audit_output, audit)
    print(
        f"gaussian_copula_quality={audit['status']} refits={len(gaussian_refits)} "
        f"forecasts={len(forecasts)} repaired={audit['repaired_refit_count']}"
    )
    print(f"Audit: {args.audit_output}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
