"""Frozen, result-agnostic methods for the FE5110 research protocol.

This module deliberately contains no data-loading entry point and does not fit any
out-of-sample model.  It centralises arithmetic and statistical definitions that
must be fixed before those results are generated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupReturns:
    simple_returns: pd.DataFrame
    group_sizes: pd.Series


@dataclass(frozen=True)
class PortfolioReconstruction:
    direct_simple_return: pd.Series
    grouped_simple_return: pd.Series
    absolute_error: pd.Series


@dataclass(frozen=True)
class DependenceGap:
    within_pair_mean: float
    between_pair_mean: float
    gap: float
    within_pair_count: int
    between_pair_count: int
    contributing_within_groups: int
    group_balanced_within_mean: float
    group_balanced_between_mean: float
    group_balanced_gap: float


@dataclass(frozen=True)
class HistoricalRisk:
    confidence: float
    var: float
    es: float
    observations: int


@dataclass(frozen=True)
class DMResult:
    mean_difference: float
    statistic: float
    p_value_two_sided: float
    hac_lag: int
    observations: int


@dataclass(frozen=True)
class CoverageResult:
    observations: int
    exceptions: int
    exception_rate: float
    likelihood_ratio: float
    p_value: float


@dataclass(frozen=True)
class IndependenceResult:
    transitions_00: int
    transitions_01: int
    transitions_10: int
    transitions_11: int
    likelihood_ratio: float
    p_value: float


def _as_float_array(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def simple_returns_from_log(log_returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Convert log returns to simple returns while preserving pandas labels."""

    values = np.expm1(log_returns)
    if np.isinf(np.asarray(values, dtype=float)).any():
        raise ValueError("log returns produced non-finite simple returns")
    return values


def group_simple_returns(
    stock_simple_returns: pd.DataFrame,
    labels: Mapping[str, str],
) -> GroupReturns:
    """Form equal-weight group simple returns for a complete active panel."""

    missing_labels = sorted(set(stock_simple_returns.columns) - set(labels))
    extra_labels = sorted(set(labels) - set(stock_simple_returns.columns))
    if missing_labels or extra_labels:
        raise ValueError(
            f"labels and return columns differ; missing={missing_labels}, extra={extra_labels}"
        )
    values = stock_simple_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("active stock-return panel must be complete and finite")
    ordered_groups = list(dict.fromkeys(labels[column] for column in stock_simple_returns.columns))
    grouped: dict[str, pd.Series] = {}
    sizes: dict[str, int] = {}
    for group in ordered_groups:
        members = [column for column in stock_simple_returns.columns if labels[column] == group]
        if not members:
            continue
        grouped[group] = stock_simple_returns[members].mean(axis=1)
        sizes[group] = len(members)
    return GroupReturns(
        simple_returns=pd.DataFrame(grouped, index=stock_simple_returns.index),
        group_sizes=pd.Series(sizes, dtype="int64", name="group_size"),
    )


def reconstruct_primary_portfolio(
    stock_simple_returns: pd.DataFrame,
    labels: Mapping[str, str],
    *,
    tolerance: float = 1e-12,
) -> PortfolioReconstruction:
    """Verify the group-size identity for the daily-rebalanced portfolio."""

    grouped = group_simple_returns(stock_simple_returns, labels)
    direct = stock_simple_returns.mean(axis=1).rename("direct_simple_return")
    weights = grouped.group_sizes / grouped.group_sizes.sum()
    reconstructed = grouped.simple_returns.mul(weights, axis="columns").sum(axis=1)
    reconstructed = reconstructed.rename("grouped_simple_return")
    error = (direct - reconstructed).abs().rename("absolute_error")
    if bool((error > tolerance).any()):
        worst = float(error.max())
        raise AssertionError(f"portfolio reconstruction error {worst} exceeds {tolerance}")
    return PortfolioReconstruction(direct, reconstructed, error)


def group_log_returns(group_returns: GroupReturns | pd.DataFrame) -> pd.DataFrame:
    """Transform group simple returns for marginal-model estimation."""

    simple = group_returns.simple_returns if isinstance(group_returns, GroupReturns) else group_returns
    if bool((simple <= -1.0).any().any()):
        raise ValueError("simple returns must be greater than -1 before log1p")
    result = np.log1p(simple)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("group log returns must be finite")
    return result


def realised_loss(portfolio_simple_return: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Apply the frozen positive-loss convention L = -R."""

    return -portfolio_simple_return


def annual_buy_and_hold_returns(
    stock_simple_returns: pd.DataFrame,
    initial_weights: Mapping[str, float],
) -> pd.Series:
    """Return a secondary portfolio with weights drifting after initialisation."""

    columns = list(stock_simple_returns.columns)
    if set(columns) != set(initial_weights):
        raise ValueError("initial weights must match stock-return columns")
    values = stock_simple_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or bool((values <= -1.0).any()):
        raise ValueError("buy-and-hold inputs must be finite simple returns greater than -1")
    weights = np.asarray([initial_weights[column] for column in columns], dtype=float)
    if not np.isfinite(weights).all() or bool((weights < 0).any()) or not math.isclose(
        float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("initial weights must be nonnegative and sum to one")
    portfolio = np.empty(len(stock_simple_returns), dtype=float)
    for row_number, row in enumerate(values):
        period_return = float(weights @ row)
        portfolio[row_number] = period_return
        wealth_multiplier = 1.0 + period_return
        if wealth_multiplier <= 0:
            raise ValueError("portfolio wealth became nonpositive")
        weights = weights * (1.0 + row) / wealth_multiplier
    return pd.Series(portfolio, index=stock_simple_returns.index, name="simple_return")


def pairwise_spearman(
    returns: pd.DataFrame,
    *,
    minimum_paired_fraction: float = 0.80,
) -> pd.DataFrame:
    """Compute pairwise Spearman correlations with an explicit completeness rule."""

    if not 0 < minimum_paired_fraction <= 1:
        raise ValueError("minimum_paired_fraction must lie in (0, 1]")
    columns = list(returns.columns)
    result = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    minimum = math.ceil(len(returns) * minimum_paired_fraction)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            pair = returns[[left, right]].dropna()
            if len(pair) < minimum:
                raise ValueError(
                    f"pair {left}/{right} has {len(pair)} observations; requires {minimum}"
                )
            rho = float(pair[left].rank(method="average").corr(pair[right].rank(method="average")))
            if not math.isfinite(rho):
                raise ValueError(f"pair {left}/{right} has undefined Spearman correlation")
            result.loc[left, right] = rho
            result.loc[right, left] = rho
    return result


def dependence_gap(correlation: pd.DataFrame, labels: Mapping[str, str]) -> DependenceGap:
    """Calculate pair-weighted primary and equal-group robustness gaps."""

    columns = list(correlation.columns)
    if list(correlation.index) != columns:
        raise ValueError("correlation matrix must be square with identically ordered labels")
    if set(columns) != set(labels):
        raise ValueError("group labels must match the correlation matrix")
    matrix = correlation.to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("correlation matrix must be finite and symmetric")

    within: list[float] = []
    between: list[float] = []
    within_by_group: dict[str, list[float]] = {}
    between_by_groups: dict[tuple[str, str], list[float]] = {}
    for left_index, left in enumerate(columns):
        for right_index in range(left_index + 1, len(columns)):
            right = columns[right_index]
            value = float(matrix[left_index, right_index])
            left_group, right_group = labels[left], labels[right]
            if left_group == right_group:
                within.append(value)
                within_by_group.setdefault(left_group, []).append(value)
            else:
                between.append(value)
                key = tuple(sorted((left_group, right_group)))
                between_by_groups.setdefault(key, []).append(value)
    if not within or not between:
        raise ValueError("dependence gap requires at least one within and one between pair")
    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))
    balanced_within = float(np.mean([np.mean(values) for values in within_by_group.values()]))
    balanced_between = float(np.mean([np.mean(values) for values in between_by_groups.values()]))
    return DependenceGap(
        within_pair_mean=within_mean,
        between_pair_mean=between_mean,
        gap=within_mean - between_mean,
        within_pair_count=len(within),
        between_pair_count=len(between),
        contributing_within_groups=len(within_by_group),
        group_balanced_within_mean=balanced_within,
        group_balanced_between_mean=balanced_between,
        group_balanced_gap=balanced_within - balanced_between,
    )


def alphabet_issuer_composite(
    simple_returns: pd.DataFrame,
    labels: Mapping[str, str],
    *,
    composite_name: str = "ALPHABET",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Replace GOOG/GOOGL with one equal-weight simple-return issuer position."""

    required = {"GOOG", "GOOGL"}
    if not required.issubset(simple_returns.columns) or not required.issubset(labels):
        raise ValueError("GOOG and GOOGL must both be present")
    if labels["GOOG"] != labels["GOOGL"]:
        raise ValueError("Alphabet share classes must have the same group label")
    result = simple_returns.drop(columns=["GOOG", "GOOGL"]).copy()
    result[composite_name] = simple_returns[["GOOG", "GOOGL"]].mean(axis=1)
    result_labels = {key: value for key, value in labels.items() if key not in required}
    result_labels[composite_name] = labels["GOOG"]
    return result, result_labels


def quantile_loss(loss: object, var: object, confidence: float) -> np.ndarray:
    """Return the frozen pinball loss for positive portfolio losses."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    realised = _as_float_array(loss, name="loss")
    forecast = _as_float_array(var, name="var")
    realised, forecast = np.broadcast_arrays(realised, forecast)
    return (confidence - (realised < forecast).astype(float)) * (realised - forecast)


def fz0_loss(loss: object, var: object, es: object, confidence: float = 0.975) -> np.ndarray:
    """Return the upper-tail, positive-loss FZ0 score."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    realised = _as_float_array(loss, name="loss")
    quantile = _as_float_array(var, name="var")
    expected_shortfall = _as_float_array(es, name="es")
    realised, quantile, expected_shortfall = np.broadcast_arrays(
        realised, quantile, expected_shortfall
    )
    if bool((expected_shortfall <= 0).any()):
        raise ValueError("FZ0 requires strictly positive ES under the loss convention")
    if bool((expected_shortfall < quantile).any()):
        raise ValueError("ES must be greater than or equal to VaR")
    tail_probability = 1.0 - confidence
    exceedance = (realised > quantile).astype(float)
    return (
        exceedance * (realised - quantile) / (tail_probability * expected_shortfall)
        + quantile / expected_shortfall
        + np.log(expected_shortfall)
        - 1.0
    )


def empirical_var_es(losses: object, confidence: float) -> HistoricalRisk:
    """Historical VaR and fractionally weighted ES from an empirical loss sample."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    values = np.sort(_as_float_array(losses, name="losses").reshape(-1))
    if len(values) == 0:
        raise ValueError("losses cannot be empty")
    order = math.ceil(len(values) * confidence)
    var = float(values[order - 1])
    tail_mass = len(values) * (1.0 - confidence)
    boundary_weight = order - len(values) * confidence
    es = float((values[order:].sum() + boundary_weight * var) / tail_mass)
    return HistoricalRisk(confidence, var, es, len(values))


def rolling_historical_var_es(
    losses: pd.Series,
    forecast_date: pd.Timestamp | str,
    confidence: float,
    *,
    calendar_years: int = 3,
    minimum_observations: int = 700,
) -> HistoricalRisk:
    """Apply the exact left-closed, right-open historical-simulation window."""

    if not isinstance(losses.index, pd.DatetimeIndex):
        raise TypeError("loss series must use a DatetimeIndex")
    forecast = pd.Timestamp(forecast_date)
    start = forecast - pd.DateOffset(years=calendar_years)
    sample = losses.loc[(losses.index >= start) & (losses.index < forecast)].dropna()
    if len(sample) < minimum_observations:
        raise ValueError(
            f"historical simulation has {len(sample)} observations; requires {minimum_observations}"
        )
    return empirical_var_es(sample.to_numpy(), confidence)


def common_uniforms(
    year: int,
    month: int,
    *,
    draws: int = 100_000,
    dimension: int = 11,
    base_seed: int = 5110,
) -> np.ndarray:
    """Create the monthly common-random-number matrix fixed by the protocol."""

    if draws <= 0 or dimension <= 0 or not 1 <= month <= 12:
        raise ValueError("draws/dimension must be positive and month must be in 1..12")
    seed = np.random.SeedSequence([base_seed, int(year), int(month)])
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    return generator.random((draws, dimension), dtype=np.float64)


def var_exceptions(loss: object, var: object) -> np.ndarray:
    """Apply the frozen strict exceedance rule, L > VaR."""

    realised = _as_float_array(loss, name="loss")
    forecast = _as_float_array(var, name="var")
    realised, forecast = np.broadcast_arrays(realised, forecast)
    return realised > forecast


def _bernoulli_log_likelihood(successes: int, trials: int, probability: float) -> float:
    failures = trials - successes
    if not 0 <= successes <= trials or not 0 <= probability <= 1:
        raise ValueError("invalid Bernoulli likelihood arguments")
    value = 0.0
    if successes:
        if probability == 0:
            return -math.inf
        value += successes * math.log(probability)
    if failures:
        if probability == 1:
            return -math.inf
        value += failures * math.log1p(-probability)
    return value


def kupiec_unconditional_coverage(
    exceptions: object,
    confidence: float,
) -> CoverageResult:
    """Kupiec likelihood-ratio test for the nominal VaR exception rate."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    indicators = np.asarray(exceptions, dtype=bool).reshape(-1)
    if len(indicators) == 0:
        raise ValueError("exception sequence cannot be empty")
    count = int(indicators.sum())
    nominal = 1.0 - confidence
    empirical = count / len(indicators)
    null_ll = _bernoulli_log_likelihood(count, len(indicators), nominal)
    alternative_ll = _bernoulli_log_likelihood(count, len(indicators), empirical)
    likelihood_ratio = max(0.0, -2.0 * (null_ll - alternative_ll))
    p_value = math.erfc(math.sqrt(likelihood_ratio / 2.0))
    return CoverageResult(len(indicators), count, empirical, likelihood_ratio, p_value)


def christoffersen_independence(exceptions: object) -> IndependenceResult:
    """Christoffersen first-order exception-independence likelihood-ratio test."""

    indicators = np.asarray(exceptions, dtype=bool).reshape(-1)
    if len(indicators) < 2:
        raise ValueError("independence test requires at least two observations")
    previous, current = indicators[:-1], indicators[1:]
    n00 = int((~previous & ~current).sum())
    n01 = int((~previous & current).sum())
    n10 = int((previous & ~current).sum())
    n11 = int((previous & current).sum())
    total = n00 + n01 + n10 + n11
    total_exceptions = n01 + n11
    restricted_probability = total_exceptions / total
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    restricted_ll = _bernoulli_log_likelihood(total_exceptions, total, restricted_probability)
    unrestricted_ll = _bernoulli_log_likelihood(n01, n00 + n01, p01)
    unrestricted_ll += _bernoulli_log_likelihood(n11, n10 + n11, p11)
    likelihood_ratio = max(0.0, -2.0 * (restricted_ll - unrestricted_ll))
    p_value = math.erfc(math.sqrt(likelihood_ratio / 2.0))
    return IndependenceResult(n00, n01, n10, n11, likelihood_ratio, p_value)


def christoffersen_conditional_coverage_p_value(
    coverage: CoverageResult,
    independence: IndependenceResult,
) -> float:
    """Return the chi-square(2) p-value for the reported conditional-coverage LR."""

    likelihood_ratio = coverage.likelihood_ratio + independence.likelihood_ratio
    return math.exp(-likelihood_ratio / 2.0)


def diebold_mariano(
    loss_differential: object,
    *,
    hac_lag: int = 7,
) -> DMResult:
    """Two-sided DM statistic using a Bartlett/Newey-West long-run variance."""

    differential = _as_float_array(loss_differential, name="loss differential").reshape(-1)
    if len(differential) <= hac_lag or hac_lag < 0:
        raise ValueError("DM sample must be longer than the nonnegative HAC lag")
    centred = differential - differential.mean()
    sample_size = len(differential)
    long_run_variance = float(centred @ centred / sample_size)
    for lag in range(1, hac_lag + 1):
        covariance = float(centred[lag:] @ centred[:-lag] / sample_size)
        bartlett_weight = 1.0 - lag / (hac_lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * covariance
    if not math.isfinite(long_run_variance) or long_run_variance <= 0:
        raise ValueError("DM long-run variance is not positive")
    statistic = float(differential.mean() / math.sqrt(long_run_variance / sample_size))
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return DMResult(
        mean_difference=float(differential.mean()),
        statistic=statistic,
        p_value_two_sided=p_value,
        hac_lag=hac_lag,
        observations=sample_size,
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values in original order."""

    values = _as_float_array(p_values, name="p-values").reshape(-1)
    if bool(((values < 0) | (values > 1)).any()):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (len(values) - np.arange(len(values))) * sorted_values)
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def benjamini_hochberg_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p-values in original order."""

    values = _as_float_array(p_values, name="p-values").reshape(-1)
    if bool(((values < 0) | (values > 1)).any()):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.arange(1, len(values) + 1)
    raw = sorted_values * len(values) / ranks
    adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted_sorted = np.minimum(1.0, adjusted_sorted)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def circular_block_bootstrap_indices(
    dates: pd.DatetimeIndex,
    *,
    replications: int = 10_000,
    block_length: int = 20,
    base_seed: int = 5110,
) -> np.ndarray:
    """Generate paired circular block indices without crossing year boundaries."""

    if replications <= 0 or block_length <= 0 or len(dates) == 0:
        raise ValueError("dates, replications, and block_length must be positive")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dates must be unique and increasing")
    generator = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([base_seed, 1])))
    result = np.empty((replications, len(dates)), dtype=np.int32)
    years = np.asarray(dates.year)
    for replication in range(replications):
        write_at = 0
        for year in dict.fromkeys(years.tolist()):
            positions = np.flatnonzero(years == year)
            count = len(positions)
            sampled: list[int] = []
            while len(sampled) < count:
                start = int(generator.integers(0, count))
                sampled.extend(positions[(start + np.arange(block_length)) % count].tolist())
            result[replication, write_at : write_at + count] = sampled[:count]
            write_at += count
    return result


def margin_fit_is_acceptable(
    *,
    phi: float,
    omega: float,
    alpha: float,
    beta: float,
    student_t_df: float,
    forecast_variance: float,
) -> bool:
    """Apply the preregistered numerical/stability constraints to a margin fit."""

    values = np.asarray([phi, omega, alpha, beta, student_t_df, forecast_variance], dtype=float)
    return bool(
        np.isfinite(values).all()
        and abs(phi) < 0.98
        and omega > 0
        and alpha >= 0
        and beta >= 0
        and alpha + beta < 0.999
        and student_t_df > 2.1
        and forecast_variance > 0
    )


def fallback_fraction_passes(fallback_count: int, total_count: int, *, maximum: float = 0.01) -> bool:
    """Evaluate a modelling fallback-rate quality gate without rounding."""

    if total_count <= 0 or fallback_count < 0 or fallback_count > total_count:
        raise ValueError("invalid fallback counts")
    return fallback_count / total_count <= maximum
