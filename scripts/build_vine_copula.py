#!/usr/bin/env python3
"""Fit primary or full robustness R-vines and produce matched M2/M4 forecasts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyvinecopulib as pv

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.build_gaussian_copula import (
        _array_sha256,
        _validate_block_bindings,
        _validate_input_frames,
        _validate_marginal_binding,
        empirical_risk_levels,
        gaussian_copula_log_density,
        gaussian_dependence_uniforms,
        marginal_innovation_draws,
    )
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path as _project_path,
        reporting_scope as _reporting_scope,
        require_current_hash_records as _require_current_hash_records,
        sha256_file as _sha256,
        write_json_atomic as _write_json_atomic,
        write_parquet_atomic as _write_parquet_atomic,
    )
    from scripts.research_methods import common_uniforms
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from build_gaussian_copula import (
        _array_sha256,
        _validate_block_bindings,
        _validate_input_frames,
        _validate_marginal_binding,
        empirical_risk_levels,
        gaussian_copula_log_density,
        gaussian_dependence_uniforms,
        marginal_innovation_draws,
    )
    from pipeline_io import (
        PROJECT_ROOT,
        project_path as _project_path,
        reporting_scope as _reporting_scope,
        require_current_hash_records as _require_current_hash_records,
        sha256_file as _sha256,
        write_json_atomic as _write_json_atomic,
        write_parquet_atomic as _write_parquet_atomic,
    )
    from research_methods import common_uniforms


VINE_MODEL_BY_GROUPING = {
    "gics_sector": ("M2", "gics", "M1"),
    "hierarchical_cluster": ("M4", "hierarchical", "M3"),
}
PRIMARY_OUTPUTS = {
    "refits": PROJECT_ROOT / "data/processed/vine_copula_refits.parquet",
    "forecasts": PROJECT_ROOT / "data/processed/vine_risk_forecasts.parquet",
    "audit": PROJECT_ROOT / "data/audit/vine_copula_quality.json",
}
ROBUSTNESS_OUTPUTS = {
    "refits": PROJECT_ROOT / "data/processed/full_vine_copula_refits.parquet",
    "forecasts": PROJECT_ROOT / "data/processed/full_vine_risk_forecasts.parquet",
    "audit": PROJECT_ROOT / "data/audit/full_vine_robustness_quality.json",
}
FAMILY_BY_CONFIG = {
    "independence": pv.BicopFamily.indep,
    "gaussian": pv.BicopFamily.gaussian,
    "student_t": pv.BicopFamily.student,
    "frank": pv.BicopFamily.frank,
    "clayton": pv.BicopFamily.clayton,
    "gumbel": pv.BicopFamily.gumbel,
}
ROTATIONLESS_FAMILIES = {
    pv.BicopFamily.indep,
    pv.BicopFamily.gaussian,
    pv.BicopFamily.student,
    pv.BicopFamily.frank,
}


@dataclass(frozen=True)
class VineFitResult:
    model: pv.Vinecop
    pair_count: int
    failed_pairs: tuple[dict[str, Any], ...]
    family_counts: dict[str, int]
    rotation_counts: dict[str, int]
    loglikelihood: float
    aic: float
    bic: float

    @property
    def failed_pair_count(self) -> int:
        return len(self.failed_pairs)

    @property
    def failed_pair_fraction(self) -> float:
        return self.failed_pair_count / self.pair_count


VineFitFunction = Callable[..., VineFitResult]


class VineFitError(RuntimeError):
    """Expected numerical or engine failure while fitting a vine model."""


class VineEvaluationError(RuntimeError):
    """Expected numerical or engine failure while evaluating a fitted vine."""


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


def _vine_protocol_mismatches(
    vine: Mapping[str, Any], portfolio: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    expected_vine = {
        "structure_selection": "dissmann_abs_kendall_tau_mst",
        "family_selection": "aic_sequential_mle",
        "pair_failure_action": "independence",
        "monthly_structure_failure_action": "matched_gaussian",
        "families": list(FAMILY_BY_CONFIG),
        "rotations_degrees": [0, 90, 180, 270],
    }
    expected_portfolio = {
        "primary_return_scale": "simple",
        "primary_rebalancing": "daily",
        "group_weighting": "group_size",
        "realised_loss_definition": "negative_simple_portfolio_return",
    }
    observed_vine = {
        **vine,
        "families": list(vine.get("families", [])),
        "rotations_degrees": list(vine.get("rotations_degrees", [])),
    }
    mismatches = {
        key: {"expected": expected, "observed": observed_vine.get(key)}
        for key, expected in expected_vine.items()
        if observed_vine.get(key) != expected
    }
    mismatches.update(
        {
            f"portfolio.{key}": {"expected": expected, "observed": portfolio.get(key)}
            for key, expected in expected_portfolio.items()
            if portfolio.get(key) != expected
        }
    )
    return mismatches


def _validate_vine_numerical_protocol(
    vine: Mapping[str, Any],
    simulation: Mapping[str, Any],
    forecast: Mapping[str, Any],
) -> None:
    lower = _finite_float(vine, "pit_clip_lower", "vine")
    upper = _finite_float(vine, "pit_clip_upper", "vine")
    failed_pair_limit = _finite_float(vine, "maximum_failed_pair_fraction", "vine")
    fallback_limit = _finite_float(vine, "maximum_whole_vine_fallback_date_fraction", "vine")
    checks = (
        (
            [float(value) for value in forecast.get("var_confidence_levels", [])]
            == [0.95, 0.975, 0.99],
            "forecast confidence levels differ from the frozen protocol",
        ),
        (
            float(forecast.get("es_confidence_level", float("nan"))) == 0.975,
            "forecast ES confidence level must be 0.975",
        ),
        (
            _positive_integer(simulation, "base_seed", "simulation") == 5110,
            "simulation.base_seed must remain 5110",
        ),
        (
            simulation.get("bit_generator") == "PCG64DXSM",
            "simulation bit generator differs from the frozen protocol",
        ),
        (
            simulation.get("seed_components") == ["base_seed", "year", "month"],
            "simulation seed components differ from the frozen protocol",
        ),
        (
            simulation.get("common_random_numbers") is True,
            "common random numbers must remain enabled",
        ),
        (
            _positive_integer(simulation, "draws_per_month", "simulation") >= 100,
            "simulation.draws_per_month must be at least 100",
        ),
        (0 < lower < upper < 1, "vine PIT bounds must lie strictly inside (0, 1)"),
        (
            _finite_float(vine, "student_t_df_minimum", "vine") == 2.1,
            "vine Student-t lower bound must remain 2.1",
        ),
        (
            _finite_float(vine, "student_t_df_maximum", "vine") == 50.0,
            "vine Student-t upper bound must remain 50",
        ),
        (0 <= failed_pair_limit <= 1, "invalid maximum failed-pair fraction"),
        (0 <= fallback_limit <= 1, "invalid maximum whole-vine fallback fraction"),
    )
    for passed, message in checks:
        if not passed:
            raise ValueError(message)


def validate_vine_protocol(
    model_config: Mapping[str, Any],
    *,
    truncation_level: int | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], int]:
    """Validate the frozen vine, simulation, forecast, and portfolio protocol."""

    vine = model_config.get("vine")
    simulation = model_config.get("simulation")
    forecast = model_config.get("forecast")
    marginal = model_config.get("marginal")
    portfolio = model_config.get("portfolio")
    sections = [vine, simulation, forecast, marginal, portfolio]
    if not all(isinstance(section, Mapping) for section in sections):
        raise TypeError("model configuration is missing a required modelling table")
    vine = cast(Mapping[str, Any], vine)
    simulation = cast(Mapping[str, Any], simulation)
    forecast = cast(Mapping[str, Any], forecast)
    marginal = cast(Mapping[str, Any], marginal)
    portfolio = cast(Mapping[str, Any], portfolio)

    mismatches = _vine_protocol_mismatches(vine, portfolio)
    if mismatches:
        raise ValueError(f"unsupported vine protocol: {mismatches}")

    dimension = _positive_integer(vine, "dimension", "vine")
    if dimension != _positive_integer(simulation, "dimension", "simulation"):
        raise ValueError("vine and simulation dimensions must agree")
    primary = _positive_integer(vine, "primary_truncation_tree", "vine")
    robustness = _positive_integer(vine, "robustness_truncation_tree", "vine")
    selected_truncation = primary if truncation_level is None else truncation_level
    if selected_truncation not in {primary, robustness} or selected_truncation >= dimension:
        raise ValueError("truncation level is not one of the frozen protocol levels")
    _validate_vine_numerical_protocol(vine, simulation, forecast)
    return vine, simulation, forecast, marginal, selected_truncation


def vine_analysis_role(vine: Mapping[str, Any], truncation_level: int) -> str:
    """Classify a frozen truncation level without inspecting model results."""

    primary = int(vine["primary_truncation_tree"])
    robustness = int(vine["robustness_truncation_tree"])
    if truncation_level == primary:
        return "primary"
    if truncation_level == robustness:
        return "robustness"
    raise ValueError("truncation level is not one of the frozen protocol levels")


def resolve_output_paths(
    vine: Mapping[str, Any],
    truncation_level: int,
    *,
    refits_output: Path | None,
    forecasts_output: Path | None,
    audit_output: Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve role-specific outputs and protect primary artifacts from robustness runs."""

    role = vine_analysis_role(vine, truncation_level)
    defaults = PRIMARY_OUTPUTS if role == "primary" else ROBUSTNESS_OUTPUTS
    resolved = (
        refits_output or defaults["refits"],
        forecasts_output or defaults["forecasts"],
        audit_output or defaults["audit"],
    )
    canonical = tuple(path.resolve() for path in resolved)
    if len(set(canonical)) != len(canonical):
        raise ValueError("vine refit, forecast, and audit outputs must be distinct")
    if role == "robustness":
        protected = {path.resolve() for path in PRIMARY_OUTPUTS.values()}
        collisions = sorted(str(path) for path in set(canonical) & protected)
        if collisions:
            raise ValueError(f"robustness run cannot overwrite primary artifacts: {collisions}")
    return resolved


def _pair_failure(
    pair: pv.Bicop,
    *,
    tree: int,
    edge: int,
    allowed_families: set[pv.BicopFamily],
    allowed_rotations: set[int],
    student_df_minimum: float,
    student_df_maximum: float,
) -> dict[str, Any] | None:
    family = pair.family
    rotation = int(pair.rotation)
    parameters = np.asarray(pair.parameters, dtype=float)
    reason: str | None = None
    if family not in allowed_families:
        reason = "family_outside_frozen_set"
    elif rotation not in allowed_rotations:
        reason = "rotation_outside_frozen_set"
    elif family in ROTATIONLESS_FAMILIES and rotation != 0:
        reason = "rotation_not_defined_for_family"
    elif not np.isfinite(parameters).all():
        reason = "nonfinite_parameters"
    elif family == pv.BicopFamily.student:
        degrees_of_freedom = float(parameters[1, 0])
        if not student_df_minimum <= degrees_of_freedom <= student_df_maximum:
            reason = "student_t_df_outside_frozen_bounds"
    if reason is None:
        return None
    return {
        "tree": tree + 1,
        "edge": edge + 1,
        "family": family.name,
        "rotation": rotation,
        "reason": reason,
    }


def _sanitize_selected_pairs(
    model: pv.Vinecop,
    vine: Mapping[str, Any],
) -> tuple[pv.Vinecop, tuple[dict[str, Any], ...]]:
    pair_copulas = [list(tree) for tree in model.pair_copulas]
    failures: list[dict[str, Any]] = []
    allowed_families = set(FAMILY_BY_CONFIG.values())
    allowed_rotations = {int(value) for value in vine["rotations_degrees"]}
    for tree_index, tree in enumerate(pair_copulas):
        for edge_index, pair in enumerate(tree):
            failure = _pair_failure(
                pair,
                tree=tree_index,
                edge=edge_index,
                allowed_families=allowed_families,
                allowed_rotations=allowed_rotations,
                student_df_minimum=float(vine["student_t_df_minimum"]),
                student_df_maximum=float(vine["student_t_df_maximum"]),
            )
            if failure is not None:
                failures.append(failure)
                pair_copulas[tree_index][edge_index] = pv.Bicop(family=pv.BicopFamily.indep)
    if failures:
        model = pv.Vinecop.from_structure(
            structure=model.structure,
            pair_copulas=pair_copulas,
        )
    return model, tuple(failures)


def _model_counts(model: pv.Vinecop) -> tuple[dict[str, int], dict[str, int]]:
    families: Counter[str] = Counter()
    rotations: Counter[str] = Counter()
    for tree in model.pair_copulas:
        for pair in tree:
            families[pair.family.name] += 1
            rotations[str(int(pair.rotation))] += 1
    return dict(sorted(families.items())), dict(sorted(rotations.items()))


def fit_vine_copula(
    pits: np.ndarray,
    vine: Mapping[str, Any],
    *,
    truncation_level: int,
    fit_seeds: Sequence[int],
) -> VineFitResult:
    """Fit and validate one Dißmann R-vine under the frozen family search."""

    values = np.asarray(pits, dtype=float)
    dimension = int(vine["dimension"])
    lower = float(vine["pit_clip_lower"])
    upper = float(vine["pit_clip_upper"])
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != dimension:
        raise ValueError("vine PIT matrix has invalid dimensions")
    if not np.isfinite(values).all() or bool(((values < lower) | (values > upper)).any()):
        raise ValueError("vine PIT matrix is non-finite or outside the frozen bounds")
    try:
        controls = pv.FitControlsVinecop(
            family_set=[FAMILY_BY_CONFIG[name] for name in vine["families"]],
            parametric_method="mle",
            trunc_lvl=truncation_level,
            tree_criterion="tau",
            selection_criterion="aic",
            preselect_families=False,
            allow_rotations=True,
            select_trunc_lvl=False,
            select_threshold=False,
            select_families=True,
            tree_algorithm="mst_prim",
            num_threads=1,
            seeds=[int(seed) for seed in fit_seeds],
        )
        training = np.asfortranarray(values)
        model = pv.Vinecop.from_data(training, controls=controls)
        if model.dim != dimension or model.trunc_lvl != truncation_level:
            raise VineFitError("vine engine returned the wrong dimension or truncation")
        model, failures = _sanitize_selected_pairs(model, vine)
        expected_pairs = sum(dimension - tree for tree in range(1, truncation_level + 1))
        observed_pairs = sum(len(tree) for tree in model.pair_copulas)
        if observed_pairs != expected_pairs:
            raise VineFitError("vine engine returned the wrong number of pair copulas")
        densities = np.asarray(model.pdf(training, num_threads=1), dtype=float)
        if not np.isfinite(densities).all() or bool((densities <= 0).any()):
            raise VineFitError("fitted vine has invalid training densities")
    except VineFitError:
        raise
    except (RuntimeError, ValueError) as exc:
        raise VineFitError(f"vine engine fit failed: {type(exc).__name__}") from exc
    loglikelihood = float(np.log(densities).sum())
    number_parameters = float(model.npars)
    family_counts, rotation_counts = _model_counts(model)
    return VineFitResult(
        model=model,
        pair_count=observed_pairs,
        failed_pairs=failures,
        family_counts=family_counts,
        rotation_counts=rotation_counts,
        loglikelihood=loglikelihood,
        aic=2.0 * number_parameters - 2.0 * loglikelihood,
        bic=np.log(len(training)) * number_parameters - 2.0 * loglikelihood,
    )


def vine_dependence_uniforms(
    independent_uniforms: np.ndarray,
    model: pv.Vinecop,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Apply a fitted vine's inverse Rosenblatt transform to fixed uniforms."""

    uniforms = np.asarray(independent_uniforms, dtype=float)
    if uniforms.ndim != 2 or uniforms.shape[1] != model.dim:
        raise ValueError("independent uniforms and vine dimensions do not agree")
    if not np.isfinite(uniforms).all() or bool(((uniforms < 0) | (uniforms > 1)).any()):
        raise ValueError("independent uniforms must be finite and lie in [0, 1]")
    if not 0 < lower < upper < 1:
        raise ValueError("vine PIT bounds must lie strictly inside (0, 1)")
    try:
        dependent = np.asarray(
            model.inverse_rosenblatt(
                np.asfortranarray(np.clip(uniforms, lower, upper)), num_threads=1
            ),
            dtype=float,
        )
    except (RuntimeError, ValueError) as exc:
        raise VineEvaluationError(
            f"inverse Rosenblatt transform failed: {type(exc).__name__}"
        ) from exc
    if dependent.shape != uniforms.shape or not np.isfinite(dependent).all():
        raise VineEvaluationError("inverse Rosenblatt transform produced invalid uniforms")
    return np.clip(dependent, lower, upper)


def vine_log_density(pits: np.ndarray, model: pv.Vinecop) -> np.ndarray:
    """Return per-observation vine copula log densities."""

    values = np.asarray(pits, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != model.dim:
        raise ValueError("PIT values and vine dimensions do not agree")
    if not np.isfinite(values).all() or bool(((values <= 0) | (values >= 1)).any()):
        raise ValueError("PIT values must be finite and lie strictly inside (0, 1)")
    try:
        densities = np.asarray(model.pdf(np.asfortranarray(values), num_threads=1), dtype=float)
    except (RuntimeError, ValueError) as exc:
        raise VineEvaluationError(f"vine density evaluation failed: {type(exc).__name__}") from exc
    if not np.isfinite(densities).all() or bool((densities <= 0).any()):
        raise VineEvaluationError("vine density is non-finite or non-positive")
    return np.log(densities)


def _prepare_frames(
    training_pits: pd.DataFrame,
    marginal_refits: pd.DataFrame,
    daily_margins: pd.DataFrame,
    group_returns: pd.DataFrame,
    gaussian_refits: pd.DataFrame,
    *,
    universe_variant: str,
    supported_groupings: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_input_frames(training_pits, marginal_refits, daily_margins, group_returns)
    required_gaussian = {
        "copula_refit_id",
        "refit_date",
        "year",
        "month",
        "universe_variant",
        "model_id",
        "grouping_id",
        "group_order_json",
        "correlation_matrix_json",
        "base_uniform_sha256",
    }
    missing = sorted(required_gaussian - set(gaussian_refits.columns))
    if missing:
        raise ValueError(f"Gaussian refit frame is missing columns: {missing}")
    frames = (
        training_pits.loc[training_pits["universe_variant"] == universe_variant].copy(),
        marginal_refits.loc[marginal_refits["universe_variant"] == universe_variant].copy(),
        daily_margins.loc[daily_margins["universe_variant"] == universe_variant].copy(),
        group_returns.loc[group_returns["universe_variant"] == universe_variant].copy(),
        gaussian_refits.loc[gaussian_refits["universe_variant"] == universe_variant].copy(),
    )
    training, refits, daily, returns, gaussian = frames
    for frame, columns in (
        (training, ("training_date", "refit_date")),
        (refits, ("refit_date", "requested_training_start")),
        (daily, ("date",)),
        (returns, ("date",)),
        (gaussian, ("refit_date",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
            if frame[column].isna().any():
                raise ValueError(f"invalid date in {column}")
    if any(frame.empty for frame in (training, refits, daily, gaussian)):
        raise ValueError(f"missing modelling rows for universe variant {universe_variant}")
    if not set(training["grouping_id"]).issubset(supported_groupings):
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
    return training, refits, daily, returns, gaussian


def _validated_seed_record(
    raw: Any,
    simulation: Mapping[str, Any],
    existing_keys: set[tuple[int, int]],
) -> tuple[tuple[int, int], Mapping[str, Any]]:
    if not isinstance(raw, Mapping):
        raise TypeError("seed manifest record must be an object")
    record = cast(Mapping[str, Any], raw)
    year = record.get("year")
    month = record.get("month")
    valid_date = (
        not isinstance(year, bool)
        and isinstance(year, int)
        and not isinstance(month, bool)
        and isinstance(month, int)
        and 1 <= month <= 12
    )
    if not valid_date:
        raise ValueError("seed manifest record has an invalid year or month")
    key = cast(int, year), cast(int, month)
    if key in existing_keys:
        raise ValueError(f"duplicate seed manifest month: {key}")
    expected_fields = {
        "seed_components": [simulation["base_seed"], *key],
        "draws": simulation["draws_per_month"],
        "dimension": simulation["dimension"],
    }
    mismatches = [
        field for field, expected in expected_fields.items() if record.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            f"seed manifest record differs from simulation config: {key}/{mismatches[0]}"
        )
    uniform_hash = record.get("base_uniform_sha256")
    valid_hash = (
        isinstance(uniform_hash, str)
        and len(uniform_hash) == 64
        and all(character in "0123456789abcdef" for character in uniform_hash)
    )
    if not valid_hash:
        raise ValueError(f"seed manifest record has an invalid uniform hash: {key}")
    return key, record


def _validated_seed_records(
    seed_manifest: Mapping[str, Any],
    simulation: Mapping[str, Any],
) -> dict[tuple[int, int], Mapping[str, Any]]:
    expected_manifest = {
        "schema_version": 1,
        "manifest_type": "monthly_common_random_numbers",
        "seed_sequence": "SeedSequence([base_seed, year, month])",
        **{
            key: simulation.get(key)
            for key in ("bit_generator", "base_seed", "draws_per_month", "dimension")
        },
    }
    mismatches = [
        key for key, expected in expected_manifest.items() if seed_manifest.get(key) != expected
    ]
    if mismatches:
        key = mismatches[0]
        if key == "schema_version":
            raise ValueError("unexpected common-random-number manifest schema version")
        if key == "manifest_type":
            raise ValueError("unexpected common-random-number manifest type")
        if key == "seed_sequence":
            raise ValueError("unexpected common-random-number seed sequence")
        raise ValueError(f"seed manifest differs from simulation config: {key}")
    raw_records = seed_manifest.get("records")
    if not isinstance(raw_records, list):
        raise TypeError("seed manifest records must be an array")
    records: dict[tuple[int, int], Mapping[str, Any]] = {}
    for raw in raw_records:
        key, record = _validated_seed_record(raw, simulation, set(records))
        records[key] = record
    return records


def _matched_gaussian(
    gaussian_refits: pd.DataFrame,
    *,
    year: int,
    month: int,
    baseline_model_id: str,
    group_order: Sequence[str],
    uniform_hash: str,
) -> tuple[str, np.ndarray]:
    selected = gaussian_refits.loc[
        (gaussian_refits["year"] == year)
        & (gaussian_refits["month"] == month)
        & (gaussian_refits["model_id"] == baseline_model_id)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"missing matched Gaussian refit for {year}-{month:02d}/{baseline_model_id}"
        )
    row = selected.iloc[0]
    if json.loads(row["group_order_json"]) != list(group_order):
        raise ValueError("matched Gaussian group order differs from the vine order")
    if row["base_uniform_sha256"] != uniform_hash:
        raise ValueError("matched Gaussian refit used different common random numbers")
    correlation = np.asarray(json.loads(row["correlation_matrix_json"]), dtype=float)
    return str(row["copula_refit_id"]), correlation


def _forecast_rows(
    daily_block: pd.DataFrame,
    *,
    group_order: Sequence[str],
    innovations: np.ndarray,
    log_scores: Mapping[pd.Timestamp, float],
    confidence_levels: Sequence[float],
    es_confidence: float,
    model_id: str,
    output_grouping: str,
    vine_refit_id: str,
    seed_components: list[int],
    margin_fallback_count: int,
    whole_vine_fallback: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, day in daily_block.groupby("date", sort=True):
        ordered = day.set_index("group_id", verify_integrity=True).reindex(group_order)
        if ordered.isna().any().any():
            raise ValueError(f"incomplete daily margin vector for {date}/{vine_refit_id}")
        means = ordered["conditional_mean_log_return"].to_numpy(dtype=float)
        volatilities = ordered["conditional_volatility_log_return"].to_numpy(dtype=float)
        weights = ordered["portfolio_weight"].to_numpy(dtype=float)
        realised_log = ordered["realised_log_return"].to_numpy(dtype=float)
        if bool((volatilities <= 0).any()):
            raise ValueError(f"non-positive marginal volatility for {date}/{vine_refit_id}")
        if bool(((weights <= 0) | (weights > 1)).any()) or not np.isclose(
            weights.sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"invalid portfolio weights for {date}/{model_id}")
        simulated_losses = -(np.expm1(means + innovations * volatilities) @ weights)
        risks = empirical_risk_levels(simulated_losses, confidence_levels)
        realised_simple = float(np.expm1(realised_log) @ weights)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "model_id": model_id,
                "grouping_id": output_grouping,
                "refit_id": vine_refit_id,
                "var_95": risks[0.95][0],
                "var_975": risks[0.975][0],
                "var_99": risks[0.99][0],
                "es_975": risks[es_confidence][1],
                "realised_simple_return": realised_simple,
                "realised_loss": -realised_simple,
                "seed_components": seed_components,
                "margin_fallback_count": margin_fallback_count,
                "whole_vine_fallback": whole_vine_fallback,
                "copula_log_score": log_scores[pd.Timestamp(date)],
                "forecast_status": (
                    "fallback" if margin_fallback_count or whole_vine_fallback else "ok"
                ),
            }
        )
    return rows


def _validated_vine_registry(
    model_by_grouping: Mapping[str, tuple[str, str, str]] | None,
) -> tuple[dict[str, tuple[str, str, str]], set[str]]:
    registry = dict(VINE_MODEL_BY_GROUPING if model_by_grouping is None else model_by_grouping)
    valid = registry and all(
        isinstance(input_grouping, str)
        and bool(input_grouping)
        and isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(item, str) and bool(item) for item in value)
        for input_grouping, value in registry.items()
    )
    if not valid:
        raise ValueError(
            "vine model registry must map groupings to model/grouping/baseline triples"
        )
    expected_models = {model_id for model_id, _, _ in registry.values()}
    unique_fields = (
        expected_models,
        {grouping_id for _, grouping_id, _ in registry.values()},
        {baseline_id for _, _, baseline_id in registry.values()},
    )
    messages = (
        "vine model registry contains duplicate model IDs",
        "vine model registry contains duplicate output groupings",
        "vine model registry contains duplicate Gaussian baselines",
    )
    for values, message in zip(unique_fields, messages, strict=True):
        if len(values) != len(registry):
            raise ValueError(message)
    return registry, expected_models


def _validated_vine_training_block(
    identifiers: tuple[Any, ...],
    block: pd.DataFrame,
    dimension: int,
) -> tuple[list[str], pd.DataFrame, pd.Timestamp, str]:
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
    return group_order, pivot, pd.Timestamp(refit_dates.iloc[0]), str(copula_ids.iloc[0])


def _validated_common_uniforms(
    year: int,
    month: int,
    *,
    seed_records: Mapping[tuple[int, int], Mapping[str, Any]],
    draws: int,
    dimension: int,
    base_seed: int,
) -> tuple[np.ndarray, str]:
    seed_key = year, month
    if seed_key not in seed_records:
        raise ValueError(f"missing common-random-number record for {seed_key}")
    independent = common_uniforms(
        year,
        month,
        draws=draws,
        dimension=dimension,
        base_seed=base_seed,
    )
    uniform_hash = _array_sha256(independent)
    if seed_records[seed_key].get("base_uniform_sha256") != uniform_hash:
        raise ValueError(f"common-random-number hash differs for {seed_key}")
    return independent, uniform_hash


def _attempt_vine_fit(
    pivot: pd.DataFrame,
    vine: Mapping[str, Any],
    *,
    truncation: int,
    fit_seeds: list[int],
    maximum_pair_failure: float,
    vine_fitter: VineFitFunction,
) -> tuple[VineFitResult | None, str | None]:
    try:
        result = vine_fitter(
            pivot.to_numpy(dtype=float),
            vine,
            truncation_level=truncation,
            fit_seeds=fit_seeds,
        )
    except VineFitError as exc:
        return None, f"structure_or_fit_failure:{type(exc).__name__}"
    reason = (
        "failed_pair_fraction_exceeded"
        if result.failed_pair_fraction > maximum_pair_failure
        else None
    )
    return result, reason


def _vine_or_gaussian_dependence(
    independent: np.ndarray,
    daily_pits: pd.DataFrame,
    fit_result: VineFitResult | None,
    fallback_reason: str | None,
    gaussian_correlation: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
    whole_vine_fallback = fallback_reason is not None
    if not whole_vine_fallback:
        if fit_result is None:
            raise RuntimeError("successful vine fit has no result")
        try:
            dependent = vine_dependence_uniforms(
                independent, fit_result.model, lower=lower, upper=upper
            )
            log_density = vine_log_density(daily_pits.to_numpy(dtype=float), fit_result.model)
            return dependent, log_density, False, None
        except VineEvaluationError as exc:
            fallback_reason = f"vine_evaluation_failure:{type(exc).__name__}"
    dependent = gaussian_dependence_uniforms(
        independent, gaussian_correlation, lower=lower, upper=upper
    )
    log_density = gaussian_copula_log_density(
        daily_pits.to_numpy(dtype=float), gaussian_correlation
    )
    return dependent, log_density, True, fallback_reason


def _vine_refit_row(
    *,
    fit_result: VineFitResult | None,
    fallback_reason: str | None,
    pair_count: int,
    vine_refit_id: str,
    source_copula_refit_id: str,
    matched_gaussian_id: str,
    refit_date: pd.Timestamp,
    year: int,
    month: int,
    universe_variant: str,
    model_id: str,
    output_grouping: str,
    pivot: pd.DataFrame,
    dimension: int,
    truncation: int,
    group_order: Sequence[str],
    base_seed: int,
    uniform_hash: str,
    whole_vine_fallback: bool,
) -> dict[str, Any]:
    failed_pairs = () if fit_result is None else fit_result.failed_pairs
    family_counts = {} if fit_result is None else fit_result.family_counts
    rotation_counts = {} if fit_result is None else fit_result.rotation_counts
    if fit_result is not None:
        pair_count = fit_result.pair_count
    return {
        "vine_refit_id": vine_refit_id,
        "source_copula_refit_id": source_copula_refit_id,
        "matched_gaussian_refit_id": matched_gaussian_id,
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
        "truncation_level": truncation,
        "group_order_json": json.dumps(group_order, separators=(",", ":")),
        "structure_matrix_json": (
            None
            if fit_result is None
            else json.dumps(
                np.asarray(fit_result.model.matrix, dtype=int).tolist(), separators=(",", ":")
            )
        ),
        "vine_model_json": None if fit_result is None else fit_result.model.to_json(),
        "pair_count": pair_count,
        "failed_pair_count": len(failed_pairs),
        "failed_pair_fraction": len(failed_pairs) / pair_count,
        "failed_pairs_json": json.dumps(failed_pairs, separators=(",", ":")),
        "family_counts_json": json.dumps(family_counts, separators=(",", ":")),
        "rotation_counts_json": json.dumps(rotation_counts, separators=(",", ":")),
        "number_parameters": None if fit_result is None else float(fit_result.model.npars),
        "loglikelihood": None if fit_result is None else fit_result.loglikelihood,
        "aic": None if fit_result is None else fit_result.aic,
        "bic": None if fit_result is None else fit_result.bic,
        "library": "pyvinecopulib",
        "library_version": pv.__version__,
        "fit_seed_components": [base_seed, year, month],
        "base_uniform_sha256": uniform_hash,
        "whole_vine_fallback": whole_vine_fallback,
        "fallback_reason": fallback_reason,
        "fit_status": (
            "gaussian_fallback"
            if whole_vine_fallback
            else "pair_fallback"
            if failed_pairs
            else "ok"
        ),
    }


def _vine_quality_metrics(
    fit_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    expected_models: set[str],
    reconstruction_tolerance: float,
) -> tuple[list[str], dict[str, int], float]:
    numeric_columns = (
        "var_95",
        "var_975",
        "var_99",
        "es_975",
        "realised_simple_return",
        "realised_loss",
        "copula_log_score",
    )
    date_counts = {
        str(key): int(value)
        for key, value in forecast_frame.groupby("model_id")["date"].nunique().items()
    }
    checks = [
        (fit_frame.duplicated(["year", "month", "model_id"]).any(), "duplicate_vine_refit"),
        (forecast_frame.duplicated(["date", "model_id"]).any(), "duplicate_vine_forecast"),
        (
            not np.isfinite(forecast_frame[list(numeric_columns)].to_numpy(dtype=float)).all(),
            "nonfinite_vine_forecast",
        ),
        (
            bool(
                (
                    (forecast_frame["var_95"] > forecast_frame["var_975"])
                    | (forecast_frame["var_975"] > forecast_frame["var_99"])
                ).any()
            ),
            "nonmonotone_var_forecast",
        ),
        (bool((forecast_frame["es_975"] < forecast_frame["var_975"]).any()), "es_below_var"),
        (
            set(date_counts) != expected_models or len(set(date_counts.values())) != 1,
            "incomplete_matched_model_coverage",
        ),
    ]
    realised = forecast_frame.pivot(
        index="date", columns="model_id", values="realised_simple_return"
    )
    if realised.isna().any().any() or not expected_models.issubset(realised.columns):
        checks.append((True, "incomplete_realised_portfolio_identity"))
        maximum_identity_error = float("inf")
    else:
        reference_model = sorted(expected_models)[0]
        maximum_identity_error = float(
            realised.loc[:, sorted(expected_models)]
            .sub(realised[reference_model], axis="index")
            .abs()
            .to_numpy(dtype=float)
            .max()
        )
        checks.append(
            (
                maximum_identity_error > reconstruction_tolerance,
                "grouping_portfolio_identity_failed",
            )
        )
    return [issue for failed, issue in checks if failed], date_counts, maximum_identity_error


def build_vine_outputs(
    training_pits: pd.DataFrame,
    marginal_refits: pd.DataFrame,
    daily_margins: pd.DataFrame,
    group_returns: pd.DataFrame,
    gaussian_refits: pd.DataFrame,
    seed_manifest: Mapping[str, Any],
    model_config: Mapping[str, Any],
    *,
    universe_variant: str = "security_primary",
    truncation_level: int | None = None,
    progress_every: int = 12,
    vine_fitter: VineFitFunction = fit_vine_copula,
    model_by_grouping: Mapping[str, tuple[str, str, str]] | None = None,
    analysis_role: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build monthly vine fits, daily risk forecasts, and quality checks."""

    model_registry, expected_models = _validated_vine_registry(model_by_grouping)
    vine, simulation, forecast, marginal, truncation = validate_vine_protocol(
        model_config, truncation_level=truncation_level
    )
    training, refits, daily, returns, gaussian = _prepare_frames(
        training_pits,
        marginal_refits,
        daily_margins,
        group_returns,
        gaussian_refits,
        universe_variant=universe_variant,
        supported_groupings=set(model_registry),
    )
    seed_records = _validated_seed_records(seed_manifest, simulation)
    dimension = int(vine["dimension"])
    draws = int(simulation["draws_per_month"])
    base_seed = int(simulation["base_seed"])
    lower = float(vine["pit_clip_lower"])
    upper = float(vine["pit_clip_upper"])
    confidence_levels = [float(value) for value in forecast["var_confidence_levels"]]
    es_confidence = float(forecast["es_confidence_level"])
    maximum_pair_failure = float(vine["maximum_failed_pair_fraction"])
    fit_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    block_columns = ["year", "month", "universe_variant", "grouping_id"]
    grouped = training.groupby(block_columns, sort=True)
    for block_index, (identifiers, block) in enumerate(grouped, start=1):
        year, month, _, input_grouping = identifiers
        year, month = int(year), int(month)
        model_id, output_grouping, baseline_model_id = model_registry[str(input_grouping)]
        group_order, pivot, refit_date, source_copula_refit_id = _validated_vine_training_block(
            identifiers, block, dimension
        )
        vine_refit_id = f"{source_copula_refit_id}:vine_t{truncation}"
        monthly_refits = refits.loc[
            (refits["year"] == year)
            & (refits["month"] == month)
            & (refits["grouping_id"] == input_grouping)
        ].copy()
        daily_block = daily.loc[
            (daily["year"] == year)
            & (daily["month"] == month)
            & (daily["grouping_id"] == input_grouping)
        ].copy()
        _validate_block_bindings(
            block,
            monthly_refits,
            daily_block,
            group_order=group_order,
            refit_date=refit_date,
            copula_refit_id=source_copula_refit_id,
        )

        independent, uniform_hash = _validated_common_uniforms(
            year,
            month,
            seed_records=seed_records,
            draws=draws,
            dimension=dimension,
            base_seed=base_seed,
        )
        matched_gaussian_id, gaussian_correlation = _matched_gaussian(
            gaussian,
            year=year,
            month=month,
            baseline_model_id=baseline_model_id,
            group_order=group_order,
            uniform_hash=uniform_hash,
        )

        pair_count = sum(dimension - tree for tree in range(1, truncation + 1))
        fit_result, fallback_reason = _attempt_vine_fit(
            pivot,
            vine,
            truncation=truncation,
            fit_seeds=[base_seed, year, month],
            maximum_pair_failure=maximum_pair_failure,
            vine_fitter=vine_fitter,
        )

        daily_pits = daily_block.pivot(index="date", columns="group_id", values="pit")
        daily_pits = daily_pits.reindex(columns=group_order).sort_index()
        if daily_pits.isna().any().any() or daily_pits.shape[1] != dimension:
            raise ValueError(f"incomplete daily PIT matrix for {vine_refit_id}")
        dependent, log_density_values, whole_vine_fallback, fallback_reason = (
            _vine_or_gaussian_dependence(
                independent,
                daily_pits,
                fit_result,
                fallback_reason,
                gaussian_correlation,
                lower=lower,
                upper=upper,
            )
        )
        innovations = marginal_innovation_draws(
            dependent,
            group_order,
            monthly_refits,
            returns,
            marginal,
        )
        fallback_count = int((monthly_refits["fallback_level"] > 0).sum())
        log_scores = {
            pd.Timestamp(date): float(score)
            for date, score in zip(daily_pits.index, log_density_values, strict=True)
        }
        forecast_rows.extend(
            _forecast_rows(
                daily_block,
                group_order=group_order,
                innovations=innovations,
                log_scores=log_scores,
                confidence_levels=confidence_levels,
                es_confidence=es_confidence,
                model_id=model_id,
                output_grouping=output_grouping,
                vine_refit_id=vine_refit_id,
                seed_components=[base_seed, year, month],
                margin_fallback_count=fallback_count,
                whole_vine_fallback=whole_vine_fallback,
            )
        )
        fit_rows.append(
            _vine_refit_row(
                fit_result=fit_result,
                fallback_reason=fallback_reason,
                pair_count=pair_count,
                vine_refit_id=vine_refit_id,
                source_copula_refit_id=source_copula_refit_id,
                matched_gaussian_id=matched_gaussian_id,
                refit_date=refit_date,
                year=year,
                month=month,
                universe_variant=universe_variant,
                model_id=model_id,
                output_grouping=output_grouping,
                pivot=pivot,
                dimension=dimension,
                truncation=truncation,
                group_order=group_order,
                base_seed=base_seed,
                uniform_hash=uniform_hash,
                whole_vine_fallback=whole_vine_fallback,
            )
        )
        if progress_every and block_index % progress_every == 0:
            print(f"vine_progress={block_index}/{len(grouped)}")

    fit_frame = pd.DataFrame(fit_rows).sort_values(["year", "month", "model_id"])
    forecast_frame = pd.DataFrame(forecast_rows).sort_values(["date", "model_id"])
    issues, date_counts, maximum_identity_error = _vine_quality_metrics(
        fit_frame,
        forecast_frame,
        expected_models,
        float(model_config["portfolio"]["reconstruction_tolerance"]),
    )
    evaluation_dates = int(forecast_frame["date"].nunique())
    fallback_date_counts = {
        str(model_id): int(rows.loc[rows["whole_vine_fallback"], "date"].nunique())
        for model_id, rows in forecast_frame.groupby("model_id")
    }
    fallback_date_fractions = {
        model_id: fallback_date_counts[model_id] / date_counts[model_id]
        for model_id in sorted(fallback_date_counts)
    }
    maximum_model_fallback_fraction = max(fallback_date_fractions.values(), default=0.0)
    any_model_fallback_dates = int(
        forecast_frame.loc[forecast_frame["whole_vine_fallback"], "date"].nunique()
    )
    any_model_fallback_fraction = any_model_fallback_dates / evaluation_dates
    fallback_limit = float(vine["maximum_whole_vine_fallback_date_fraction"])
    resolved_analysis_role = analysis_role or vine_analysis_role(vine, truncation)
    restrictions = []
    if maximum_model_fallback_fraction > fallback_limit:
        restrictions.append("whole_vine_fallback_fraction_exceeded")
    aggregate_families: Counter[str] = Counter()
    for encoded in fit_frame["family_counts_json"]:
        aggregate_families.update(json.loads(encoded))
    audit = {
        "schema_version": 1,
        "gate_name": (
            "vine_copula_quality_v1"
            if resolved_analysis_role == "primary"
            else "full_vine_robustness_quality_v1"
        ),
        "analysis_role": resolved_analysis_role,
        "status": "pass" if not issues else "fail",
        "universe_variant": universe_variant,
        "model_ids": sorted(expected_models),
        "truncation_level": truncation,
        "refit_count": len(fit_frame),
        "forecast_count": len(forecast_frame),
        "evaluation_date_count": evaluation_dates,
        "model_date_counts": date_counts,
        "dimension": dimension,
        "draws_per_month": draws,
        "pair_count": int(fit_frame["pair_count"].sum()),
        "failed_pair_count": int(fit_frame["failed_pair_count"].sum()),
        "maximum_monthly_failed_pair_fraction": float(fit_frame["failed_pair_fraction"].max()),
        "whole_vine_fallback_refit_count": int(fit_frame["whole_vine_fallback"].sum()),
        "whole_vine_fallback_date_count": any_model_fallback_dates,
        "whole_vine_fallback_date_fraction": any_model_fallback_fraction,
        "model_whole_vine_fallback_date_counts": fallback_date_counts,
        "model_whole_vine_fallback_date_fractions": fallback_date_fractions,
        "maximum_model_whole_vine_fallback_date_fraction": maximum_model_fallback_fraction,
        "maximum_whole_vine_fallback_date_fraction": fallback_limit,
        "eligible_to_be_declared_best": not restrictions,
        "family_counts": dict(sorted(aggregate_families.items())),
        "maximum_realised_portfolio_identity_error": maximum_identity_error,
        "portfolio_identity_tolerance": float(
            model_config["portfolio"]["reconstruction_tolerance"]
        ),
        "issues": issues,
        "best_model_restrictions": restrictions,
    }
    return fit_frame.reset_index(drop=True), forecast_frame.reset_index(drop=True), audit


def _validate_upstream_audits(
    *,
    gaussian_audit: Mapping[str, Any],
    gaussian_refits: Path,
    seed_manifest: Path,
) -> None:
    if gaussian_audit.get("status") != "pass":
        raise RuntimeError("Gaussian-copula quality gate is not passed")
    outputs = gaussian_audit.get("outputs", {})
    expected = {
        "gaussian_copula_refits": gaussian_refits,
        "simulation_seed_manifest": seed_manifest,
    }
    for name, path in expected.items():
        if outputs.get(name, {}).get("sha256") != _sha256(path):
            raise RuntimeError(f"{name} differs from the passed Gaussian audit")


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
        "--gaussian-refits",
        type=Path,
        default=PROJECT_ROOT / "data/processed/gaussian_copula_refits.parquet",
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/simulation_seed_manifest.json",
    )
    parser.add_argument(
        "--marginal-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/marginal_model_quality.json",
    )
    parser.add_argument(
        "--gaussian-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/gaussian_copula_quality.json",
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
        help="Defaults to the role-specific primary or full-vine path.",
    )
    parser.add_argument(
        "--forecasts-output",
        type=Path,
        help="Defaults to the role-specific primary or full-vine path.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Defaults to the role-specific primary or full-vine path.",
    )
    parser.add_argument("--universe-variant", default="security_primary")
    parser.add_argument("--truncation-level", type=int)
    parser.add_argument("--progress-every", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    vine, _, _, _, truncation = validate_vine_protocol(
        model_config, truncation_level=args.truncation_level
    )
    refits_output, forecasts_output, audit_output = resolve_output_paths(
        vine,
        truncation,
        refits_output=args.refits_output,
        forecasts_output=args.forecasts_output,
        audit_output=args.audit_output,
    )
    foundation = json.loads(args.foundation_status.read_text(encoding="utf-8"))
    _require_current_hash_records(
        foundation,
        source_name=_project_path(args.foundation_status),
    )
    scope = _reporting_scope(foundation)
    marginal_audit = json.loads(args.marginal_audit.read_text(encoding="utf-8"))
    _require_current_hash_records(
        marginal_audit,
        source_name=_project_path(args.marginal_audit),
    )
    _validate_marginal_binding(
        marginal_audit,
        training_pits=args.training_pits,
        marginal_refits=args.marginal_refits,
        daily_margins=args.daily_margins,
        group_returns=args.group_returns,
        model_config=args.model_config,
    )
    gaussian_audit = json.loads(args.gaussian_audit.read_text(encoding="utf-8"))
    _require_current_hash_records(
        gaussian_audit,
        source_name=_project_path(args.gaussian_audit),
    )
    _validate_upstream_audits(
        gaussian_audit=gaussian_audit,
        gaussian_refits=args.gaussian_refits,
        seed_manifest=args.seed_manifest,
    )
    seed_manifest = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    vine_refits, forecasts, audit = build_vine_outputs(
        pd.read_parquet(args.training_pits),
        pd.read_parquet(args.marginal_refits),
        pd.read_parquet(args.daily_margins),
        pd.read_parquet(args.group_returns),
        pd.read_parquet(args.gaussian_refits),
        seed_manifest,
        model_config,
        universe_variant=args.universe_variant,
        truncation_level=truncation,
        progress_every=args.progress_every,
    )
    _write_parquet_atomic(refits_output, vine_refits)
    _write_parquet_atomic(forecasts_output, forecasts)
    output_prefix = "vine" if audit["analysis_role"] == "primary" else "full_vine"
    audit.update(
        {
            "reporting_scope": scope,
            "method": {
                "library": "pyvinecopulib",
                "library_version": pv.__version__,
                "structure_selection": model_config["vine"]["structure_selection"],
                "family_selection": model_config["vine"]["family_selection"],
                "families": model_config["vine"]["families"],
                "rotations_degrees": model_config["vine"]["rotations_degrees"],
                "truncation_level": truncation,
                "num_threads": 1,
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
                "gaussian_copula_refits": {
                    "path": _project_path(args.gaussian_refits),
                    "sha256": _sha256(args.gaussian_refits),
                },
                "simulation_seed_manifest": {
                    "path": _project_path(args.seed_manifest),
                    "sha256": _sha256(args.seed_manifest),
                },
                "marginal_model_quality": {
                    "path": _project_path(args.marginal_audit),
                    "sha256": _sha256(args.marginal_audit),
                },
                "gaussian_copula_quality": {
                    "path": _project_path(args.gaussian_audit),
                    "sha256": _sha256(args.gaussian_audit),
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
                f"{output_prefix}_copula_refits": {
                    "path": _project_path(refits_output),
                    "sha256": _sha256(refits_output),
                    "rows": len(vine_refits),
                },
                f"{output_prefix}_risk_forecasts": {
                    "path": _project_path(forecasts_output),
                    "sha256": _sha256(forecasts_output),
                    "rows": len(forecasts),
                },
            },
        }
    )
    _write_json_atomic(audit_output, audit)
    print(
        f"vine_copula_quality={audit['status']} truncation={truncation} "
        f"refits={len(vine_refits)} forecasts={len(forecasts)} "
        f"whole_fallbacks={audit['whole_vine_fallback_refit_count']}"
    )
    print(f"Audit: {audit_output}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
