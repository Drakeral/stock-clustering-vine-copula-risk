#!/usr/bin/env python3
"""Evaluate exploratory M5--M8 forecasts against matched core baselines."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.build_ml_groupings import validate_ml_protocol
    from scripts.evaluate_risk_models import (
        DAILY_SCORE_COLUMNS,
        EXCEPTION_COLUMNS,
        MODEL_GROUPINGS,
        MODEL_IDS,
        PRIMARY_SCORES,
        _normalize_forecast_source,
        _validate_risk_numbers,
        score_forecasts,
    )
    from scripts.ml_risk_common import (
        ML_MODEL_GROUPINGS,
        ML_MODEL_IDS,
        ML_RISK_ANALYSIS_ROLE,
        artifact_record,
        load_json,
        require_audit_output,
    )
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        write_json_atomic,
        write_parquet_atomic,
    )
    from scripts.research_methods import (
        benjamini_hochberg_adjust,
        christoffersen_conditional_coverage_p_value,
        christoffersen_independence,
        diebold_mariano,
        kupiec_unconditional_coverage,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from build_ml_groupings import validate_ml_protocol
    from evaluate_risk_models import (
        DAILY_SCORE_COLUMNS,
        EXCEPTION_COLUMNS,
        MODEL_GROUPINGS,
        MODEL_IDS,
        PRIMARY_SCORES,
        _normalize_forecast_source,
        _validate_risk_numbers,
        score_forecasts,
    )
    from ml_risk_common import (
        ML_MODEL_GROUPINGS,
        ML_MODEL_IDS,
        ML_RISK_ANALYSIS_ROLE,
        artifact_record,
        load_json,
        require_audit_output,
    )
    from pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        write_json_atomic,
        write_parquet_atomic,
    )
    from research_methods import (
        benjamini_hochberg_adjust,
        christoffersen_conditional_coverage_p_value,
        christoffersen_independence,
        diebold_mariano,
        kupiec_unconditional_coverage,
    )


def validate_ml_risk_protocol(
    ml_config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Bind the exploratory inference choices to the frozen core settings."""

    validate_ml_protocol(ml_config)
    exploratory = cast(Mapping[str, Any], ml_config["exploratory_inference"])
    inference = cast(Mapping[str, Any], model_config.get("inference"))
    forecast = cast(Mapping[str, Any], model_config.get("forecast"))
    if not isinstance(inference, Mapping) or not isinstance(forecast, Mapping):
        raise TypeError("model configuration is missing inference or forecast settings")
    comparisons = exploratory["comparisons"]
    scores = exploratory["loss_scores"]
    settings = {
        "DM HAC lag": (exploratory["dm_hac_lag"], inference["dm_hac_lag"]),
        "multiple testing": (
            exploratory["multiplicity"],
            inference["ml_multiple_testing"],
        ),
        "FDR": (exploratory["false_discovery_rate"], inference["ml_fdr"]),
        "score columns": (scores, list(PRIMARY_SCORES)),
        "family size": (exploratory["family_size"], len(comparisons) * len(scores)),
        "calibration levels": (
            exploratory["calibration_confidence_levels"],
            forecast["var_confidence_levels"],
        ),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in settings.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"ML risk protocol differs from the core freeze: {mismatches}")
    return exploratory


def _validate_core_risk_values(frame: pd.DataFrame) -> None:
    finite_columns = [
        "realised_loss",
        "var_95",
        "var_975",
        "var_99",
        "es_975",
        *PRIMARY_SCORES,
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError("core daily scores contain non-finite risk or loss values")
    if bool((frame["var_95"] > frame["var_975"]).any()) or bool(
        (frame["var_975"] > frame["var_99"]).any()
    ):
        raise ValueError("core daily VaR forecasts are not monotone")
    if bool((frame["es_975"] <= 0).any()) or bool((frame["es_975"] < frame["var_975"]).any()):
        raise ValueError("core daily ES forecasts are invalid")


def _validate_core_model_contracts(frame: pd.DataFrame) -> None:
    if not frame["forecast_status"].isin(["ok", "fallback"]).all():
        raise ValueError("core daily scores contain an unsupported forecast status")
    for model_id, grouping_id in MODEL_GROUPINGS.items():
        observed = set(frame.loc[frame["model_id"] == model_id, "grouping_id"].astype(str))
        if observed != {grouping_id}:
            raise ValueError(f"core {model_id} does not use grouping {grouping_id}")
    exception_values = frame.loc[:, list(EXCEPTION_COLUMNS.values())]
    if exception_values.isna().any().any() or not exception_values.isin([True, False]).all().all():
        raise ValueError("core daily scores contain invalid exception indicators")
    non_m0 = frame["model_id"] != "M0"
    if not np.isfinite(frame.loc[non_m0, "copula_log_score"].to_numpy(dtype=float)).all():
        raise ValueError("core copula scores must be finite")
    if frame.loc[~non_m0, "copula_log_score"].notna().any():
        raise ValueError("core M0 records must not contain copula scores")


def _validate_core_date_alignment(frame: pd.DataFrame) -> None:
    reference_dates = pd.DatetimeIndex(frame.loc[frame["model_id"] == "M0", "date"]).sort_values()
    for model_id in MODEL_IDS[1:]:
        model_dates = pd.DatetimeIndex(
            frame.loc[frame["model_id"] == model_id, "date"]
        ).sort_values()
        if not model_dates.equals(reference_dates):
            raise ValueError("core daily score dates are not matched across M0--M4")


def _normalize_core_scores(core_scores: pd.DataFrame) -> pd.DataFrame:
    """Validate the core comparison panel independently of its upstream audit."""

    if set(core_scores.columns) != set(DAILY_SCORE_COLUMNS):
        raise ValueError("core daily score schema differs from the frozen schema")
    frame = core_scores.loc[:, DAILY_SCORE_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("core daily scores contain missing dates")
    if set(frame["model_id"].astype(str)) != set(MODEL_IDS):
        raise ValueError("core daily scores do not contain exactly M0--M4")
    if frame.duplicated(["date", "model_id"]).any():
        raise ValueError("core daily scores contain duplicate model-date rows")
    if bool((frame["year"].to_numpy(dtype=int) != frame["date"].dt.year.to_numpy()).any()):
        raise ValueError("core daily score years do not match their dates")
    _validate_core_risk_values(frame)
    _validate_core_model_contracts(frame)
    _validate_core_date_alignment(frame)
    return frame.sort_values(["date", "model_id"]).reset_index(drop=True)


def combine_ml_forecasts(
    gaussian: pd.DataFrame,
    vine: pd.DataFrame,
    core_scores: pd.DataFrame,
    *,
    identity_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate M5--M8 schemas, dates, groupings, and realised-loss identity."""

    if not np.isfinite(identity_tolerance) or identity_tolerance <= 0:
        raise ValueError("realised-loss identity tolerance must be positive and finite")
    normalized = [
        _normalize_forecast_source(gaussian, {"M5", "M7"}, 1),
        _normalize_forecast_source(vine, {"M6", "M8"}, 2),
    ]
    combined = pd.concat(normalized, ignore_index=True)
    _validate_risk_numbers(combined)
    if not np.isfinite(combined["copula_log_score"].to_numpy(dtype=float)).all():
        raise ValueError("ML forecasts require finite copula log scores")
    if not combined["forecast_status"].isin(["ok", "fallback"]).all():
        raise ValueError("ML forecasts contain a failed or unsupported status")
    sign_error = np.abs(combined["realised_loss"] + combined["realised_simple_return"])
    if float(sign_error.max()) > identity_tolerance:
        raise ValueError("ML realised loss does not equal negative simple return")
    for model_id, grouping_id in ML_MODEL_GROUPINGS.items():
        observed = set(combined.loc[combined["model_id"] == model_id, "grouping_id"].astype(str))
        if observed != {grouping_id}:
            raise ValueError(f"{model_id} does not use grouping {grouping_id}")

    core = _normalize_core_scores(core_scores)
    reference = (
        core.loc[core["model_id"] == "M0", ["date", "realised_loss"]].set_index("date").sort_index()
    )
    dates = pd.DatetimeIndex(reference.index).sort_values()
    realised = combined.pivot(index="date", columns="model_id", values="realised_loss")
    if (
        set(realised.columns) != set(ML_MODEL_IDS)
        or not realised.index.sort_values().equals(dates)
        or realised.isna().any().any()
    ):
        raise ValueError("ML forecast dates or model coverage do not match the core evaluation")
    maximum_identity_error = float(
        realised.sub(reference["realised_loss"], axis="index").abs().to_numpy(dtype=float).max()
    )
    if maximum_identity_error > identity_tolerance:
        raise ValueError("ML and core models do not share the same realised portfolio loss")
    order = {model_id: index for index, model_id in enumerate(ML_MODEL_IDS)}
    combined["_model_order"] = combined["model_id"].map(order)
    combined = combined.sort_values(["date", "_model_order"]).drop(columns="_model_order")
    validation = {
        "evaluation_date_count": len(dates),
        "forecast_count": len(combined),
        "model_date_counts": {
            model_id: int((combined["model_id"] == model_id).sum()) for model_id in ML_MODEL_IDS
        },
        "evaluation_start": dates.min().date().isoformat(),
        "evaluation_end": dates.max().date().isoformat(),
        "maximum_core_ml_realised_loss_identity_error": maximum_identity_error,
        "realised_loss_identity_tolerance": identity_tolerance,
    }
    return combined.reset_index(drop=True), validation


def ml_dm_results(
    ml_scores: pd.DataFrame,
    core_scores: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run and jointly BH-adjust the frozen 24 matched loss comparisons."""

    core = _normalize_core_scores(core_scores)
    results: list[dict[str, Any]] = []
    for comparison in protocol["comparisons"]:
        ml_model = str(comparison["ml_model"])
        baseline_model = str(comparison["baseline_model"])
        ml = ml_scores.loc[ml_scores["model_id"] == ml_model].set_index("date")
        baseline = core.loc[core["model_id"] == baseline_model].set_index("date")
        if not ml.index.equals(baseline.index):
            raise ValueError(f"DM dates are not matched for {ml_model} and {baseline_model}")
        for score_column in protocol["loss_scores"]:
            differential = ml[score_column].to_numpy() - baseline[score_column].to_numpy()
            result = diebold_mariano(differential, hac_lag=int(protocol["dm_hac_lag"]))
            results.append(
                {
                    "comparison_id": f"{ml_model}_minus_{baseline_model}:{score_column}",
                    "ml_model": ml_model,
                    "baseline_model": baseline_model,
                    "baseline_grouping": str(comparison["baseline_grouping"]),
                    "dependence": str(comparison["dependence"]),
                    "score": str(score_column),
                    "observations": result.observations,
                    "mean_loss_difference": result.mean_difference,
                    "dm_statistic": result.statistic,
                    "p_value_two_sided": result.p_value_two_sided,
                    "hac_lag": result.hac_lag,
                }
            )
    if len(results) != int(protocol["family_size"]):
        raise AssertionError("exploratory DM family has the wrong size")
    adjusted = benjamini_hochberg_adjust([row["p_value_two_sided"] for row in results])
    fdr = float(protocol["false_discovery_rate"])
    for row, adjusted_p in zip(results, adjusted, strict=True):
        difference = float(row["mean_loss_difference"])
        row["bh_fdr_p_value"] = float(adjusted_p)
        row["significant_after_bh_fdr"] = bool(adjusted_p < fdr)
        row["direction"] = (
            "ml_lower_loss"
            if difference < 0
            else "baseline_lower_loss"
            if difference > 0
            else "equal_loss"
        )
        row["favours_ml_after_bh_fdr"] = bool(adjusted_p < fdr and difference < 0)
        row["favours_baseline_after_bh_fdr"] = bool(adjusted_p < fdr and difference > 0)
    return results


def descriptive_calibration(
    scores: pd.DataFrame, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return explicitly unadjusted full-period and annual ML calibration diagnostics."""

    def record(
        model_id: str,
        confidence: float,
        indicators: np.ndarray,
        *,
        year: int | None,
    ) -> dict[str, Any]:
        coverage = kupiec_unconditional_coverage(indicators, confidence)
        independence = christoffersen_independence(indicators)
        row: dict[str, Any] = {
            "model_id": model_id,
            "confidence": confidence,
            "inference_role": "descriptive_unadjusted",
            "observations": coverage.observations,
            "exceptions": coverage.exceptions,
            "exception_rate": coverage.exception_rate,
            "kupiec_likelihood_ratio": coverage.likelihood_ratio,
            "kupiec_p_value_unadjusted": coverage.p_value,
            "christoffersen_independence_likelihood_ratio": independence.likelihood_ratio,
            "christoffersen_independence_p_value_unadjusted": independence.p_value,
            "conditional_coverage_p_value_unadjusted": (
                christoffersen_conditional_coverage_p_value(coverage, independence)
            ),
        }
        if year is not None:
            row["year"] = year
        return row

    full: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    for model_id in ML_MODEL_IDS:
        model = scores.loc[scores["model_id"] == model_id].sort_values("date")
        for configured_confidence in protocol["calibration_confidence_levels"]:
            confidence = float(configured_confidence)
            full.append(
                record(
                    model_id,
                    confidence,
                    model[EXCEPTION_COLUMNS[confidence]].to_numpy(dtype=bool),
                    year=None,
                )
            )
        for year, annual_model in model.groupby("year", sort=True):
            for configured_confidence in protocol["calibration_confidence_levels"]:
                confidence = float(configured_confidence)
                annual.append(
                    record(
                        model_id,
                        confidence,
                        annual_model[EXCEPTION_COLUMNS[confidence]].to_numpy(dtype=bool),
                        year=int(year),
                    )
                )
    return full, annual


def model_summaries(scores: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize ML forecast loss, calibration counts, and copula log score."""

    rows: list[dict[str, Any]] = []
    for model_id in ML_MODEL_IDS:
        model = scores.loc[scores["model_id"] == model_id]
        row: dict[str, Any] = {
            "model_id": model_id,
            "grouping_id": ML_MODEL_GROUPINGS[model_id],
            "observations": len(model),
            "mean_quantile_loss_95": float(model["quantile_loss_95"].mean()),
            "mean_quantile_loss_99": float(model["quantile_loss_99"].mean()),
            "mean_fz0_975": float(model["fz0_975"].mean()),
            "mean_copula_log_score": float(model["copula_log_score"].mean()),
        }
        for confidence, exception_column in EXCEPTION_COLUMNS.items():
            suffix = str(confidence).replace("0.", "").replace(".", "")
            row[f"exceptions_{suffix}"] = int(model[exception_column].sum())
            row[f"exception_rate_{suffix}"] = float(model[exception_column].mean())
        rows.append(row)
    return rows


def copula_log_score_comparisons(scores: pd.DataFrame) -> list[dict[str, Any]]:
    """Compare Gaussian and vine log scores only within the same ML grouping."""

    rows: list[dict[str, Any]] = []
    for gaussian_id, vine_id, grouping_id in (
        ("M5", "M6", "spectral"),
        ("M7", "M8", "pca_kmeans"),
    ):
        gaussian = scores.loc[scores["model_id"] == gaussian_id].set_index("date")
        vine = scores.loc[scores["model_id"] == vine_id].set_index("date")
        if not gaussian.index.equals(vine.index):
            raise ValueError(f"copula log-score dates differ for {grouping_id}")
        difference = vine["copula_log_score"] - gaussian["copula_log_score"]
        rows.append(
            {
                "grouping_id": grouping_id,
                "gaussian_model": gaussian_id,
                "vine_model": vine_id,
                "mean_gaussian_log_score": float(gaussian["copula_log_score"].mean()),
                "mean_vine_log_score": float(vine["copula_log_score"].mean()),
                "mean_vine_minus_gaussian_log_score": float(difference.mean()),
                "inference_role": "descriptive",
            }
        )
    return rows


def build_ml_evaluation_outputs(
    gaussian: pd.DataFrame,
    vine: pd.DataFrame,
    core_scores: pd.DataFrame,
    ml_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build M5--M8 daily scores and the exploratory inference audit."""

    protocol = validate_ml_risk_protocol(ml_config, model_config)
    forecasts, validation = combine_ml_forecasts(
        gaussian,
        vine,
        core_scores,
        identity_tolerance=float(model_config["portfolio"]["reconstruction_tolerance"]),
    )
    scores = score_forecasts(forecasts)
    dm = ml_dm_results(scores, core_scores, protocol)
    full_calibration, annual_calibration = descriptive_calibration(scores, protocol)
    ml_favoured = sum(bool(row["favours_ml_after_bh_fdr"]) for row in dm)
    baseline_favoured = sum(bool(row["favours_baseline_after_bh_fdr"]) for row in dm)
    audit = {
        "schema_version": 1,
        "gate_name": "ml_model_evaluation_v1",
        "analysis_role": ML_RISK_ANALYSIS_ROLE,
        "status": "pass",
        "issues": [],
        "forecast_validation": validation,
        "model_summaries": model_summaries(scores),
        "full_period_calibration_descriptive": full_calibration,
        "annual_calibration_descriptive": annual_calibration,
        "diebold_mariano_comparisons": dm,
        "copula_log_score_comparisons": copula_log_score_comparisons(scores),
        "interpretation": {
            "multiplicity": "benjamini_hochberg",
            "false_discovery_rate": float(protocol["false_discovery_rate"]),
            "comparison_family_size": len(dm),
            "tests_favouring_ml_after_bh_fdr": ml_favoured,
            "tests_favouring_baseline_after_bh_fdr": baseline_favoured,
            "evidence": (
                "ml_models_favoured"
                if ml_favoured and not baseline_favoured
                else "core_baselines_favoured"
                if baseline_favoured and not ml_favoured
                else "mixed"
                if ml_favoured and baseline_favoured
                else "no_fdr_adjusted_loss_difference"
            ),
            "core_hypotheses_and_ranking_revised": False,
        },
    }
    return scores, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gaussian-forecasts",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_gaussian_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--vine-forecasts",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_vine_risk_forecasts.parquet",
    )
    parser.add_argument(
        "--core-scores",
        type=Path,
        default=PROJECT_ROOT / "data/processed/risk_evaluation_daily.parquet",
    )
    parser.add_argument(
        "--gaussian-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_gaussian_copula_quality.json",
    )
    parser.add_argument(
        "--vine-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_vine_copula_quality.json",
    )
    parser.add_argument(
        "--core-evaluation-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/model_evaluation.json",
    )
    parser.add_argument(
        "--model-config", type=Path, default=PROJECT_ROOT / "config/model_config.toml"
    )
    parser.add_argument(
        "--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_extension_config.toml"
    )
    parser.add_argument(
        "--foundation-status",
        type=Path,
        default=PROJECT_ROOT / "data/audit/current_gate_status.json",
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_risk_evaluation_daily.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_model_evaluation.json",
    )
    return parser.parse_args()


def _require_inputs(args: argparse.Namespace, scope: str) -> None:
    for path, audit_name, output_key, output_path in (
        (
            args.gaussian_audit,
            "ML Gaussian audit",
            "gaussian_risk_forecasts",
            args.gaussian_forecasts,
        ),
        (args.vine_audit, "ML vine audit", "vine_risk_forecasts", args.vine_forecasts),
        (
            args.core_evaluation_audit,
            "core evaluation audit",
            "risk_evaluation_daily",
            args.core_scores,
        ),
    ):
        audit = load_json(path)
        require_current_hash_records(audit, source_name=project_path(path))
        require_audit_output(
            audit,
            audit_name=audit_name,
            output_key=output_key,
            output_path=output_path,
            reporting_scope=scope,
            analysis_role=(ML_RISK_ANALYSIS_ROLE if audit_name.startswith("ML") else None),
        )


def main() -> int:
    args = parse_args()
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    with args.ml_config.open("rb") as handle:
        ml_config = tomllib.load(handle)
    validate_ml_risk_protocol(ml_config, model_config)
    foundation = load_json(args.foundation_status)
    require_current_hash_records(
        foundation,
        source_name=project_path(args.foundation_status),
    )
    scope = reporting_scope(foundation)
    _require_inputs(args, scope)
    scores, audit = build_ml_evaluation_outputs(
        pd.read_parquet(args.gaussian_forecasts),
        pd.read_parquet(args.vine_forecasts),
        pd.read_parquet(args.core_scores),
        ml_config,
        model_config,
    )
    write_parquet_atomic(args.daily_output, scores)
    audit.update(
        {
            "reporting_scope": scope,
            "inputs": {
                "gaussian_risk_forecasts": artifact_record(args.gaussian_forecasts),
                "vine_risk_forecasts": artifact_record(args.vine_forecasts),
                "core_risk_evaluation_daily": artifact_record(args.core_scores),
                "ml_gaussian_copula_quality": artifact_record(args.gaussian_audit),
                "ml_vine_copula_quality": artifact_record(args.vine_audit),
                "core_model_evaluation": artifact_record(args.core_evaluation_audit),
                "model_config": artifact_record(args.model_config),
                "ml_config": artifact_record(args.ml_config),
                "foundation_status": artifact_record(args.foundation_status),
            },
            "outputs": {
                "ml_risk_evaluation_daily": artifact_record(args.daily_output, rows=len(scores))
            },
        }
    )
    write_json_atomic(args.audit_output, audit)
    interpretation = audit["interpretation"]
    print(
        f"ml_model_evaluation={audit['status']} forecasts={len(scores)} "
        f"evidence={interpretation['evidence']}"
    )
    print(f"Audit: {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
