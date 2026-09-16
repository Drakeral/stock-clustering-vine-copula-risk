#!/usr/bin/env python3
"""Rerun M0--M8 under the frozen WBA DAP-value sensitivity scenarios."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    from scripts.build_gaussian_copula import (
        MODEL_BY_GROUPING,
        build_gaussian_outputs,
    )
    from scripts.build_historical_simulation import build_historical_simulation_outputs
    from scripts.build_marginal_models import build_marginal_outputs
    from scripts.build_vine_copula import VINE_MODEL_BY_GROUPING, build_vine_outputs
    from scripts.evaluate_ml_risk_models import build_ml_evaluation_outputs
    from scripts.evaluate_risk_models import (
        DAILY_SCORE_COLUMNS,
        MODEL_IDS,
        PRIMARY_SCORES,
        calibration_results,
        combine_forecasts,
        dm_results,
        hypothesis_decisions,
        model_summaries,
        score_forecasts,
        validate_evaluation_protocol,
    )
    from scripts.ml_risk_common import (
        ML_GAUSSIAN_MODEL_BY_GROUPING,
        ML_MODEL_IDS,
        ML_VINE_MODEL_BY_GROUPING,
        artifact_record,
        require_same_seed_manifest,
    )
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        write_json_atomic,
        write_parquet_atomic,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from build_gaussian_copula import MODEL_BY_GROUPING, build_gaussian_outputs
    from build_historical_simulation import build_historical_simulation_outputs
    from build_marginal_models import build_marginal_outputs
    from build_vine_copula import VINE_MODEL_BY_GROUPING, build_vine_outputs
    from evaluate_ml_risk_models import build_ml_evaluation_outputs
    from evaluate_risk_models import (
        DAILY_SCORE_COLUMNS,
        MODEL_IDS,
        PRIMARY_SCORES,
        calibration_results,
        combine_forecasts,
        dm_results,
        hypothesis_decisions,
        model_summaries,
        score_forecasts,
        validate_evaluation_protocol,
    )
    from ml_risk_common import (
        ML_GAUSSIAN_MODEL_BY_GROUPING,
        ML_MODEL_IDS,
        ML_VINE_MODEL_BY_GROUPING,
        artifact_record,
        require_same_seed_manifest,
    )
    from pipeline_io import (
        PROJECT_ROOT,
        project_path,
        reporting_scope,
        require_current_hash_records,
        write_json_atomic,
        write_parquet_atomic,
    )


ALL_MODEL_IDS = (*MODEL_IDS, *ML_MODEL_IDS)
ALL_GAUSSIAN_MODELS = {**MODEL_BY_GROUPING, **ML_GAUSSIAN_MODEL_BY_GROUPING}
ALL_VINE_MODELS = {**VINE_MODEL_BY_GROUPING, **ML_VINE_MODEL_BY_GROUPING}
GROUP_ASSIGNMENT_FIELDS = {
    "gics_sector": "gics_sector",
    "hierarchical_cluster": "hierarchical_cluster",
    "spectral_cluster": "spectral_cluster",
    "pca_kmeans_cluster": "pca_kmeans_cluster",
}
RISK_COLUMNS = ("var_95", "var_975", "var_99", "es_975")
SENSITIVITY_ANALYSIS_ROLE = "corporate_action_valuation_sensitivity"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    dap_value_per_old_share: float


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a table")
    return cast(Mapping[str, Any], value)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, project_path(path))


def _require_exact(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs from the frozen sensitivity protocol")


def validate_sensitivity_protocol(
    sensitivity_config: Mapping[str, Any],
    corporate_actions: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[Scenario, ...]]:
    """Require the exact pre-result WBA event, scenarios, and execution scope."""

    _require_exact(sensitivity_config.get("schema_version"), 1, "schema_version")
    _require_exact(
        sensitivity_config.get("protocol_status"),
        "frozen_before_sensitivity_results",
        "protocol_status",
    )
    _require_exact(
        sensitivity_config.get("analysis_role"),
        SENSITIVITY_ANALYSIS_ROLE,
        "analysis_role",
    )
    event = _mapping(sensitivity_config.get("event"), "event")
    execution = _mapping(sensitivity_config.get("execution"), "execution")
    evaluation = _mapping(sensitivity_config.get("evaluation"), "evaluation")
    frozen_event = {
        "action_id": "WBA_2025_SYCAMORE_CASH_DAP_ACQUISITION",
        "ticker": "WBA",
        "effective_date": "2025-08-28",
        "cash_per_old_share": 11.45,
        "primary_dap_value_per_old_share": 0.53,
        "return_formula": ("(cash_per_old_share + dap_value_per_old_share) / previous_close - 1"),
    }
    for key, expected in frozen_event.items():
        _require_exact(event.get(key), expected, f"event.{key}")
    frozen_execution = {
        "model_ids": list(ALL_MODEL_IDS),
        "grouping_ids": list(GROUP_ASSIGNMENT_FIELDS),
        "universe_variant": "security_primary",
        "vine_truncation_tree": 3,
        "common_random_numbers": "reuse_primary_simulation_seed_manifest",
        "annual_assignments": "hold_primary_assignments_fixed",
        "affected_group_return_rule": (
            "primary_group_simple_return_plus_stock_return_delta_divided_by_group_size"
        ),
        "full_vine_cross_sensitivity": False,
    }
    for key, expected in frozen_execution.items():
        _require_exact(execution.get(key), expected, f"execution.{key}")
    frozen_evaluation = {
        "primary_reference_dap_value_per_old_share": 0.53,
        "core_inference": "rerun_frozen_calibration_dm_and_ranking_without_retesting_h1",
        "ml_inference": "rerun_frozen_exploratory_bh_dm_and_descriptive_calibration",
        "h1_treatment": ("unchanged_not_retested_because_annual_assignments_are_held_fixed"),
        "comparison": "scenario_minus_primary",
        "result_role": "robustness_not_new_confirmatory_hypothesis",
    }
    for key, expected in frozen_evaluation.items():
        _require_exact(evaluation.get(key), expected, f"evaluation.{key}")

    raw_scenarios = sensitivity_config.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise TypeError("scenarios must be an array of tables")
    scenarios = tuple(
        Scenario(
            scenario_id=str(_mapping(item, "scenario")["scenario_id"]),
            dap_value_per_old_share=float(_mapping(item, "scenario")["dap_value_per_old_share"]),
        )
        for item in raw_scenarios
    )
    _require_exact(
        [(item.scenario_id, item.dap_value_per_old_share) for item in scenarios],
        [("dap_zero", 0.0), ("dap_cap", 3.0)],
        "scenarios",
    )

    actions = corporate_actions.get("actions")
    if not isinstance(actions, list):
        raise TypeError("manual corporate actions must contain an actions array")
    matches = [
        _mapping(action, "corporate action")
        for action in actions
        if isinstance(action, Mapping) and action.get("action_id") == event["action_id"]
    ]
    if len(matches) != 1:
        raise ValueError("the frozen WBA action is absent or duplicated")
    action = matches[0]
    expected_action = {
        "event_type": "cash_acquisition",
        "research_ticker": event["ticker"],
        "effective_date": event["effective_date"],
        "primary_shares_per_old_share": 0.0,
        "cash_per_old_share": event["cash_per_old_share"],
        "contingent_value_per_old_share": event["primary_dap_value_per_old_share"],
        "contingent_value_sensitivity": {"low": 0.0, "primary": 0.53, "high": 3.0},
    }
    for key, expected in expected_action.items():
        _require_exact(action.get(key), expected, f"manual corporate action {key}")
    return action, scenarios


def _year_record(assignments: Mapping[str, Any], year: int) -> Mapping[str, Any]:
    years = assignments.get("years")
    if not isinstance(years, list):
        raise TypeError("assignment artifact must contain a years array")
    matches = [
        _mapping(item, "assignment year")
        for item in years
        if isinstance(item, Mapping) and item.get("year") == year
    ]
    if len(matches) != 1:
        raise ValueError(f"assignment artifact has no unique {year} record")
    return matches[0]


def _wba_group(assignments: Mapping[str, Any], year: int, field: str) -> tuple[str, list[str]]:
    annual = _year_record(assignments, year)
    rows = annual.get("assignments")
    if not isinstance(rows, list):
        raise TypeError("annual assignment record must contain assignments")
    normalized = [_mapping(row, "security assignment") for row in rows]
    matches = [row for row in normalized if row.get("ticker") == "WBA"]
    if len(matches) != 1 or field not in matches[0]:
        raise ValueError(f"WBA has no unique {field} assignment")
    tickers = [str(row["ticker"]) for row in normalized]
    if len(tickers) != len(set(tickers)):
        raise ValueError("annual assignments contain duplicate tickers")
    return str(matches[0][field]), tickers


def construct_scenario_inputs(
    stock_returns: pd.DataFrame,
    security_daily: pd.DataFrame,
    core_group_returns: pd.DataFrame,
    ml_group_returns: pd.DataFrame,
    core_assignments: Mapping[str, Any],
    ml_assignments: Mapping[str, Any],
    scenario: Scenario,
    event: Mapping[str, Any],
    *,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply exactly one stock-return shock and propagate it through fixed groups."""

    date = pd.Timestamp(str(event["effective_date"]))
    year = int(date.year)
    ticker = str(event["research_ticker"])
    stock = stock_returns.copy()
    stock.index = pd.to_datetime(stock.index, errors="raise")
    if ticker not in stock.columns or date not in stock.index:
        raise ValueError("WBA event observation is absent from the stock return panel")
    event_rows = security_daily.loc[
        (pd.to_datetime(security_daily["date"], errors="raise") == date)
        & (security_daily["research_ticker"].astype(str) == ticker)
    ]
    if len(event_rows) != 1:
        raise ValueError("security daily panel has no unique WBA event row")
    event_row = event_rows.iloc[0]
    if (
        event_row["manual_action_id"] != event["action_id"]
        or event_row["manual_action_type"] != "cash_acquisition"
        or not bool(event_row["is_synthetic"])
    ):
        raise ValueError("security daily WBA row is not the frozen synthetic cash acquisition")
    previous_close = float(event_row["previous_close"])
    if not np.isfinite(previous_close) or previous_close <= 0:
        raise ValueError("WBA event previous close must be positive and finite")
    cash = float(event["cash_per_old_share"])
    primary_dap = float(event["contingent_value_per_old_share"])
    primary_return = (cash + primary_dap) / previous_close - 1.0
    observed_primary_return = float(stock.at[date, ticker])
    if not np.isclose(observed_primary_return, primary_return, rtol=0.0, atol=tolerance):
        raise ValueError("primary WBA panel return does not reproduce the frozen formula")
    scenario_return = (cash + scenario.dap_value_per_old_share) / previous_close - 1.0
    return_delta = scenario_return - observed_primary_return
    stock.at[date, ticker] = scenario_return
    changed_stock_cells = int(
        np.count_nonzero(
            ~np.isclose(
                stock.to_numpy(dtype=float),
                stock_returns.to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )
    )
    if changed_stock_cells != 1:
        raise AssertionError("the scenario must change exactly one stock-return cell")

    core = core_group_returns.loc[
        core_group_returns["universe_variant"].eq("security_primary")
    ].copy()
    ml = ml_group_returns.loc[ml_group_returns["universe_variant"].eq("security_primary")].copy()
    groups = pd.concat([core, ml], ignore_index=True)
    groups["date"] = pd.to_datetime(groups["date"], errors="raise")
    group_records: list[dict[str, Any]] = []
    group_ticker_sets: dict[str, list[str]] = {}
    for grouping_id, field in GROUP_ASSIGNMENT_FIELDS.items():
        source = (
            ml_assignments if grouping_id in ML_GAUSSIAN_MODEL_BY_GROUPING else core_assignments
        )
        group_id, tickers = _wba_group(source, year, field)
        group_ticker_sets[grouping_id] = tickers
        mask = (
            groups["date"].eq(date)
            & groups["year"].eq(year)
            & groups["sample_role"].eq("evaluation")
            & groups["grouping_id"].eq(grouping_id)
            & groups["group_id"].eq(group_id)
        )
        if int(mask.sum()) != 1:
            raise ValueError(f"no unique affected group-return row for {grouping_id}")
        row_index = groups.index[mask][0]
        group_size = int(groups.at[row_index, "group_size"])
        old_group_return = float(groups.at[row_index, "simple_return"])
        new_group_return = old_group_return + return_delta / group_size
        if new_group_return <= -1.0 or not np.isfinite(new_group_return):
            raise ValueError("scenario creates an invalid affected group return")
        groups.at[row_index, "simple_return"] = new_group_return
        groups.at[row_index, "log_return"] = np.log1p(new_group_return)
        group_records.append(
            {
                "grouping_id": grouping_id,
                "group_id": group_id,
                "group_size": group_size,
                "primary_group_simple_return": old_group_return,
                "scenario_group_simple_return": new_group_return,
                "group_simple_return_delta": new_group_return - old_group_return,
            }
        )

    reference_tickers = group_ticker_sets["gics_sector"]
    if any(tickers != reference_tickers for tickers in group_ticker_sets.values()):
        raise ValueError("fixed grouping artifacts disagree on the annual active security set")
    direct_portfolio_return = float(stock.loc[date, reference_tickers].mean())
    identity_errors: dict[str, float] = {}
    for grouping_id in GROUP_ASSIGNMENT_FIELDS:
        rows = groups.loc[
            groups["date"].eq(date)
            & groups["year"].eq(year)
            & groups["sample_role"].eq("evaluation")
            & groups["grouping_id"].eq(grouping_id)
        ]
        grouped_portfolio_return = float((rows["portfolio_weight"] * rows["simple_return"]).sum())
        error = abs(grouped_portfolio_return - direct_portfolio_return)
        identity_errors[grouping_id] = error
        if error > tolerance:
            raise ValueError(f"{grouping_id} scenario portfolio identity failed")
    construction = {
        "scenario_id": scenario.scenario_id,
        "event_date": date.date().isoformat(),
        "ticker": ticker,
        "cash_per_old_share": cash,
        "dap_value_per_old_share": scenario.dap_value_per_old_share,
        "total_consideration_per_old_share": cash + scenario.dap_value_per_old_share,
        "previous_close": previous_close,
        "primary_simple_return": observed_primary_return,
        "scenario_simple_return": scenario_return,
        "simple_return_delta": return_delta,
        "changed_stock_return_cell_count": changed_stock_cells,
        "annual_active_security_count": len(reference_tickers),
        "scenario_portfolio_simple_return": direct_portfolio_return,
        "maximum_group_portfolio_identity_error": max(identity_errors.values()),
        "group_portfolio_identity_errors": identity_errors,
        "affected_groups": group_records,
        "annual_assignments_reestimated": False,
    }
    return stock, groups.sort_values(["year", "date", "grouping_id", "group_id"]), construction


def _core_vine_eligible(vine_audit: Mapping[str, Any]) -> bool:
    fractions = _mapping(
        vine_audit.get("model_whole_vine_fallback_date_fractions"),
        "vine fallback fractions",
    )
    limit = float(vine_audit["maximum_whole_vine_fallback_date_fraction"])
    return all(float(fractions[model_id]) <= limit for model_id in ("M2", "M4"))


def _compact_quality(audit: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: audit[field] for field in fields if field in audit}


def evaluate_scenario(
    stock_returns: pd.DataFrame,
    group_returns: pd.DataFrame,
    active_schedule: Mapping[str, Any],
    primary_seed_manifest: Mapping[str, Any],
    model_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    ml_config: Mapping[str, Any],
    *,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit and evaluate every primary and exploratory model for one scenario."""

    protocol = validate_evaluation_protocol(evaluation_config, model_config)
    historical, windows, historical_audit = build_historical_simulation_outputs(
        stock_returns, active_schedule, model_config
    )
    marginal_refits, daily_margins, training_pits, marginal_audit = build_marginal_outputs(
        group_returns,
        model_config,
        universe_variant="security_primary",
        progress_every=progress_every,
    )
    gaussian_refits, gaussian_forecasts, generated_seed_manifest, gaussian_audit = (
        build_gaussian_outputs(
            training_pits,
            marginal_refits,
            daily_margins,
            group_returns,
            model_config,
            universe_variant="security_primary",
            progress_every=progress_every,
            model_by_grouping=ALL_GAUSSIAN_MODELS,
        )
    )
    require_same_seed_manifest(generated_seed_manifest, primary_seed_manifest)
    vine_refits, vine_forecasts, vine_audit = build_vine_outputs(
        training_pits,
        marginal_refits,
        daily_margins,
        group_returns,
        gaussian_refits,
        primary_seed_manifest,
        model_config,
        universe_variant="security_primary",
        truncation_level=3,
        progress_every=progress_every,
        model_by_grouping=ALL_VINE_MODELS,
        analysis_role=SENSITIVITY_ANALYSIS_ROLE,
    )
    quality_audits = (historical_audit, marginal_audit, gaussian_audit, vine_audit)
    if any(audit.get("status") != "pass" for audit in quality_audits):
        failures = [
            audit.get("gate_name") for audit in quality_audits if audit.get("status") != "pass"
        ]
        raise RuntimeError(f"scenario model quality failed: {failures}")

    core_gaussian = gaussian_forecasts.loc[gaussian_forecasts["model_id"].isin(["M1", "M3"])]
    core_vine = vine_forecasts.loc[vine_forecasts["model_id"].isin(["M2", "M4"])]
    core_forecasts, forecast_validation = combine_forecasts(
        historical, core_gaussian, core_vine, protocol["evaluation"]
    )
    core_scores = score_forecasts(core_forecasts)
    calibration, annual_calibration = calibration_results(core_scores, protocol["calibration"])
    dm = dm_results(core_scores, protocol["h2"])
    summaries = model_summaries(core_scores, calibration)
    placeholder_h1 = {"decision": "not_retested"}
    decisions = hypothesis_decisions(
        placeholder_h1,
        dm,
        summaries,
        vine_eligible=_core_vine_eligible(vine_audit),
    )
    decisions["H1_cluster_information"] = {
        "status": "unchanged_not_retested",
        "rule": "annual assignments held fixed in this corporate-action valuation sensitivity",
    }

    ml_gaussian = gaussian_forecasts.loc[gaussian_forecasts["model_id"].isin(["M5", "M7"])]
    ml_vine = vine_forecasts.loc[vine_forecasts["model_id"].isin(["M6", "M8"])]
    ml_scores, ml_evaluation = build_ml_evaluation_outputs(
        ml_gaussian, ml_vine, core_scores, ml_config, model_config
    )
    scores = pd.concat([core_scores, ml_scores], ignore_index=True)
    model_order = {model_id: index for index, model_id in enumerate(ALL_MODEL_IDS)}
    scores["_model_order"] = scores["model_id"].map(model_order)
    scores = scores.sort_values(["date", "_model_order"]).drop(columns="_model_order")
    if set(scores["model_id"]) != set(ALL_MODEL_IDS):
        raise AssertionError("scenario output does not contain exactly M0--M8")

    quality = {
        "historical": _compact_quality(
            historical_audit,
            ("status", "forecast_count", "minimum_training_observations", "issues"),
        ),
        "marginal": _compact_quality(
            marginal_audit,
            (
                "status",
                "refit_count",
                "selected_method_counts",
                "fallback_fit_count",
                "ewma_fit_count",
                "ewma_fit_fraction",
                "issues",
            ),
        ),
        "gaussian": _compact_quality(
            gaussian_audit,
            (
                "status",
                "model_ids",
                "refit_count",
                "repaired_refit_count",
                "maximum_condition_number",
                "maximum_realised_portfolio_identity_error",
                "issues",
            ),
        ),
        "vine": _compact_quality(
            vine_audit,
            (
                "status",
                "model_ids",
                "refit_count",
                "failed_pair_count",
                "whole_vine_fallback_refit_count",
                "model_whole_vine_fallback_date_fractions",
                "maximum_realised_portfolio_identity_error",
                "issues",
            ),
        ),
    }
    evaluation = {
        "forecast_validation": forecast_validation,
        "core_model_summaries": summaries,
        "core_full_period_calibration": calibration,
        "core_annual_calibration_descriptive": annual_calibration,
        "core_diebold_mariano_comparisons": dm,
        "core_hypotheses": decisions,
        "ml_model_summaries": ml_evaluation["model_summaries"],
        "ml_diebold_mariano_comparisons": ml_evaluation["diebold_mariano_comparisons"],
        "ml_full_period_calibration_descriptive": (
            ml_evaluation["full_period_calibration_descriptive"]
        ),
        "ml_interpretation": ml_evaluation["interpretation"],
    }
    del windows, daily_margins, training_pits
    gc.collect()
    return (
        scores.reset_index(drop=True),
        marginal_refits,
        gaussian_refits,
        vine_refits,
        {"quality": quality, "evaluation": evaluation},
    )


def compare_with_primary(
    scenario_scores: pd.DataFrame,
    primary_scores: pd.DataFrame,
    *,
    event_date: pd.Timestamp,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize scenario-minus-primary changes and enforce pre-event forecast identity."""

    keys = ["date", "model_id"]
    scenario = scenario_scores.copy()
    primary = primary_scores.copy()
    scenario["date"] = pd.to_datetime(scenario["date"], errors="raise")
    primary["date"] = pd.to_datetime(primary["date"], errors="raise")
    if set(scenario.columns) != set(DAILY_SCORE_COLUMNS) or set(primary.columns) != set(
        DAILY_SCORE_COLUMNS
    ):
        raise ValueError("sensitivity comparison requires the frozen daily-score schema")
    if scenario.duplicated(keys).any() or primary.duplicated(keys).any():
        raise ValueError("sensitivity comparison contains duplicate model-date rows")
    merged = scenario.merge(
        primary, on=keys, how="outer", suffixes=("_scenario", "_primary"), indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("scenario and primary score panels do not have matched model dates")

    before_or_event = merged["date"].le(event_date)
    pre_event_errors = {
        column: float(
            np.max(
                np.abs(
                    merged.loc[before_or_event, f"{column}_scenario"].to_numpy(dtype=float)
                    - merged.loc[before_or_event, f"{column}_primary"].to_numpy(dtype=float)
                )
            )
        )
        for column in RISK_COLUMNS
    }
    if max(pre_event_errors.values()) > tolerance:
        raise ValueError(
            "risk forecasts changed before information from the DAP event was available"
        )
    event_loss_deltas = merged.loc[
        merged["date"].eq(event_date), "realised_loss_scenario"
    ].to_numpy(dtype=float) - merged.loc[
        merged["date"].eq(event_date), "realised_loss_primary"
    ].to_numpy(dtype=float)
    if len(event_loss_deltas) != len(ALL_MODEL_IDS) or np.ptp(event_loss_deltas) > tolerance:
        raise ValueError("models do not share one scenario event-loss shock")

    rows: list[dict[str, Any]] = []
    for model_id in ALL_MODEL_IDS:
        model = merged.loc[merged["model_id"].eq(model_id)]
        row: dict[str, Any] = {
            "model_id": model_id,
            "observations": len(model),
            "event_realised_loss_delta": float(
                model.loc[model["date"].eq(event_date), "realised_loss_scenario"].iloc[0]
                - model.loc[model["date"].eq(event_date), "realised_loss_primary"].iloc[0]
            ),
        }
        for column in (*RISK_COLUMNS, *PRIMARY_SCORES):
            scenario_values = model[f"{column}_scenario"].to_numpy(dtype=float)
            primary_values = model[f"{column}_primary"].to_numpy(dtype=float)
            differences = scenario_values - primary_values
            row[f"primary_mean_{column}"] = float(primary_values.mean())
            row[f"scenario_mean_{column}"] = float(scenario_values.mean())
            row[f"mean_delta_{column}"] = float(differences.mean())
            row[f"maximum_absolute_delta_{column}"] = float(np.max(np.abs(differences)))
            row[f"changed_date_count_{column}"] = int(
                np.count_nonzero(np.abs(differences) > tolerance)
            )
        for suffix in ("95", "975", "99"):
            column = f"exception_{suffix}"
            scenario_count = int(model[f"{column}_scenario"].sum())
            primary_count = int(model[f"{column}_primary"].sum())
            row[f"primary_{column}_count"] = primary_count
            row[f"scenario_{column}_count"] = scenario_count
            row[f"delta_{column}_count"] = scenario_count - primary_count
        rows.append(row)
    identity = {
        "risk_forecast_identity_period": f"through_{event_date.date().isoformat()}_inclusive",
        "maximum_pre_event_risk_forecast_errors": pre_event_errors,
        "risk_forecast_identity_tolerance": tolerance,
        "common_event_realised_loss_delta": float(event_loss_deltas[0]),
    }
    return rows, identity


def _require_artifact_binding(
    audit: Mapping[str, Any], section: str, artifact_key: str, path: Path, audit_name: str
) -> None:
    require_current_hash_records(audit, source_name=audit_name)
    if audit.get("status") != "pass":
        raise RuntimeError(f"{audit_name} is not passed")
    records = _mapping(audit.get(section), f"{audit_name}.{section}")
    record = _mapping(records.get(artifact_key), f"{audit_name}.{artifact_key}")
    current = artifact_record(path)
    if record.get("sha256") != current["sha256"]:
        raise RuntimeError(f"{audit_name} does not bind the current {artifact_key}")


def _require_output_binding(
    audit: Mapping[str, Any], output_key: str, path: Path, audit_name: str
) -> None:
    _require_artifact_binding(audit, "outputs", output_key, path, audit_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensitivity-config",
        type=Path,
        default=PROJECT_ROOT / "config/wba_dap_sensitivity.toml",
    )
    parser.add_argument(
        "--corporate-actions",
        type=Path,
        default=PROJECT_ROOT / "config/manual_corporate_actions.json",
    )
    parser.add_argument(
        "--model-config", type=Path, default=PROJECT_ROOT / "config/model_config.toml"
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=PROJECT_ROOT / "config/evaluation_config.toml",
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
        "--security-daily",
        type=Path,
        default=PROJECT_ROOT / "data/processed/security_daily.parquet",
    )
    parser.add_argument(
        "--stock-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/portfolio_constituent_simple_returns.parquet",
    )
    parser.add_argument(
        "--active-schedule",
        type=Path,
        default=PROJECT_ROOT / "data/processed/active_universe_by_year.json",
    )
    parser.add_argument(
        "--core-group-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/annual_group_returns.parquet",
    )
    parser.add_argument(
        "--ml-group-returns",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_annual_group_returns.parquet",
    )
    parser.add_argument(
        "--core-assignments",
        type=Path,
        default=PROJECT_ROOT / "data/processed/annual_group_assignments.json",
    )
    parser.add_argument(
        "--ml-assignments",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_annual_group_assignments.json",
    )
    parser.add_argument(
        "--primary-core-scores",
        type=Path,
        default=PROJECT_ROOT / "data/processed/risk_evaluation_daily.parquet",
    )
    parser.add_argument(
        "--primary-ml-scores",
        type=Path,
        default=PROJECT_ROOT / "data/processed/ml_risk_evaluation_daily.parquet",
    )
    parser.add_argument(
        "--primary-seed-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/simulation_seed_manifest.json",
    )
    parser.add_argument(
        "--clustering-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/clustering_diagnostics.json",
    )
    parser.add_argument(
        "--ml-clustering-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_clustering_diagnostics.json",
    )
    parser.add_argument(
        "--core-evaluation-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/model_evaluation.json",
    )
    parser.add_argument(
        "--ml-evaluation-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/ml_model_evaluation.json",
    )
    parser.add_argument(
        "--primary-gaussian-audit",
        type=Path,
        default=PROJECT_ROOT / "data/audit/gaussian_copula_quality.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "data/processed/wba_dap_sensitivity",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/wba_dap_sensitivity.json",
    )
    parser.add_argument("--progress-every", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    with args.sensitivity_config.open("rb") as handle:
        sensitivity_config = tomllib.load(handle)
    with args.model_config.open("rb") as handle:
        model_config = tomllib.load(handle)
    with args.evaluation_config.open("rb") as handle:
        evaluation_config = tomllib.load(handle)
    with args.ml_config.open("rb") as handle:
        ml_config = tomllib.load(handle)
    corporate_actions = _load_json(args.corporate_actions)
    event, scenarios = validate_sensitivity_protocol(sensitivity_config, corporate_actions)
    validate_evaluation_protocol(evaluation_config, model_config)

    foundation = _load_json(args.foundation_status)
    require_current_hash_records(foundation, source_name=project_path(args.foundation_status))
    scope = reporting_scope(foundation)
    clustering_audit = _load_json(args.clustering_audit)
    ml_clustering_audit = _load_json(args.ml_clustering_audit)
    core_evaluation_audit = _load_json(args.core_evaluation_audit)
    ml_evaluation_audit = _load_json(args.ml_evaluation_audit)
    primary_gaussian_audit = _load_json(args.primary_gaussian_audit)
    _require_output_binding(
        clustering_audit,
        "annual_group_returns",
        args.core_group_returns,
        project_path(args.clustering_audit),
    )
    _require_output_binding(
        clustering_audit,
        "annual_group_assignments",
        args.core_assignments,
        project_path(args.clustering_audit),
    )
    _require_output_binding(
        ml_clustering_audit,
        "ml_annual_group_returns",
        args.ml_group_returns,
        project_path(args.ml_clustering_audit),
    )
    _require_output_binding(
        ml_clustering_audit,
        "ml_annual_group_assignments",
        args.ml_assignments,
        project_path(args.ml_clustering_audit),
    )
    _require_output_binding(
        core_evaluation_audit,
        "risk_evaluation_daily",
        args.primary_core_scores,
        project_path(args.core_evaluation_audit),
    )
    _require_output_binding(
        ml_evaluation_audit,
        "ml_risk_evaluation_daily",
        args.primary_ml_scores,
        project_path(args.ml_evaluation_audit),
    )
    _require_artifact_binding(
        clustering_audit,
        "inputs",
        "simple_return_panel",
        args.stock_returns,
        project_path(args.clustering_audit),
    )
    _require_artifact_binding(
        clustering_audit,
        "inputs",
        "active_universe",
        args.active_schedule,
        project_path(args.clustering_audit),
    )
    _require_output_binding(
        primary_gaussian_audit,
        "simulation_seed_manifest",
        args.primary_seed_manifest,
        project_path(args.primary_gaussian_audit),
    )
    for audit in (
        clustering_audit,
        ml_clustering_audit,
        core_evaluation_audit,
        ml_evaluation_audit,
        primary_gaussian_audit,
    ):
        if audit.get("reporting_scope") != scope:
            raise RuntimeError("sensitivity upstream audit reporting scopes disagree")

    stock_returns = pd.read_parquet(args.stock_returns)
    security_daily = pd.read_parquet(args.security_daily)
    core_group_returns = pd.read_parquet(args.core_group_returns)
    ml_group_returns = pd.read_parquet(args.ml_group_returns)
    core_assignments = _load_json(args.core_assignments)
    ml_assignments = _load_json(args.ml_assignments)
    active_schedule = _load_json(args.active_schedule)
    primary_seed_manifest = _load_json(args.primary_seed_manifest)
    primary_scores = pd.concat(
        [pd.read_parquet(args.primary_core_scores), pd.read_parquet(args.primary_ml_scores)],
        ignore_index=True,
    ).loc[:, DAILY_SCORE_COLUMNS]
    tolerance = float(model_config["portfolio"]["reconstruction_tolerance"])
    event_date = pd.Timestamp(str(event["effective_date"]))

    scenario_audits: list[dict[str, Any]] = []
    output_records: dict[str, Any] = {}
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        print(f"wba_dap_sensitivity_start={scenario.scenario_id}", flush=True)
        scenario_stock, scenario_groups, construction = construct_scenario_inputs(
            stock_returns,
            security_daily,
            core_group_returns,
            ml_group_returns,
            core_assignments,
            ml_assignments,
            scenario,
            event,
            tolerance=tolerance,
        )
        scores, marginal_refits, gaussian_refits, vine_refits, model_audit = evaluate_scenario(
            scenario_stock,
            scenario_groups,
            active_schedule,
            primary_seed_manifest,
            model_config,
            evaluation_config,
            ml_config,
            progress_every=args.progress_every,
        )
        comparison, forecast_identity = compare_with_primary(
            scores,
            primary_scores,
            event_date=event_date,
            tolerance=tolerance,
        )
        scenario_directory = args.output_directory / scenario.scenario_id
        paths = {
            "daily_scores": scenario_directory / "daily_scores.parquet",
            "marginal_refits": scenario_directory / "marginal_refits.parquet",
            "gaussian_refits": scenario_directory / "gaussian_refits.parquet",
            "vine_refits": scenario_directory / "vine_refits.parquet",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(paths["daily_scores"], scores)
        write_parquet_atomic(paths["marginal_refits"], marginal_refits)
        write_parquet_atomic(paths["gaussian_refits"], gaussian_refits)
        write_parquet_atomic(paths["vine_refits"], vine_refits)
        scenario_outputs = {
            name: artifact_record(path, rows=len(frame))
            for name, path, frame in (
                ("daily_scores", paths["daily_scores"], scores),
                ("marginal_refits", paths["marginal_refits"], marginal_refits),
                ("gaussian_refits", paths["gaussian_refits"], gaussian_refits),
                ("vine_refits", paths["vine_refits"], vine_refits),
            )
        }
        output_records[scenario.scenario_id] = scenario_outputs
        scenario_audits.append(
            {
                "scenario_id": scenario.scenario_id,
                "construction": construction,
                "forecast_identity": forecast_identity,
                "primary_comparison_by_model": comparison,
                **model_audit,
                "outputs": scenario_outputs,
            }
        )
        print(
            f"wba_dap_sensitivity_complete={scenario.scenario_id} scores={len(scores)} ",
            f"marginal_refits={len(marginal_refits)} gaussian_refits={len(gaussian_refits)} ",
            f"vine_refits={len(vine_refits)}",
            flush=True,
        )
        del scenario_stock, scenario_groups, scores, marginal_refits, gaussian_refits, vine_refits
        gc.collect()

    primary_hypotheses = _mapping(
        core_evaluation_audit.get("hypotheses"), "primary core hypotheses"
    )
    primary_ml_interpretation = _mapping(
        ml_evaluation_audit.get("interpretation"), "primary ML interpretation"
    )
    conclusion_stability = []
    for scenario_audit in scenario_audits:
        scenario_hypotheses = scenario_audit["evaluation"]["core_hypotheses"]
        scenario_ml = scenario_audit["evaluation"]["ml_interpretation"]
        conclusion_stability.append(
            {
                "scenario_id": scenario_audit["scenario_id"],
                "H2_primary_status": primary_hypotheses["H2_copula_specification"]["status"],
                "H2_scenario_status": scenario_hypotheses["H2_copula_specification"]["status"],
                "H3_primary_status": primary_hypotheses["H3_integrated_model"]["status"],
                "H3_scenario_status": scenario_hypotheses["H3_integrated_model"]["status"],
                "ml_primary_evidence": primary_ml_interpretation["evidence"],
                "ml_scenario_evidence": scenario_ml["evidence"],
            }
        )

    audit = {
        "schema_version": 1,
        "gate_name": "wba_dap_sensitivity_v1",
        "analysis_role": SENSITIVITY_ANALYSIS_ROLE,
        "status": "pass",
        "reporting_scope": scope,
        "issues": [],
        "method": {
            "event_action_id": event["action_id"],
            "primary_dap_value_per_old_share": event["contingent_value_per_old_share"],
            "scenario_dap_values_per_old_share": {
                scenario.scenario_id: scenario.dap_value_per_old_share for scenario in scenarios
            },
            "annual_assignments": "held fixed",
            "models_rerun": list(ALL_MODEL_IDS),
            "vine_truncation_tree": 3,
            "full_vine_cross_sensitivity": False,
            "h1": "unchanged and not retested",
        },
        "primary_reference": {
            "core_hypotheses": primary_hypotheses,
            "ml_interpretation": primary_ml_interpretation,
        },
        "conclusion_stability": conclusion_stability,
        "scenarios": scenario_audits,
        "inputs": {
            "sensitivity_config": artifact_record(args.sensitivity_config),
            "corporate_actions": artifact_record(args.corporate_actions),
            "model_config": artifact_record(args.model_config),
            "evaluation_config": artifact_record(args.evaluation_config),
            "ml_config": artifact_record(args.ml_config),
            "foundation_status": artifact_record(args.foundation_status),
            "security_daily": artifact_record(args.security_daily),
            "stock_returns": artifact_record(args.stock_returns),
            "active_schedule": artifact_record(args.active_schedule),
            "core_group_returns": artifact_record(args.core_group_returns),
            "ml_group_returns": artifact_record(args.ml_group_returns),
            "core_assignments": artifact_record(args.core_assignments),
            "ml_assignments": artifact_record(args.ml_assignments),
            "primary_core_scores": artifact_record(args.primary_core_scores),
            "primary_ml_scores": artifact_record(args.primary_ml_scores),
            "primary_seed_manifest": artifact_record(args.primary_seed_manifest),
            "clustering_audit": artifact_record(args.clustering_audit),
            "ml_clustering_audit": artifact_record(args.ml_clustering_audit),
            "core_evaluation_audit": artifact_record(args.core_evaluation_audit),
            "ml_evaluation_audit": artifact_record(args.ml_evaluation_audit),
            "primary_gaussian_audit": artifact_record(args.primary_gaussian_audit),
        },
        "outputs": output_records,
    }
    write_json_atomic(args.audit_output, audit)
    print(f"wba_dap_sensitivity=pass scenarios={len(scenarios)} audit={args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
