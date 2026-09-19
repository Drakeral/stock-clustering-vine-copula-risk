#!/usr/bin/env python3
"""Build deterministic, report-ready tables and figures from audited results."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from scripts.pipeline_io import (
        PROJECT_ROOT,
        require_current_hash_records,
        sha256_file,
        temporary_sibling,
        write_csv_atomic,
        write_json_atomic,
        write_text_atomic,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import (  # type: ignore[no-redef]
        PROJECT_ROOT,
        require_current_hash_records,
        sha256_file,
        temporary_sibling,
        write_csv_atomic,
        write_json_atomic,
        write_text_atomic,
    )

REPORTING_SCOPE = "provisional_research_results"
TABLE_DIRECTORY = "tables"
FIGURE_DIRECTORY = "figures"
MANIFEST_NAME = "report_manifest.json"
SVG_METADATA = {"Date": None, "Creator": "FE5110 deterministic report builder"}

CORE_PERFORMANCE_FIELDS = [
    "model_id",
    "model_label",
    "grouping_id",
    "observations",
    "mean_quantile_loss_95",
    "mean_quantile_loss_99",
    "mean_fz0_975",
    "exceptions_95",
    "exception_rate_95",
    "exceptions_975",
    "exception_rate_975",
    "exceptions_99",
    "exception_rate_99",
    "average_primary_score_rank",
    "overall_rank",
    "adjusted_calibration_rejection_count",
    "systematic_calibration_failure",
]
ML_PERFORMANCE_FIELDS = [
    "model_id",
    "model_label",
    "grouping_id",
    "observations",
    "mean_quantile_loss_95",
    "mean_quantile_loss_99",
    "mean_fz0_975",
    "exceptions_95",
    "exception_rate_95",
    "exceptions_975",
    "exception_rate_975",
    "exceptions_99",
    "exception_rate_99",
    "mean_copula_log_score",
]
DM_FIELDS = [
    "comparison_id",
    "grouping_id",
    "vine_model",
    "gaussian_model",
    "score",
    "observations",
    "mean_loss_difference",
    "dm_statistic",
    "p_value_two_sided",
    "holm_p_value",
    "direction",
    "significant_after_holm",
    "favours_vine_after_holm",
    "favours_gaussian_after_holm",
]
CALIBRATION_FIELDS = [
    "model_id",
    "confidence",
    "observations",
    "exceptions",
    "exception_rate",
    "expected_exceptions",
    "kupiec_p_value",
    "kupiec_holm_p_value",
    "kupiec_reject_after_holm",
    "christoffersen_independence_p_value",
    "christoffersen_independence_holm_p_value",
    "christoffersen_independence_reject_after_holm",
    "conditional_coverage_p_value",
]
HYPOTHESIS_FIELDS = [
    "hypothesis_id",
    "research_claim",
    "decision",
    "decision_rule_or_evidence",
    "key_estimate",
    "uncertainty_or_eligibility",
    "analysis_role",
]
CLUSTERING_FIELDS = [
    "year",
    "method_id",
    "method_label",
    "dependence_gap",
    "within_pair_mean",
    "between_pair_mean",
    "within_pair_count",
    "between_pair_count",
    "gap_minus_gics",
    "ari_vs_previous_year",
    "nmi_vs_gics",
]
ROBUSTNESS_FIELDS = [
    "analysis_id",
    "analysis_role",
    "evidence",
    "adjusted_tests_favouring_alternative",
    "adjusted_tests_favouring_baseline",
    "core_conclusions_revised",
    "detail",
]


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a table/object")
    return value


def _require_sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def load_reporting_config(path: Path) -> dict[str, Any]:
    """Load and fail-closed validate the frozen report specification."""

    with path.open("rb") as handle:
        config = tomllib.load(handle)
    required_scalars = {
        "schema_version": 1,
        "protocol_status": "frozen_before_report_artifact_generation",
        "analysis_role": "descriptive_reporting_only",
        "reporting_scope": REPORTING_SCOPE,
    }
    for key, expected in required_scalars.items():
        if config.get(key) != expected:
            raise ValueError(f"reporting config {key!r} must equal {expected!r}")
    if config.get("float_significant_digits") not in range(6, 18):
        raise ValueError("float_significant_digits must be an integer from 6 through 17")
    expected_orders = {
        "core_model_order": ["M0", "M1", "M2", "M3", "M4"],
        "ml_model_order": ["M5", "M6", "M7", "M8"],
        "clustering_method_order": ["gics", "hierarchical", "spectral", "pca_kmeans"],
    }
    for key, expected in expected_orders.items():
        if config.get(key) != expected:
            raise ValueError(f"reporting config {key!r} differs from the frozen order")
    for section in ("sources", "model_labels", "method_labels", "palette", "artifacts"):
        _require_mapping(config.get(section), f"reporting config [{section}]")
    return config


def _resolve_project_file(project_root: Path, configured_path: object, name: str) -> Path:
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError(f"source {name!r} must be a non-empty project-relative path")
    path = (project_root / configured_path).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"source {name!r} escapes the project root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"source {name!r} is missing: {configured_path}")
    return path


def load_audited_sources(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Load passed, lineage-current audits with the frozen reporting scope."""

    sources = _require_mapping(config["sources"], "reporting sources")
    loaded: dict[str, Any] = {}
    for name, configured_path in sources.items():
        path = _resolve_project_file(project_root, configured_path, name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"source {name!r} must contain a JSON object")
        if payload.get("status") != "pass":
            raise RuntimeError(f"source {name!r} has not passed its quality gate")
        if payload.get("reporting_scope") != config["reporting_scope"]:
            raise RuntimeError(f"source {name!r} has an incompatible reporting scope")
        require_current_hash_records(payload, project_root=project_root, source_name=str(path))
        loaded[name] = payload
    return loaded


def _ordered_records(
    records: object,
    order: Sequence[str],
    *,
    id_field: str,
    source_name: str,
) -> list[dict[str, Any]]:
    rows = _require_sequence(records, source_name)
    by_id = {}
    for row in rows:
        record = dict(_require_mapping(row, f"{source_name} record"))
        identifier = record.get(id_field)
        if identifier in by_id:
            raise ValueError(f"duplicate {id_field}={identifier!r} in {source_name}")
        by_id[identifier] = record
    if set(by_id) != set(order):
        raise ValueError(
            f"{source_name} identifiers differ from frozen order: "
            f"expected {list(order)!r}, observed {sorted(by_id)!r}"
        )
    return [by_id[identifier] for identifier in order]


def build_model_performance_rows(
    evaluation: Mapping[str, Any],
    order: Sequence[str],
    labels: Mapping[str, Any],
    *,
    ml: bool,
) -> list[dict[str, Any]]:
    """Extract ordered core or exploratory-ML model summaries."""

    records = _ordered_records(
        evaluation.get("model_summaries"),
        order,
        id_field="model_id",
        source_name="model_summaries",
    )
    fields = ML_PERFORMANCE_FIELDS if ml else CORE_PERFORMANCE_FIELDS
    output = []
    for record in records:
        model_id = str(record["model_id"])
        row = {field: record.get(field) for field in fields}
        row["model_label"] = labels.get(model_id)
        if not isinstance(row["model_label"], str):
            raise ValueError(f"missing label for model {model_id}")
        output.append(row)
    return output


def build_dm_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract the six frozen confirmatory vine-minus-Gaussian DM tests."""

    records = _require_sequence(evaluation.get("diebold_mariano_comparisons"), "DM comparisons")
    rows = [
        {field: record.get(field) for field in DM_FIELDS}
        for record in (_require_mapping(value, "DM comparison") for value in records)
    ]
    if len(rows) != 6:
        raise ValueError(f"expected six confirmatory DM comparisons, observed {len(rows)}")
    return sorted(rows, key=lambda row: (str(row["grouping_id"]), str(row["score"])))


def build_calibration_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract the frozen 20-test full-period calibration family."""

    output = []
    records = _require_sequence(evaluation.get("full_period_calibration"), "calibration records")
    for value in records:
        record = _require_mapping(value, "calibration record")
        row = {field: record.get(field) for field in CALIBRATION_FIELDS}
        row["expected_exceptions"] = float(record["observations"]) * (
            1.0 - float(record["confidence"])
        )
        output.append(row)
    if len(output) != 10:
        raise ValueError(f"expected ten model-confidence calibration rows, observed {len(output)}")
    return sorted(output, key=lambda row: (str(row["model_id"]), float(row["confidence"])))


def build_hypothesis_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render the frozen H1-H3 decisions without re-running inference."""

    hypotheses = _require_mapping(evaluation.get("hypotheses"), "hypotheses")
    h1 = _require_mapping(evaluation.get("h1_bootstrap"), "H1 bootstrap")
    h1_decision = _require_mapping(hypotheses.get("H1_cluster_information"), "H1 decision")
    h2 = _require_mapping(hypotheses.get("H2_copula_specification"), "H2 decision")
    h3 = _require_mapping(hypotheses.get("H3_integrated_model"), "H3 decision")
    interval = _require_mapping(h1.get("cluster_minus_gics_confidence_interval"), "H1 interval")
    return [
        {
            "hypothesis_id": "H1",
            "research_claim": "Hierarchical clusters separate dependence better than GICS sectors",
            "decision": h1_decision.get("status"),
            "decision_rule_or_evidence": h1_decision.get("rule"),
            "key_estimate": h1.get("mean_cluster_minus_gics_gap"),
            "uncertainty_or_eligibility": (
                f"95% paired block-bootstrap CI [{float(interval['lower']):.6f}, "
                f"{float(interval['upper']):.6f}]"
            ),
            "analysis_role": "confirmatory_methodological",
        },
        {
            "hypothesis_id": "H2",
            "research_claim": "Vines improve matched risk-forecast loss over Gaussian copulas",
            "decision": h2.get("status"),
            "decision_rule_or_evidence": "All six Holm-adjusted DM tests must favour the vine",
            "key_estimate": h2.get("vine_favouring_adjusted_tests"),
            "uncertainty_or_eligibility": (
                f"{int(h2['vine_favouring_adjusted_tests'])} of "
                f"{int(h2['required_tests_for_full_support'])} adjusted tests favour vine"
            ),
            "analysis_role": "confirmatory_statistical",
        },
        {
            "hypothesis_id": "H3",
            "research_claim": "The integrated hierarchical-vine model is the best eligible core model",
            "decision": h3.get("status"),
            "decision_rule_or_evidence": "Lowest average primary-score rank with quality eligibility",
            "key_estimate": h3.get("best_eligible_model"),
            "uncertainty_or_eligibility": (
                f"vine_quality_eligible={str(bool(h3['m4_vine_quality_eligible'])).lower()}; "
                f"systematic_calibration_failure="
                f"{str(bool(h3['m4_systematic_calibration_failure'])).lower()}"
            ),
            "analysis_role": "confirmatory_integrated_ranking",
        },
    ]


def _dependence_row(
    *,
    year: int,
    method_id: str,
    label: str,
    gap: Mapping[str, Any],
    gap_minus_gics: float | None,
    ari: float | None,
    nmi: float | None,
) -> dict[str, Any]:
    return {
        "year": year,
        "method_id": method_id,
        "method_label": label,
        "dependence_gap": gap.get("gap"),
        "within_pair_mean": gap.get("within_pair_mean"),
        "between_pair_mean": gap.get("between_pair_mean"),
        "within_pair_count": gap.get("within_pair_count"),
        "between_pair_count": gap.get("between_pair_count"),
        "gap_minus_gics": gap_minus_gics,
        "ari_vs_previous_year": ari,
        "nmi_vs_gics": nmi,
    }


def build_clustering_rows(
    core: Mapping[str, Any],
    ml: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Combine annual core and exploratory clustering diagnostics."""

    core_years = {
        int(row["year"]): _require_mapping(row, "core clustering year")
        for row in _require_sequence(core.get("years"), "core clustering years")
    }
    ml_years = {
        int(row["year"]): _require_mapping(row, "ML clustering year")
        for row in _require_sequence(ml.get("years"), "ML clustering years")
    }
    if set(core_years) != set(ml_years) or len(core_years) != 6:
        raise ValueError("core and ML clustering years must match across six evaluation years")
    rows = []
    for year in sorted(core_years):
        core_year = core_years[year]
        ml_year = ml_years[year]
        gics = _require_mapping(core_year.get("gics_dependence_gap"), "GICS gap")
        hierarchical = _require_mapping(
            core_year.get("hierarchical_dependence_gap"), "hierarchical gap"
        )
        rows.extend(
            [
                _dependence_row(
                    year=year,
                    method_id="gics",
                    label=str(labels["gics"]),
                    gap=gics,
                    gap_minus_gics=0.0,
                    ari=None,
                    nmi=1.0,
                ),
                _dependence_row(
                    year=year,
                    method_id="hierarchical",
                    label=str(labels["hierarchical"]),
                    gap=hierarchical,
                    gap_minus_gics=float(core_year["cluster_minus_gics_pair_weighted_gap"]),
                    ari=core_year.get("ari_vs_previous_year"),
                    nmi=core_year.get("nmi_hierarchical_vs_gics"),
                ),
            ]
        )
        methods = _require_mapping(ml_year.get("methods"), "ML clustering methods")
        for source_id, report_id in (
            ("spectral_cluster", "spectral"),
            ("pca_kmeans_cluster", "pca_kmeans"),
        ):
            method = _require_mapping(methods.get(source_id), source_id)
            rows.append(
                _dependence_row(
                    year=year,
                    method_id=report_id,
                    label=str(labels[report_id]),
                    gap=_require_mapping(method.get("dependence_gap"), f"{source_id} gap"),
                    gap_minus_gics=float(method["gap_minus_gics_pair_weighted"]),
                    ari=method.get("ari_vs_previous_year"),
                    nmi=method.get("nmi_vs_gics"),
                )
            )
    return rows


def build_robustness_rows(sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Summarize pre-authorized robustness analyses without cross-target ranking."""

    full_vine = _require_mapping(sources["full_vine"]["interpretation"], "full-vine result")
    group = _require_mapping(
        sources["group_balanced"]["evaluation"]["interpretation"],
        "group-balanced result",
    )
    ml = _require_mapping(sources["ml_evaluation"]["interpretation"], "ML result")
    current = _require_mapping(
        sources["current_composition"]["interpretation"], "current-composition result"
    )
    wba = _require_sequence(sources["wba_dap"]["conclusion_stability"], "WBA conclusions")
    stable_wba = all(
        row["H2_primary_status"] == row["H2_scenario_status"]
        and row["H3_primary_status"] == row["H3_scenario_status"]
        and row["ml_primary_evidence"] == row["ml_scenario_evidence"]
        for row in (_require_mapping(value, "WBA scenario") for value in wba)
    )
    return [
        {
            "analysis_id": "full_vine_tree_10",
            "analysis_role": "exploratory_robustness",
            "evidence": full_vine["adjusted_loss_evidence"],
            "adjusted_tests_favouring_alternative": full_vine["full_vine_favouring_adjusted_tests"],
            "adjusted_tests_favouring_baseline": full_vine[
                "truncated_vine_favouring_adjusted_tests"
            ],
            "core_conclusions_revised": full_vine["primary_hypotheses_changed"],
            "detail": "Tree 10 versus frozen tree 3; matched information and random numbers",
        },
        {
            "analysis_id": "ml_groupings",
            "analysis_role": sources["ml_evaluation"]["analysis_role"],
            "evidence": ml["evidence"],
            "adjusted_tests_favouring_alternative": ml["tests_favouring_ml_after_bh_fdr"],
            "adjusted_tests_favouring_baseline": ml["tests_favouring_baseline_after_bh_fdr"],
            "core_conclusions_revised": ml["core_hypotheses_and_ranking_revised"],
            "detail": "24 comparisons controlled by Benjamini-Hochberg FDR at 5%",
        },
        {
            "analysis_id": "group_balanced_portfolios",
            "analysis_role": group["analysis_role"],
            "evidence": group["evidence"],
            "adjusted_tests_favouring_alternative": group["tests_favouring_vine_after_holm"],
            "adjusted_tests_favouring_baseline": group["tests_favouring_gaussian_after_holm"],
            "core_conclusions_revised": group["core_H1_H2_H3_revised"],
            "detail": "Within-portfolio comparisons only; no cross-portfolio ranking",
        },
        {
            "analysis_id": "wba_dap_valuation",
            "analysis_role": sources["wba_dap"]["analysis_role"],
            "evidence": "all_frozen_conclusions_stable" if stable_wba else "conclusion_changed",
            "adjusted_tests_favouring_alternative": None,
            "adjusted_tests_favouring_baseline": None,
            "core_conclusions_revised": not stable_wba,
            "detail": f"{len(wba)} scenarios: DAP values $0 and $3 versus primary $0.53",
        },
        {
            "analysis_id": "current_composition_historical_simulation",
            "analysis_role": current["analysis_role"],
            "evidence": current["evidence"],
            "adjusted_tests_favouring_alternative": current[
                "tests_favouring_current_composition_after_holm"
            ],
            "adjusted_tests_favouring_baseline": current[
                "tests_favouring_realised_history_after_holm"
            ],
            "core_conclusions_revised": current["core_H1_H2_H3_revised"],
            "detail": "Alternative constituent-history assumption; no cross-portfolio ranking",
        },
    ]


def _format_cell(value: Any, significant_digits: int) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("report tables cannot contain non-finite values")
        return format(value, f".{significant_digits}g")
    return value


def _write_table(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    significant_digits: int,
) -> None:
    formatted = [
        {field: _format_cell(row.get(field), significant_digits) for field in fields}
        for row in rows
    ]
    write_csv_atomic(path, formatted, fieldnames=fields)


def _configure_plots() -> None:
    matplotlib.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "savefig.transparent": False,
            "svg.hashsalt": "fe5110-report-v1",
        }
    )


def _save_svg(path: Path, draw: Callable[[], Any]) -> None:
    _configure_plots()
    figure = draw()
    with temporary_sibling(path) as temporary:
        figure.savefig(
            temporary,
            format="svg",
            bbox_inches="tight",
            metadata=SVG_METADATA,
        )
        temporary.replace(path)
    plt.close(figure)


def _mark_provisional(axis: Any) -> None:
    axis.text(
        1.0,
        1.01,
        "Provisional research results",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )


def plot_annual_dependence_gaps(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Any],
    palette: Mapping[str, Any],
    path: Path,
) -> None:
    """Plot annual within-minus-between correlation gaps by grouping method."""

    def draw() -> Any:
        figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        for method_id in ("gics", "hierarchical", "spectral", "pca_kmeans"):
            selected = [row for row in rows if row["method_id"] == method_id]
            axis.plot(
                [int(row["year"]) for row in selected],
                [float(row["dependence_gap"]) for row in selected],
                marker="o",
                linewidth=2,
                markersize=4.5,
                color=str(palette[method_id]),
                label=str(labels[method_id]),
            )
        axis.axhline(0.0, color="#999999", linewidth=0.8)
        axis.set_xlabel("Year")
        axis.set_title("Out-of-sample dependence separation", pad=24)
        axis.set_ylabel("Within-group minus between-group correlation")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        axis.legend(frameon=False, ncols=2)
        _mark_provisional(axis)
        return figure

    _save_svg(path, draw)


def plot_core_model_score_ranks(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Plot the frozen average primary-score rank for M0-M4."""

    def draw() -> Any:
        figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        model_labels = [f"{row['model_id']}  {row['model_label']}" for row in rows]
        ranks = [float(row["average_primary_score_rank"]) for row in rows]
        colors = ["#D9D9D9", "#9ECAE1", "#6BAED6", "#FDBE85", "#F58518"]
        bars = axis.barh(model_labels, ranks, color=colors)
        axis.invert_yaxis()
        axis.set_xlim(0, 5.4)
        axis.set_xlabel("Average rank across QL95, QL99 and FZ0 (lower is better)")
        axis.set_title("Core model forecast-loss ranking", pad=24)
        axis.grid(axis="x", color="#E5E5E5", linewidth=0.8)
        for bar, rank in zip(bars, ranks, strict=True):
            axis.text(rank + 0.08, bar.get_y() + bar.get_height() / 2, f"{rank:.2f}", va="center")
        _mark_provisional(axis)
        return figure

    _save_svg(path, draw)


def plot_core_exception_rates(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Plot observed core-model exception rates against nominal rates."""

    def draw() -> Any:
        figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        confidences = [95.0, 97.5, 99.0]
        expected = [5.0, 2.5, 1.0]
        for row in rows:
            observed = [
                100.0 * float(row["exception_rate_95"]),
                100.0 * float(row["exception_rate_975"]),
                100.0 * float(row["exception_rate_99"]),
            ]
            axis.plot(confidences, observed, marker="o", linewidth=1.6, label=str(row["model_id"]))
        axis.plot(
            confidences,
            expected,
            color="#222222",
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label="Nominal",
        )
        axis.set_xticks(confidences, ["95% VaR", "97.5% VaR", "99% VaR"])
        axis.set_ylabel("Observed exception rate (%)")
        axis.set_title("Core model VaR exception rates", pad=24)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        axis.legend(frameon=False, ncols=3)
        _mark_provisional(axis)
        return figure

    _save_svg(path, draw)


def build_results_summary(
    sources: Mapping[str, Any], core_rows: Sequence[Mapping[str, Any]]
) -> str:
    """Build a concise, audit-derived interpretation with explicit boundaries."""

    evaluation = sources["core_evaluation"]
    hypotheses = evaluation["hypotheses"]
    h1 = evaluation["h1_bootstrap"]
    interval = h1["cluster_minus_gics_confidence_interval"]
    h2 = hypotheses["H2_copula_specification"]
    h3 = hypotheses["H3_integrated_model"]
    best = next(row for row in core_rows if row["model_id"] == h3["best_eligible_model"])
    ml = sources["ml_evaluation"]["interpretation"]
    robustness = build_robustness_rows(sources)
    robustness_lines = "\n".join(
        f"- `{row['analysis_id']}`: {str(row['evidence']).replace('_', ' ')}; "
        f"core conclusions revised = {str(bool(row['core_conclusions_revised'])).lower()}."
        for row in robustness
    )
    return f"""# Audited Results Summary

> **Scope:** Provisional research results. The universe is supported by the PA-001 Yahoo-based
> foundation, not by licensed point-in-time S&P/GICS reconciliation. These artifacts summarize
> existing audited outputs and do not introduce new inference or refit any model.

## Core findings

- **H1 — not supported.** The mean hierarchical-minus-GICS dependence-gap difference is
  {float(h1["mean_cluster_minus_gics_gap"]):.6f}; its paired circular block-bootstrap 95% interval
  is [{float(interval["lower"]):.6f}, {float(interval["upper"]):.6f}]. The interval is not strictly
  positive.
- **H2 — no support.** {int(h2["vine_favouring_adjusted_tests"])} of
  {int(h2["required_tests_for_full_support"])} Holm-adjusted matched DM tests favour the vine.
- **H3 — supported under the frozen ranking rule.** `{h3["best_eligible_model"]}` is the unique
  best eligible core model, with average primary-score rank
  {float(best["average_primary_score_rank"]):.3f}. This is a relative forecast-loss ranking, not a
  claim that vines dominate Gaussian copulas in the H2 pairwise tests.

## Exploratory ML extension

Spectral clustering and PCA-plus-k-means do not change the core conclusions. Across the frozen
{int(ml["comparison_family_size"])}-test family, {int(ml["tests_favouring_ml_after_bh_fdr"])}
comparisons favour an ML grouping and {int(ml["tests_favouring_baseline_after_bh_fdr"])} favour a
baseline grouping after Benjamini-Hochberg adjustment.

## Robustness boundary

{robustness_lines}

## Artifact guide

- `tables/core_model_performance.csv` and `core_model_score_ranks.svg` report the frozen M0-M4
  loss ranking.
- `tables/core_dm_tests.csv` and `tables/core_calibration.csv` preserve the adjusted inferential
  results and full-period calibration diagnostics.
- `tables/clustering_diagnostics.csv` and `annual_dependence_gaps.svg` report annual grouping
  separation for the core and exploratory clustering methods.
- `report_manifest.json` binds every table, figure and this summary to the exact audited inputs.
"""


def _artifact_record(project_root: Path, path: Path, **extra: Any) -> dict[str, Any]:
    record = {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    record.update(extra)
    return record


def build_report_artifacts(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Generate all frozen report artifacts and return their lineage manifest."""

    project_root = project_root.resolve()
    config_path = config_path or project_root / "config/reporting_config.toml"
    config_path = config_path if config_path.is_absolute() else project_root / config_path
    config = load_reporting_config(config_path)
    configured_output = output_root or project_root / str(config["output_root"])
    configured_output = (
        configured_output if configured_output.is_absolute() else project_root / configured_output
    ).resolve()
    try:
        configured_output.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("report output root must be inside the project") from exc
    sources = load_audited_sources(project_root, config)
    labels = _require_mapping(config["model_labels"], "model labels")
    method_labels = _require_mapping(config["method_labels"], "method labels")
    palette = _require_mapping(config["palette"], "plot palette")
    digits = int(config["float_significant_digits"])

    core_rows = build_model_performance_rows(
        sources["core_evaluation"], config["core_model_order"], labels, ml=False
    )
    ml_rows = build_model_performance_rows(
        sources["ml_evaluation"], config["ml_model_order"], labels, ml=True
    )
    table_specs = {
        "core_model_performance.csv": (core_rows, CORE_PERFORMANCE_FIELDS),
        "core_dm_tests.csv": (build_dm_rows(sources["core_evaluation"]), DM_FIELDS),
        "core_calibration.csv": (
            build_calibration_rows(sources["core_evaluation"]),
            CALIBRATION_FIELDS,
        ),
        "hypothesis_conclusions.csv": (
            build_hypothesis_rows(sources["core_evaluation"]),
            HYPOTHESIS_FIELDS,
        ),
        "clustering_diagnostics.csv": (
            build_clustering_rows(sources["clustering"], sources["ml_clustering"], method_labels),
            CLUSTERING_FIELDS,
        ),
        "ml_model_performance.csv": (ml_rows, ML_PERFORMANCE_FIELDS),
        "robustness_summary.csv": (build_robustness_rows(sources), ROBUSTNESS_FIELDS),
    }
    configured_tables = list(config["artifacts"]["tables"])
    if set(configured_tables) != set(table_specs):
        raise ValueError("configured table artifacts differ from implemented frozen tables")
    table_records = []
    for filename in configured_tables:
        rows, fields = table_specs[filename]
        path = configured_output / TABLE_DIRECTORY / filename
        _write_table(path, rows, fields, significant_digits=digits)
        table_records.append(
            _artifact_record(project_root, path, rows=len(rows), columns=len(fields))
        )

    clustering_rows = table_specs["clustering_diagnostics.csv"][0]
    figure_specs: dict[str, Callable[[Path], None]] = {
        "annual_dependence_gaps.svg": lambda path: plot_annual_dependence_gaps(
            clustering_rows, method_labels, palette, path
        ),
        "core_model_score_ranks.svg": lambda path: plot_core_model_score_ranks(core_rows, path),
        "core_var_exception_rates.svg": lambda path: plot_core_exception_rates(core_rows, path),
    }
    configured_figures = list(config["artifacts"]["figures"])
    if set(configured_figures) != set(figure_specs):
        raise ValueError("configured figure artifacts differ from implemented frozen figures")
    figure_records = []
    for filename in configured_figures:
        path = configured_output / FIGURE_DIRECTORY / filename
        figure_specs[filename](path)
        figure_records.append(_artifact_record(project_root, path, media_type="image/svg+xml"))

    configured_documents = list(config["artifacts"]["documents"])
    if configured_documents != ["results_summary.md"]:
        raise ValueError("configured document artifacts differ from the frozen document set")
    summary_path = configured_output / configured_documents[0]
    write_text_atomic(summary_path, build_results_summary(sources, core_rows))
    document_records = [_artifact_record(project_root, summary_path, media_type="text/markdown")]

    source_config = _require_mapping(config["sources"], "reporting sources")
    source_records = {
        name: _artifact_record(project_root, _resolve_project_file(project_root, path, name))
        for name, path in sorted(source_config.items())
    }
    manifest = {
        "schema_version": 1,
        "manifest_type": "deterministic_report_artifacts",
        "status": "pass",
        "analysis_role": config["analysis_role"],
        "reporting_scope": config["reporting_scope"],
        "protocol_status": config["protocol_status"],
        "generator": _artifact_record(project_root, Path(__file__)),
        "configuration": _artifact_record(project_root, config_path),
        "rendering_environment": {
            "matplotlib": matplotlib.__version__,
            "backend": str(matplotlib.get_backend()),
            "svg_hashsalt": str(matplotlib.rcParams["svg.hashsalt"]),
        },
        "sources": source_records,
        "outputs": {
            "tables": table_records,
            "figures": figure_records,
            "documents": document_records,
        },
        "quality": {
            "source_gate_count": len(source_records),
            "all_sources_passed": True,
            "all_source_lineage_current": True,
            "new_inference_performed": False,
            "model_refits_performed": False,
        },
    }
    write_json_atomic(configured_output / MANIFEST_NAME, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("config/reporting_config.toml"))
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_report_artifacts(
        args.project_root,
        config_path=args.config,
        output_root=args.output_root,
    )
    output_count = sum(len(records) for records in manifest["outputs"].values())
    print(
        f"report_artifacts={manifest['status']} outputs={output_count} "
        f"scope={manifest['reporting_scope']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
