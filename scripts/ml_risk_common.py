"""Shared constants and artifact checks for the exploratory M5--M8 pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

try:
    from scripts.pipeline_io import project_path, sha256_file
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import project_path, sha256_file


ML_RISK_ANALYSIS_ROLE = "exploratory_unsupervised_risk_extension"
ML_INPUT_GROUPINGS = ("spectral_cluster", "pca_kmeans_cluster")
ML_GAUSSIAN_MODEL_BY_GROUPING = {
    "spectral_cluster": ("M5", "spectral"),
    "pca_kmeans_cluster": ("M7", "pca_kmeans"),
}
ML_VINE_MODEL_BY_GROUPING = {
    "spectral_cluster": ("M6", "spectral", "M5"),
    "pca_kmeans_cluster": ("M8", "pca_kmeans", "M7"),
}
ML_MODEL_GROUPINGS = {
    "M5": "spectral",
    "M6": "spectral",
    "M7": "pca_kmeans",
    "M8": "pca_kmeans",
}
ML_MODEL_IDS = tuple(ML_MODEL_GROUPINGS)


def mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return a mapping or fail with a field-specific error."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def load_json(path: Path) -> Mapping[str, Any]:
    """Load one JSON object with an explicit type check."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return mapping(payload, project_path(path))


def artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    """Return a portable hash-bound artifact record."""

    record: dict[str, Any] = {
        "path": project_path(path),
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def require_audit_output(
    audit: Mapping[str, Any],
    *,
    audit_name: str,
    output_key: str,
    output_path: Path,
    reporting_scope: str,
    analysis_role: str | None = None,
) -> None:
    """Require a passed, scope-matched audit bound to one current output."""

    if audit.get("status") != "pass" or audit.get("reporting_scope") != reporting_scope:
        raise RuntimeError(f"{audit_name} is not a passed, scope-matched audit")
    if analysis_role is not None and audit.get("analysis_role") != analysis_role:
        raise RuntimeError(f"{audit_name} has an incompatible analysis role")
    outputs = mapping(audit.get("outputs"), f"{audit_name}.outputs")
    record = mapping(outputs.get(output_key), f"{audit_name}.{output_key}")
    if record.get("sha256") != sha256_file(output_path):
        raise RuntimeError(f"{audit_name} does not bind the current {output_key}")


def require_same_seed_manifest(generated: Mapping[str, Any], primary: Mapping[str, Any]) -> None:
    """Require ML simulations to reuse the exact primary monthly uniform draws."""

    if generated != primary:
        raise RuntimeError("ML Gaussian simulation did not reproduce the primary seed manifest")
