"""Shared filesystem and reporting-scope helpers for modelling pipelines."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(path: Path) -> str:
    """Return a portable project-relative path when possible."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Atomically replace a Parquet artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_path(destination: Path) -> Path:
    """Reserve a unique sibling path for one atomic artifact write."""

    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
    )
    os.fchmod(descriptor, 0o644)
    os.close(descriptor)
    return Path(name)


def reporting_scope(gate: Mapping[str, Any]) -> str:
    """Return the approved reporting scope after checking modelling readiness."""

    foundation = gate.get("foundation_v2")
    gates = gate.get("gates")
    if not isinstance(foundation, Mapping) or not isinstance(gates, Mapping):
        raise RuntimeError("foundation status is missing required gate tables")
    readiness = gates.get("modelling_readiness")
    if not isinstance(readiness, Mapping):
        raise RuntimeError("foundation status is missing modelling readiness")
    if foundation.get("status") not in {"pass", "provisional_pass"}:
        raise RuntimeError("foundation_v2 is not ready for modelling")
    if readiness.get("status") not in {"pass", "pass_provisional"}:
        raise RuntimeError("modelling-readiness gate is not passed")
    if (foundation.get("status"), readiness.get("status")) not in {
        ("pass", "pass"),
        ("provisional_pass", "pass_provisional"),
    }:
        raise RuntimeError("foundation and modelling-readiness statuses are inconsistent")
    scope = foundation.get("reporting_scope")
    if scope not in {"confirmatory", "provisional_research_results"}:
        raise RuntimeError(f"unsupported reporting scope: {scope}")
    return str(scope)
