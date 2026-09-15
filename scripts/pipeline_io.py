"""Shared filesystem and reporting-scope helpers for modelling pipelines."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
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


def artifact_hash_issues(
    payload: object,
    *,
    project_root: Path = PROJECT_ROOT,
    source_name: str = "payload",
) -> list[str]:
    """Return missing or stale ``path``/``sha256`` record issues in a payload."""

    issues: list[str] = []

    def visit(value: object, location: str) -> None:
        if isinstance(value, Mapping):
            if "path" in value and "sha256" in value:
                recorded_path = value["path"]
                recorded_hash = value["sha256"]
                if not isinstance(recorded_path, str) or not recorded_path:
                    issues.append(f"{source_name}:{location}:invalid_path")
                elif (
                    not isinstance(recorded_hash, str)
                    or len(recorded_hash) != 64
                    or any(character not in "0123456789abcdef" for character in recorded_hash)
                ):
                    issues.append(f"{source_name}:{location}:invalid_sha256")
                else:
                    target = Path(recorded_path)
                    if not target.is_absolute():
                        target = project_root / target
                    if not target.is_file():
                        issues.append(f"{source_name}:{location}:missing:{recorded_path}")
                    elif sha256_file(target) != recorded_hash:
                        issues.append(f"{source_name}:{location}:stale:{recorded_path}")
            for key, child in value.items():
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    visit(payload, "$")
    return issues


def require_current_hash_records(
    payload: object,
    *,
    project_root: Path = PROJECT_ROOT,
    source_name: str = "payload",
) -> None:
    """Fail when any recorded artifact path is missing or has a stale digest."""

    issues = artifact_hash_issues(
        payload,
        project_root=project_root,
        source_name=source_name,
    )
    if issues:
        raise RuntimeError("artifact lineage check failed: " + "; ".join(issues))


def project_path(path: Path) -> str:
    """Return a portable project-relative path when possible."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON artifact."""

    write_text_atomic(
        path,
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace a text artifact using a unique sibling temporary file."""

    with temporary_sibling(path) as temporary:
        temporary.write_text(content, encoding=encoding)
        temporary.replace(path)


def write_csv_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    encoding: str = "utf-8",
) -> None:
    """Atomically replace a headered CSV artifact from mapping records."""

    with temporary_sibling(path) as temporary:
        with temporary.open("w", encoding=encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def write_parquet_atomic(
    path: Path,
    frame: pd.DataFrame,
    *,
    index: bool = False,
    compression: str = "snappy",
) -> None:
    """Atomically replace a Parquet artifact."""

    with temporary_sibling(path) as temporary:
        frame.to_parquet(
            temporary,
            index=index,
            compression=compression,
            engine="pyarrow",
        )
        temporary.replace(path)


@contextmanager
def temporary_sibling(destination: Path) -> Iterator[Path]:
    """Yield a unique sibling path and remove it unless it was atomically moved."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        yield temporary
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
