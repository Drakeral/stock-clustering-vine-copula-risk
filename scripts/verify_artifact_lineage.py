#!/usr/bin/env python3
"""Verify every path-bound SHA-256 record in audit and manifest JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.pipeline_io import PROJECT_ROOT, artifact_hash_issues
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import PROJECT_ROOT, artifact_hash_issues


def lineage_documents(project_root: Path) -> list[Path]:
    """Return the repository audit and manifest JSON documents to inspect."""

    paths: list[Path] = []
    for directory in (project_root / "data/audit", project_root / "data/manifests"):
        if directory.is_dir():
            paths.extend(directory.glob("*.json"))
    report_manifest = project_root / "reports/report_manifest.json"
    if report_manifest.is_file():
        paths.append(report_manifest)
    return sorted(paths)


def verify_repository_lineage(
    project_root: Path,
    *,
    exclude_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Inspect all JSON hash records and return a deterministic summary."""

    excluded = {path.resolve() for path in (exclude_paths or set())}
    documents = [path for path in lineage_documents(project_root) if path.resolve() not in excluded]
    issues: list[str] = []
    hash_record_count = 0
    for path in documents:
        relative = path.relative_to(project_root).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{relative}:invalid_json:{type(exc).__name__}")
            continue
        document_issues = artifact_hash_issues(
            payload,
            project_root=project_root,
            source_name=relative,
        )
        issues.extend(document_issues)

        def count_records(value: object) -> int:
            if isinstance(value, dict):
                own = int("path" in value and "sha256" in value)
                return own + sum(count_records(child) for child in value.values())
            if isinstance(value, list):
                return sum(count_records(child) for child in value)
            return 0

        hash_record_count += count_records(payload)
    return {
        "status": "pass" if not issues else "fail",
        "document_count": len(documents),
        "hash_record_count": hash_record_count,
        "issues": issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    excluded = {
        path.resolve() if path.is_absolute() else (project_root / path).resolve()
        for path in args.exclude
    }
    result = verify_repository_lineage(project_root, exclude_paths=excluded)
    print(
        f"artifact_lineage={result['status']} documents={result['document_count']} "
        f"hash_records={result['hash_record_count']} issues={len(result['issues'])}"
    )
    for issue in result["issues"]:
        print(issue)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
