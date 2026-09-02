#!/usr/bin/env python3
"""Capture code, environment, configuration, input, output, and seed identity."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(project_root: Path, path: Path) -> dict[str, Any]:
    path = path if path.is_absolute() else project_root / path
    try:
        portable = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Run-manifest file must be inside the project: {path}") from exc
    if not path.is_file():
        return {"path": portable, "status": "missing"}
    return {
        "path": portable,
        "status": "present",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def file_records(project_root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = [file_record(project_root, path) for path in paths]
    return sorted(records, key=lambda record: record["path"])


def git_state(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )

    inside = run("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0:
        return {
            "repository_initialized": False,
            "commit_present": False,
            "commit": None,
            "worktree_clean": None,
        }
    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "repository_initialized": True,
        "commit_present": head.returncode == 0,
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "worktree_clean": not bool(status.stdout.strip()),
    }


def direct_dependency_names(pyproject_path: Path) -> list[str]:
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    names = []
    for requirement in project.get("dependencies", []):
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if match:
            names.append(match.group(0))
    return sorted(set(names), key=str.lower)


def installed_dependency_versions(pyproject_path: Path) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in direct_dependency_names(pyproject_path):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def default_config_paths(project_root: Path) -> list[Path]:
    """Return every machine-readable configuration file, including nested schemas."""

    return sorted(
        (
            path.relative_to(project_root)
            for path in (project_root / "config").rglob("*")
            if path.is_file()
        ),
        key=lambda path: path.as_posix(),
    )


def default_input_paths() -> list[Path]:
    return [
        Path("data/raw/universe_sp100_2020-01-02.json"),
        Path("data/manifests/universe_source_manifest.json"),
        Path("data/manifests/yahoo_universe_reference_manifest.json"),
        Path("data/manifests/market_input_manifest.json"),
    ]


def default_output_paths() -> list[Path]:
    return [
        Path("data/audit/universe_provenance.json"),
        Path("data/audit/market_input_integrity.json"),
        Path("data/audit/data_quality_report.json"),
        Path("data/audit/portfolio_arithmetic.json"),
        Path("data/audit/current_gate_status.json"),
        Path("data/audit/clustering_diagnostics.json"),
        Path("data/processed/security_daily.parquet"),
        Path("data/processed/daily_total_returns.parquet"),
        Path("data/processed/daily_simple_total_returns.parquet"),
        Path("data/processed/portfolio_constituent_returns.parquet"),
        Path("data/processed/portfolio_constituent_simple_returns.parquet"),
        Path("data/processed/final_universe.json"),
        Path("data/processed/active_universe_by_year.json"),
        Path("data/processed/annual_group_assignments.json"),
        Path("data/processed/annual_group_returns.parquet"),
    ]


def build_run_manifest(
    project_root: Path,
    config_paths: Iterable[Path],
    input_paths: Iterable[Path],
    output_paths: Iterable[Path],
    base_seed: int = 5110,
) -> dict[str, Any]:
    pyproject_path = project_root / "pyproject.toml"
    lock_paths = [Path(".python-version"), Path("pyproject.toml"), Path("uv.lock")]
    configs = file_records(project_root, config_paths)
    inputs = file_records(project_root, input_paths)
    outputs = file_records(project_root, output_paths)
    git = git_state(project_root)
    configuration_complete = bool(configs) and all(item["status"] == "present" for item in configs)
    inputs_complete = bool(inputs) and all(item["status"] == "present" for item in inputs)
    outputs_complete = bool(outputs) and all(item["status"] == "present" for item in outputs)
    git_complete = bool(git.get("commit_present") and git.get("worktree_clean"))
    model_config_path = project_root / "config/model_config.toml"
    if model_config_path.is_file():
        with model_config_path.open("rb") as handle:
            frozen_seed = int(tomllib.load(handle)["simulation"]["base_seed"])
        if base_seed != frozen_seed:
            raise ValueError(
                f"Run-manifest base seed {base_seed} differs from frozen seed {frozen_seed}"
            )
    gate_path = project_root / "data/audit/current_gate_status.json"
    if gate_path.is_file():
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
        foundation = gate_payload.get("foundation_v2", {})
        foundation_status = {
            "status": foundation.get("status", "unknown"),
            "blockers": foundation.get("blockers", []),
            "run_scope": foundation.get(
                "reporting_scope",
                "confirmatory"
                if foundation.get("status") == "pass"
                else "provisional_diagnostics_only",
            ),
        }
    else:
        foundation_status = {
            "status": "unknown",
            "blockers": ["current_gate_status_absent"],
            "run_scope": "provisional_diagnostics_only",
        }
    return {
        "schema_version": 1,
        "manifest_type": "research_run_manifest",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git,
        "foundation_v2": foundation_status,
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "direct_dependency_versions": installed_dependency_versions(pyproject_path),
        },
        "locks": file_records(project_root, lock_paths),
        "configuration": configs,
        "inputs": inputs,
        "outputs": outputs,
        "seed_configuration": {
            "base_seed": base_seed,
            "bit_generator": "PCG64DXSM",
            "monthly_seed_sequence": "SeedSequence([base_seed, year, month])",
            "bootstrap_seed_sequence": "SeedSequence([base_seed, 1])",
            "common_random_numbers": True,
        },
        "completeness": {
            "git_commit_and_clean_worktree_at_capture": git_complete,
            "configuration": configuration_complete,
            "inputs": inputs_complete,
            "outputs": outputs_complete,
            "complete": bool(
                git_complete and configuration_complete and inputs_complete and outputs_complete
            ),
        },
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument("--base-seed", type=int, default=5110)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/run_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    configs = args.config or default_config_paths(project_root)
    inputs = args.input or default_input_paths()
    outputs = list(dict.fromkeys([*default_output_paths(), *args.result]))
    manifest = build_run_manifest(project_root, configs, inputs, outputs, args.base_seed)
    output = args.output if args.output.is_absolute() else project_root / args.output
    write_json_atomic(output, manifest)
    print(f"Run manifest written: {output}")


if __name__ == "__main__":
    main()
