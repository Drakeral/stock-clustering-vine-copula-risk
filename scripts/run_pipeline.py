"""Run the FE5110 research pipeline through one fail-fast entry point.

The individual stage programs remain independently executable and testable.  This
orchestrator supplies the single main Python program requested by the project brief
while preserving each stage's existing command-line boundary and audit outputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
COMPLEXITY_MODULES: Final[tuple[str, ...]] = (
    "scripts/evaluate_risk_models.py",
    "scripts/evaluate_full_vine_robustness.py",
    "scripts/evaluate_ml_risk_models.py",
    "scripts/evaluate_group_balanced_robustness.py",
    "scripts/evaluate_current_composition_historical_simulation.py",
    "scripts/build_group_balanced_portfolios.py",
    "scripts/build_report_artifacts.py",
    "scripts/download_market_data.py",
    "scripts/pipeline_io.py",
    "scripts/run_pipeline.py",
)


@dataclass(frozen=True)
class Stage:
    """One deterministic pipeline command."""

    name: str
    description: str
    arguments: tuple[str, ...]

    def command(self, python_executable: str = sys.executable) -> tuple[str, ...]:
        return (python_executable, *self.arguments)


REPRODUCE_STAGES: Final[tuple[Stage, ...]] = (
    Stage(
        "verify-inputs",
        "Verify hashes, schemas, rows, and XNYS coverage",
        ("scripts/verify_foundation_inputs.py",),
    ),
    Stage(
        "build-returns",
        "Construct lifecycle-aware total-return panels",
        ("scripts/build_return_panel.py",),
    ),
    Stage(
        "validate-portfolio",
        "Verify stock/group portfolio arithmetic",
        ("scripts/validate_portfolio_arithmetic.py",),
    ),
    Stage(
        "foundation-gate",
        "Require strict or authorised provisional readiness",
        ("scripts/update_foundation_status.py", "--require-modelling-ready"),
    ),
    Stage(
        "historical-simulation",
        "Build the M0 forecast baseline",
        ("scripts/build_historical_simulation.py",),
    ),
    Stage(
        "current-composition",
        "Evaluate the historical-composition robustness",
        ("scripts/evaluate_current_composition_historical_simulation.py",),
    ),
    Stage(
        "core-groupings",
        "Build annual GICS and hierarchical groupings",
        ("scripts/build_annual_groupings.py",),
    ),
    Stage(
        "core-margins", "Fit monthly core marginal models", ("scripts/build_marginal_models.py",)
    ),
    Stage(
        "core-gaussian",
        "Fit and simulate M1/M3 Gaussian copulas",
        ("scripts/build_gaussian_copula.py",),
    ),
    Stage("core-vine", "Fit and simulate M2/M4 tree-3 vines", ("scripts/build_vine_copula.py",)),
    Stage(
        "core-evaluation",
        "Evaluate M0-M4 and frozen hypotheses",
        ("scripts/evaluate_risk_models.py",),
    ),
    Stage(
        "full-vine",
        "Fit the tree-10 vine robustness",
        ("scripts/build_vine_copula.py", "--truncation-level", "10"),
    ),
    Stage(
        "full-vine-evaluation",
        "Evaluate tree-3 against tree-10 vines",
        ("scripts/evaluate_full_vine_robustness.py",),
    ),
    Stage(
        "ml-groupings",
        "Build spectral and PCA-plus-k-means groupings",
        ("scripts/build_ml_groupings.py",),
    ),
    Stage(
        "ml-margins", "Fit exploratory ML marginal models", ("scripts/build_ml_marginal_models.py",)
    ),
    Stage(
        "ml-gaussian",
        "Fit and simulate M5/M7 Gaussian copulas",
        ("scripts/build_ml_gaussian_copula.py",),
    ),
    Stage("ml-vine", "Fit and simulate M6/M8 tree-3 vines", ("scripts/build_ml_vine_copula.py",)),
    Stage(
        "ml-evaluation",
        "Evaluate the exploratory ML models",
        ("scripts/evaluate_ml_risk_models.py",),
    ),
    Stage(
        "wba-sensitivity",
        "Evaluate WBA DAP valuation endpoints",
        ("scripts/evaluate_wba_dap_sensitivity.py",),
    ),
    Stage(
        "balanced-portfolios",
        "Build equal-group portfolio robustness inputs",
        ("scripts/build_group_balanced_portfolios.py",),
    ),
    Stage(
        "balanced-evaluation",
        "Evaluate equal-group portfolio robustness",
        ("scripts/evaluate_group_balanced_robustness.py",),
    ),
    Stage(
        "report-artifacts",
        "Rebuild deterministic report tables and figures",
        ("scripts/build_report_artifacts.py",),
    ),
    Stage(
        "run-manifest",
        "Capture the complete run identity",
        ("scripts/generate_run_manifest.py", "--require-complete"),
    ),
    Stage(
        "lineage", "Verify every tracked artifact binding", ("scripts/verify_artifact_lineage.py",)
    ),
)


VALIDATE_STAGES: Final[tuple[Stage, ...]] = (
    Stage(
        "lint", "Run the maintained Ruff lint profile", ("-m", "ruff", "check", "scripts", "tests")
    ),
    Stage(
        "complexity",
        "Enforce complexity limits on maintained orchestration modules",
        ("-m", "ruff", "check", *COMPLEXITY_MODULES, "--select", "C901"),
    ),
    Stage(
        "security",
        "Run the Ruff security rules over all pipeline code",
        ("-m", "ruff", "check", "scripts", "--select", "S"),
    ),
    Stage(
        "format",
        "Check deterministic Python formatting",
        ("-m", "ruff", "format", "--check", "scripts", "tests"),
    ),
    Stage("compile", "Compile all Python sources", ("-m", "compileall", "-q", "scripts", "tests")),
    Stage(
        "tests",
        "Run the portable and artifact-bound test suite",
        ("-m", "unittest", "discover", "-s", "tests", "-v"),
    ),
    Stage(
        "foundation-gate",
        "Require strict or authorised provisional readiness",
        ("scripts/update_foundation_status.py", "--require-modelling-ready"),
    ),
    Stage(
        "report-artifacts",
        "Rebuild deterministic report tables and figures",
        ("scripts/build_report_artifacts.py",),
    ),
    Stage(
        "run-manifest",
        "Capture the complete run identity",
        ("scripts/generate_run_manifest.py", "--require-complete"),
    ),
    Stage(
        "lineage", "Verify every tracked artifact binding", ("scripts/verify_artifact_lineage.py",)
    ),
)


PROFILES: Final = {"reproduce": REPRODUCE_STAGES, "validate": VALIDATE_STAGES}


def select_stages(
    stages: Sequence[Stage], start: str | None = None, stop: str | None = None
) -> tuple[Stage, ...]:
    """Return an inclusive, ordered slice of stages and reject ambiguous bounds."""

    names = [stage.name for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("Pipeline stage names must be unique within a profile")
    if start is not None and start not in names:
        raise ValueError(f"Unknown start stage {start!r}; choose from {names}")
    if stop is not None and stop not in names:
        raise ValueError(f"Unknown stop stage {stop!r}; choose from {names}")
    first = names.index(start) if start is not None else 0
    last = names.index(stop) if stop is not None else len(stages) - 1
    if first > last:
        raise ValueError(f"Start stage {start!r} occurs after stop stage {stop!r}")
    return tuple(stages[first : last + 1])


def run_stages(stages: Sequence[Stage], *, dry_run: bool = False) -> None:
    """Execute stages in order, stopping immediately on the first failure."""

    total = len(stages)
    for position, stage in enumerate(stages, start=1):
        command = stage.command()
        rendered = " ".join(command)
        print(f"[{position:02d}/{total:02d}] {stage.name}: {stage.description}", flush=True)
        print(f"          {rendered}", flush=True)
        if dry_run:
            continue
        started = time.monotonic()
        # Every argument is selected from the immutable stage registry above.  No shell is used.
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)  # noqa: S603
        elapsed = time.monotonic() - started
        print(f"          completed in {elapsed:.1f}s", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="validate",
        help="validate checks current artifacts; reproduce rebuilds from verified local inputs",
    )
    parser.add_argument("--from-stage", help="Begin at this stage, inclusive")
    parser.add_argument("--through-stage", help="Stop after this stage, inclusive")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--list-stages", action="store_true", help="List the selected profile")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = PROFILES[args.profile]
    if args.list_stages:
        for stage in profile:
            print(f"{stage.name:24} {stage.description}")
        return 0
    try:
        stages = select_stages(profile, args.from_stage, args.through_stage)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_stages(stages, dry_run=args.dry_run)
    status = "dry_run" if args.dry_run else "pass"
    print(f"pipeline={args.profile} status={status} stages={len(stages)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
