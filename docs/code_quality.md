# Code Quality and Checkpoint Procedure

This project uses a small, reproducible quality toolchain. Ruff is pinned in the
development dependency group, and the exact environment is locked by `uv.lock`.
The enforced profile covers syntax and Pyflakes errors, import ordering, common
bug patterns, Python 3.12 upgrades, avoidable collection/control-flow patterns,
and stale suppression comments.

Run the checkpoint from the repository root:

```zsh
uv sync --frozen
uv run ruff check scripts tests
uv run ruff check scripts --select C901
uv run ruff check scripts --select S
uv run ruff format --check scripts tests
uv run python -m compileall -q scripts tests
uv run python -m unittest discover -s tests -v
uv run python scripts/update_foundation_status.py --require-modelling-ready
uv run python scripts/build_report_artifacts.py
uv run python scripts/generate_run_manifest.py --require-complete
uv run python scripts/verify_artifact_lineage.py
git status --short
```

The equivalent single-program release-candidate path is:

```zsh
uv run python scripts/run_pipeline.py --profile validate
```

The orchestrator is intentionally thin: it invokes the same fail-closed stage
programs and does not duplicate numerical logic or suppress a failed subprocess.

The checkpoint is acceptable only when linting, formatting, compilation and all
tests pass; the aggregate gate reports either strict or explicitly provisional
modelling readiness; the run manifest is complete; and Git has no uncommitted
tracked changes after the manifest is committed.

## Automated clean-clone checks

GitHub Actions runs the portable lint, security, format, compilation and unit-test
checks on every push to `main` and on every pull request under both supported
Python 3.12 and 3.13 versions. The workflow installs the pinned uv version, then
reconstructs each environment from `uv.lock` with `uv sync --frozen`. Ruff's
cyclomatic-complexity threshold is enforced across every module under `scripts/`.

Thirteen acceptance-test classes bind the checked-in audit records to large generated
Parquet artifacts. Those files are intentionally excluded from Git, so the
classes report as skipped in a clean public clone. They run automatically when
the artifacts exist locally. One additional report-reproducibility test skips
when its ignored upstream artifacts are absent, giving 14 expected skips in a
source-only clone. The full checkpoint above additionally requires
`foundation_v2` modelling readiness and therefore cannot pass merely because
the artifact-bound tests were skipped. Artifact-bound guards classify the full
recorded lineage: only missing generated files under `data/processed/` may
skip. A missing tracked input or audit, stale digest, malformed record,
external path, or mixture of missing and invalid records must run and fail the
corresponding acceptance check.

The lineage verifier recursively inspects every JSON object carrying both a
`path` and `sha256` under `data/audit/`, `data/manifests/`, and the report
manifest. Missing files, invalid digests, or stale bindings fail the checkpoint.
Run-manifest generation performs the same check while excluding the manifest
being replaced, and records the result in its completeness decision.

## Repository boundaries

- Commit code, tests, configuration, documentation, sanitized manifests and
  public audit summaries.
- Do not commit credentials, `.env` files, licensed raw data, provider market
  payloads, raw Yahoo responses or large derived files under `data/processed/`.
- Bind ignored derived outputs to tracked audits/manifests with SHA-256 hashes.
- Write generated JSON and Parquet outputs atomically so an interrupted run does
  not replace a valid artifact with a partial file.
- Reuse `scripts/pipeline_io.py` for modelling-stage paths, hashes, reporting
  scope checks and atomic writes. Its temporary files are unique per writer and
  are removed after either success or failure.
- Catch only expected numerical and solver exceptions at model-fallback
  boundaries. Programming defects must propagate so tests and CI expose them.

## Architecture map for reviewers

The repository is organized as a fail-closed sequence rather than one monolithic
program:

1. `prepare_universe.py`, `download_market_data.py`, and
   `verify_foundation_inputs.py` establish source provenance and input integrity.
2. `build_return_panel.py` applies lifecycle and corporate-action rules;
   `validate_portfolio_arithmetic.py` independently checks the resulting return
   identities.
3. `build_annual_groupings.py` and `build_ml_groupings.py` create leakage-free
   annual assignments and group-return panels.
4. `build_marginal_models.py`, `build_gaussian_copula.py`, and
   `build_vine_copula.py` fit the monthly risk models. Validation, fitting,
   forecasting, fallback handling, and quality summaries are separate helpers.
5. The `evaluate_*` modules perform frozen inference and robustness comparisons;
   `build_report_artifacts.py` only renders passed, hash-bound results.
6. `run_pipeline.py` supplies the single fail-fast entry point, while
   `pipeline_io.py` centralizes safe paths, hashes, reporting scope, and atomic
   writes.

Every stage validates its own input schema and upstream bindings. Expected
numerical failures can invoke an explicitly recorded fallback; programming
errors propagate. Large numerical outputs stay untracked but are bound to public
audit records by SHA-256.

## Maintainability boundary

All pipeline functions now pass the repository-wide Ruff `C901` threshold. The
former long ingestion, lifecycle, marginal, Gaussian, and vine orchestration
functions were decomposed into focused validation, transformation, forecasting,
and quality-gate helpers under their existing characterization tests. This was a
structural cleanup only: future refactors must continue to preserve numerical
outputs and artifact schemas unless a protocol change is explicitly authorized.
