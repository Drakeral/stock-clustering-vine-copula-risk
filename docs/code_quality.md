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
uv run ruff format --check scripts tests
uv run python -m compileall -q scripts tests
uv run python -m unittest discover -s tests -v
uv run python scripts/update_foundation_status.py --require-modelling-ready
uv run python scripts/generate_run_manifest.py
git status --short
```

The checkpoint is acceptable only when linting, formatting, compilation and all
tests pass; the aggregate gate reports either strict or explicitly provisional
modelling readiness; the run manifest is complete; and Git has no uncommitted
tracked changes after the manifest is committed.

## Automated clean-clone checks

GitHub Actions runs the portable lint, format, compilation and unit-test checks
on every push to `main` and on every pull request. The workflow installs the
pinned uv version, then reconstructs the environment from `uv.lock` with
`uv sync --frozen`.

Six acceptance-test classes bind the checked-in audit records to large generated
Parquet artifacts. Those files are intentionally excluded from Git, so the
classes report as skipped in a clean public clone. They run automatically when
the artifacts exist locally. The full checkpoint above additionally requires
`foundation_v2` modelling readiness and therefore cannot pass merely because
the artifact-bound tests were skipped.

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

## Current bounded technical debt

Some mature ingestion and return-construction functions are long because they
encode tightly coupled lifecycle, corporate-action and audit rules. They are
covered by targeted acceptance tests and should be decomposed only alongside
new behavioural tests, not as an unrelated formatting exercise. New modelling
modules should prefer small pure functions, explicit input validation and typed
structured records.
