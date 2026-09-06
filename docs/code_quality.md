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

## Repository boundaries

- Commit code, tests, configuration, documentation, sanitized manifests and
  public audit summaries.
- Do not commit credentials, `.env` files, licensed raw data, provider market
  payloads, raw Yahoo responses or large derived files under `data/processed/`.
- Bind ignored derived outputs to tracked audits/manifests with SHA-256 hashes.
- Write generated JSON and Parquet outputs atomically so an interrupted run does
  not replace a valid artifact with a partial file.

## Current bounded technical debt

Some mature ingestion and return-construction functions are long because they
encode tightly coupled lifecycle, corporate-action and audit rules. They are
covered by targeted acceptance tests and should be decomposed only alongside
new behavioural tests, not as an unrelated formatting exercise. New modelling
modules should prefer small pure functions, explicit input validation and typed
structured records.
